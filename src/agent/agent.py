"""Module that defines the base class for agents.

エージェントの基底クラスを定義するモジュール.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from dotenv import load_dotenv
from jinja2 import Template
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

from aiwolf_nlp_common.packet import Info, Packet, Request, Role, Setting, Status, Talk

from utils.agent_logger import AgentLogger
from utils.stoppable_thread import StoppableThread

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
T = TypeVar("T")


class Agent:
    """Base class for agents.

    エージェントの基底クラス.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,
    ) -> None:
        """Initialize the agent.

        エージェントの初期化を行う.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role / 役職
        """
        self.config = config
        self.agent_name = name
        self.agent_logger = AgentLogger(config, name, game_id)
        self.request: Request | None = None
        self.info: Info | None = None
        self.setting: Setting | None = None
        self.talk_history: list[Talk] = []
        self.whisper_history: list[Talk] = []
        self.role = role

        self.sent_talk_count: int = 0
        self.sent_whisper_count: int = 0
        self.llm_model: BaseChatModel | None = None

        load_dotenv(Path(__file__).parent.joinpath("./../../config/.env"))

    @staticmethod
    def timeout(func: Callable[P, T]) -> Callable[P, T]:
        """Decorator to set action timeout.

        アクションタイムアウトを設定するデコレータ.

        Args:
            func (Callable[P, T]): Function to be decorated / デコレート対象の関数

        Returns:
            Callable[P, T]: Function with timeout functionality / タイムアウト機能を追加した関数
        """

        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            res: T | Exception = Exception("No result")

            def execute_with_timeout() -> None:
                nonlocal res
                try:
                    res = func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    res = e

            thread = StoppableThread(target=execute_with_timeout)
            thread.start()
            self = args[0] if args else None
            if not isinstance(self, Agent):
                raise TypeError(self, " is not an Agent instance")
            timeout_value = (self.setting.timeout.action if hasattr(self, "setting") and self.setting else 0) // 1000
            if timeout_value > 0:
                thread.join(timeout=timeout_value)
                if thread.is_alive():
                    self.agent_logger.logger.warning(
                        "アクションがタイムアウトしました: %s",
                        self.request,
                    )
                    if bool(self.config["agent"]["kill_on_timeout"]):
                        thread.stop()
                        self.agent_logger.logger.warning(
                            "アクションを強制終了しました: %s",
                            self.request,
                        )
            else:
                thread.join()
            if isinstance(res, Exception):  # type: ignore[arg-type]
                raise res
            return res

        return _wrapper

    def set_packet(self, packet: Packet) -> None:
        """Set packet information.

        パケット情報をセットする.

        Args:
            packet (Packet): Received packet / 受信したパケット
        """
        self.request = packet.request
        if packet.info:
            self.info = packet.info
        if packet.setting:
            self.setting = packet.setting
        if packet.talk_history:
            self.talk_history.extend(packet.talk_history)
        if packet.whisper_history:
            self.whisper_history.extend(packet.whisper_history)
        if self.request == Request.INITIALIZE:
            self.talk_history: list[Talk] = []
            self.whisper_history: list[Talk] = []
        self.agent_logger.logger.debug(packet)

    def get_alive_agents(self) -> list[str]:
        """Get the list of alive agents.

        生存しているエージェントのリストを取得する.

        Returns:
            list[str]: List of alive agent names / 生存エージェント名のリスト
        """
        if not self.info:
            return []
        return [k for k, v in self.info.status_map.items() if v == Status.ALIVE]

    def _build_profiles_for_template(self, profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build profiles list for template rendering.

        テンプレートレンダリング用のプロファイルリストを作成する.
        自分自身のプロファイルには役職を表示し、他のエージェントは「役職不明」とする.

        Args:
            profiles (list[dict[str, Any]]): List of all profiles / 全プロファイルのリスト

        Returns:
            list[dict[str, Any]]: Filtered profiles for current game participants /
                                    現在のゲーム参加者用にフィルタリングされたプロファイル
        """
        if not self.info or not self.info.status_map:
            return []

        agent_names = list(self.info.status_map.keys())
        profiles_dict = {p["name"]: p for p in profiles}
        result = []

        for agent_name in agent_names:
            if agent_name in profiles_dict:
                profile = profiles_dict[agent_name].copy()
                if agent_name == self.info.agent:
                    profile["role"] = self.role.value if self.role else "役職不明"
                else:
                    profile["role"] = "役職不明"
                result.append(profile)

        return result

    def _send_message_to_llm(self, request: Request | None) -> str | None:
        """Send message to LLM and get response.

        LLMにメッセージを送信して応答を取得する.

        Args:
            request (Request | None): The request type to process / 処理するリクエストタイプ

        Returns:
            str | None: LLM response or None if error occurred / LLMの応答またはエラー時はNone
        """
        if request is None:
            return None
        if request.lower() not in self.config["prompt"]:
            return None
        prompt = self.config["prompt"][request.lower()]
        if float(self.config["llm"]["sleep_time"]) > 0:
            sleep(float(self.config["llm"]["sleep_time"]))
        profiles = self.config.get("profiles", [])
        profiles_for_template = self._build_profiles_for_template(profiles)
        key = {
            "info": self.info,
            "setting": self.setting,
            "talk_history": self.talk_history,
            "whisper_history": self.whisper_history,
            "role": self.role,
            "sent_talk_count": self.sent_talk_count,
            "sent_whisper_count": self.sent_whisper_count,
            "profiles": profiles_for_template,
        }
        template: Template = Template(prompt)
        prompt = template.render(**key).strip()
        if self.llm_model is None:
            self.agent_logger.logger.error("LLM is not initialized")
            return None
        try:
            response = (self.llm_model | StrOutputParser()).invoke([HumanMessage(content=prompt)])
            self.agent_logger.logger.info(["LLM", prompt, response])
        except Exception:
            self.agent_logger.logger.exception("Failed to send message to LLM")
            return None
        else:
            return response

    @timeout
    def name(self) -> str:
        """Return response to name request.

        名前リクエストに対する応答を返す.

        Returns:
            str: Agent name / エージェント名
        """
        return self.agent_name

    def initialize(self) -> None:
        """Perform initialization for game start request.

        ゲーム開始リクエストに対する初期化処理を行う.
        """
        if self.info is None:
            return

        model_type = str(self.config["llm"]["type"])
        match model_type:
            case "openai":
                self.llm_model = ChatOpenAI(
                    model=str(self.config["openai"]["model"]),
                    temperature=float(self.config["openai"]["temperature"]),
                    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
                )
            case "google":
                self.llm_model = ChatGoogleGenerativeAI(
                    model=str(self.config["google"]["model"]),
                    temperature=float(self.config["google"]["temperature"]),
                    vertexai=True,
                )
            case "ollama":
                self.llm_model = ChatOllama(
                    model=str(self.config["ollama"]["model"]),
                    temperature=float(self.config["ollama"]["temperature"]),
                    base_url=str(self.config["ollama"]["base_url"]),
                )
            case "claude":
                self.llm_model = ChatAnthropic(
                    model_name=str(self.config["claude"]["model"]),
                    timeout=None,
                    stop=None,
                    api_key=SecretStr(os.environ["CLAUDE_API_KEY"]),
                )
            case _:
                raise ValueError(model_type, "Unknown LLM type")
        self.llm_model = self.llm_model
        self._send_message_to_llm(self.request)

    def daily_initialize(self) -> None:
        """Perform processing for daily initialization request.

        昼開始リクエストに対する処理を行う.
        """
        self._send_message_to_llm(self.request)

    def whisper(self) -> str:
        """Return response to whisper request.

        囁きリクエストに対する応答を返す.

        Returns:
            str: Whisper message / 囁きメッセージ
        """
        response = self._send_message_to_llm(self.request)
        self.sent_whisper_count = len(self.whisper_history)
        return response or ""

    def _clean_dialogue(self, dialogue: str) -> str:
        """Clean extracted dialogue by removing parenthetical content and brackets.

        抽出したセリフから括弧で囲われた部分や鍵括弧、ダブルクォートを除去する.

        Args:
            dialogue (str): Raw dialogue string / 未加工のセリフ文字列

        Returns:
            str: Cleaned dialogue / 整形済みのセリフ
        """
        # () や （） で囲われた部分を削除
        cleaned = re.sub(r"\([^)]*\)", "", dialogue)
        cleaned = re.sub(r"（[^）]*）", "", cleaned)
        # 「」の鍵括弧のみを削除（中身は残す）
        cleaned = cleaned.replace("「", "").replace("」", "")
        # ""のダブルクォートを削除（中身は残す）
        cleaned = cleaned.replace('"', "")
        # *を削除
        cleaned = cleaned.replace("*", "")
        # 余分な空白を整理
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _get_profile_names(self) -> list[str]:
        """Get list of character names from profiles.

        profilesから登場人物名のリストを取得する.

        Returns:
            list[str]: List of character names / 登場人物名のリスト
        """
        # まずstatus_mapから取得（ゲーム参加者のみ）
        if self.info and self.info.status_map:
            return list(self.info.status_map.keys())
        # フォールバック: configのprofilesから取得
        profiles = self.config.get("profiles", [])
        return [p.get("name", "") for p in profiles if p.get("name")]

    def _build_history_texts(self) -> set[str]:
        """Build a set of history dialogue texts for filtering.

        履歴のセリフテキストをフィルタリング用のセットとして構築する.
        プロンプトに渡された履歴部分を除外するために使用する.

        Returns:
            set[str]: Set of history dialogue texts / 履歴のセリフテキストのセット
        """
        history_texts: set[str] = set()
        history_portion = self.talk_history[self.sent_talk_count :]
        for talk in history_portion:
            if talk.text:
                # 正規化されたテキストを追加（空白の差異を吸収）
                normalized = re.sub(r"\s+", " ", talk.text).strip()
                history_texts.add(normalized)
                # **で囲まれた形式も追加
                history_texts.add(f"**{normalized}**")
        return history_texts

    def _is_history_dialogue(self, dialogue: str, history_texts: set[str]) -> bool:
        """Check if a dialogue is from the history.

        セリフが履歴からのものかどうかをチェックする.

        Args:
            dialogue (str): Dialogue text to check / チェックするセリフテキスト
            history_texts (set[str]): Set of history texts / 履歴テキストのセット

        Returns:
            bool: True if the dialogue is from history / 履歴からのセリフならTrue
        """
        normalized = re.sub(r"\s+", " ", dialogue).strip()
        # 完全一致チェック
        if normalized in history_texts:
            return True
        # **で囲まれた形式のチェック
        if f"**{normalized}**" in history_texts:
            return True
        # 先頭・末尾の**を除去してチェック
        stripped = normalized.strip("*").strip()
        if stripped in history_texts or f"**{stripped}**" in history_texts:
            return True
        return False

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.
        LLMからの応答（台本）から自分のエージェント名（info.agent）のセリフを探し、
        履歴部分を除外した追記分の最初のセリフを抽出して返す.
        抽出後、括弧内の内容と鍵括弧、ダブルクォートを除去する.

        Returns:
            str: Talk message / 発言メッセージ（セリフ部分）
        """
        # 履歴テキストを事前に取得（LLM呼び出し前）
        history_texts = self._build_history_texts()

        response = self._send_message_to_llm(self.request)
        self.sent_talk_count = len(self.talk_history)
        if response and self.info:
            # 自分のエージェント名を取得
            my_agent_name = self.info.agent
            if my_agent_name:
                # 自分のエージェント名 + 「: 」または「：」のパターンを構築
                # **名前** のようにマークダウン太字で囲まれている場合も対応
                escaped_name = re.escape(my_agent_name)
                pattern = rf"\*\*{escaped_name}\*\*[:：]\s*(.*)"
                # findallで全てのマッチを取得し、履歴でないものを探す
                matches = re.findall(pattern, response)
                for dialogue in matches:
                    dialogue = dialogue.strip()
                    if not self._is_history_dialogue(dialogue, history_texts):
                        return self._clean_dialogue(dialogue)
                # フォールバック: 太字なしの形式も試す
                pattern_plain = rf"{escaped_name}[:：]\s*(.*)"
                matches_plain = re.findall(pattern_plain, response)
                for dialogue in matches_plain:
                    dialogue = dialogue.strip()
                    if not self._is_history_dialogue(dialogue, history_texts):
                        return self._clean_dialogue(dialogue)
                # 全てが履歴だった場合は最後のマッチを使用
                if matches:
                    return self._clean_dialogue(matches[-1].strip())
                if matches_plain:
                    return self._clean_dialogue(matches_plain[-1].strip())
            # フォールバック: 最初の行を返す（後処理も適用）
            return self._clean_dialogue(response.split("\n")[0])
        return ""

    def daily_finish(self) -> None:
        """Perform processing for daily finish request.

        昼終了リクエストに対する処理を行う.
        """
        self._send_message_to_llm(self.request)

    def divine(self) -> str:
        """Return response to divine request.

        占いリクエストに対する応答を返す.

        Returns:
            str: Agent name to divine / 占い対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def guard(self) -> str:
        """Return response to guard request.

        護衛リクエストに対する応答を返す.

        Returns:
            str: Agent name to guard / 護衛対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def attack(self) -> str:
        """Return response to attack request.

        襲撃リクエストに対する応答を返す.

        Returns:
            str: Agent name to attack / 襲撃対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def finish(self) -> None:
        """Perform processing for game finish request.

        ゲーム終了リクエストに対する処理を行う.
        """

    @timeout
    def action(self) -> str | None:  # noqa: C901, PLR0911
        """Execute action according to request type.

        リクエストの種類に応じたアクションを実行する.

        Returns:
            str | None: Action result string or None / アクションの結果文字列またはNone
        """
        match self.request:
            case Request.NAME:
                return self.name()
            case Request.TALK:
                return self.talk()
            case Request.WHISPER:
                return self.whisper()
            case Request.VOTE:
                return self.vote()
            case Request.DIVINE:
                return self.divine()
            case Request.GUARD:
                return self.guard()
            case Request.ATTACK:
                return self.attack()
            case Request.INITIALIZE:
                self.initialize()
            case Request.DAILY_INITIALIZE:
                self.daily_initialize()
            case Request.DAILY_FINISH:
                self.daily_finish()
            case Request.FINISH:
                self.finish()
            case _:
                pass
        return None
