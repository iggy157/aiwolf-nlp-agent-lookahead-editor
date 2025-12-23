"""Script to launch agents according to configuration.

設定に応じたエージェントを起動するスクリプト.
"""

import argparse
import logging
import multiprocessing
from pathlib import Path

import yaml

from starter import connect

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
console_handler.setFormatter(formatter)


def execute(config_path: Path) -> None:
    """Execute based on the configuration file.

    設定ファイルをもとに実行する.

    Args:
        config_path (Path): Path to the configuration file / 設定ファイルのパス
    """
    with Path.open(config_path) as f:
        config = yaml.safe_load(f)
        logger.info("設定ファイルを読み込みました")

    profiles_path = config_path.parent / "profiles.yml"
    if profiles_path.exists():
        with Path.open(profiles_path) as f:
            profiles_config = yaml.safe_load(f)
            config["profiles"] = profiles_config.get("profiles", [])
            logger.info("プロファイルファイルを読み込みました")
    else:
        config["profiles"] = []

    agent_num = int(config["agent"]["num"])
    threads: list[multiprocessing.Process] = []
    for i in range(agent_num):
        thread = multiprocessing.Process(
            target=connect,
            args=(config, i + 1),
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        nargs="+",
        default=["./config/config.yml"],
        help="設定ファイルのパス (複数指定可)",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for config_path in args.config:
        glob_path = Path(config_path)
        paths.extend([path for path in Path.glob(glob_path.parent, glob_path.name) if path.is_file()])

    multiprocessing.set_start_method("spawn")
    threads: list[multiprocessing.Process] = []
    for path in paths:
        thread = multiprocessing.Process(
            target=execute,
            args=(Path(path),),
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
