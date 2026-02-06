"""``rich`` ライブラリを使用した構造化ロギングユーティリティ。"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console(stderr=True)


def setup_logging(*, verbose: bool = False) -> logging.Logger:
    """移行ツールのルートロガーを設定して返す。"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    logger = logging.getLogger("az_search_to_milvus")
    logger.setLevel(level)
    return logger


def get_logger(name: str = "az_search_to_milvus") -> logging.Logger:
    return logging.getLogger(name)
