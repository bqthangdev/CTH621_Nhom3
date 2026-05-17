"""
Infrastructure — Logger
Cung cấp logger chuẩn cho toàn bộ pipeline CTH621.
KHÔNG dùng print() trong batch pipeline — chỉ dùng logger này.
"""

import logging
import os
from pathlib import Path


def get_logger(name: str, log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """
    Trả về Logger đã cấu hình, ghi log ra logs/pipeline.log và console.

    Args:
        name:    Tên module (thường truyền __name__).
        log_dir: Thư mục chứa file log.
        level:   Mức log mặc định ("DEBUG", "INFO", "WARNING", "ERROR").

    Returns:
        logging.Logger đã gắn FileHandler + StreamHandler.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # tránh duplicate handlers khi import lại

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — ghi tất cả từ DEBUG trở lên
    fh = logging.FileHandler(
        os.path.join(log_dir, "pipeline.log"), encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # Console handler — chỉ ghi từ level được cấu hình
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
