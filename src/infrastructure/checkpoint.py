"""
Infrastructure — Checkpoint Manager
Theo dõi tiến độ pipeline qua progress.json.
Mỗi bước kiểm tra is_done() trước khi chạy; gọi mark_done() sau khi hoàn thành.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

_PROGRESS_FILE = "progress.json"


def init_checkpoint(base_dir: str = ".") -> None:
    """
    Khởi tạo đường dẫn progress.json dựa theo base_dir từ params.yaml.
    Gọi một lần trong main() sau khi nạp config, trước khi chạy bất kỳ bước nào.
    Cho phép trỏ vào thư mục đồng bộ chung (Google Drive, OneDrive) để
    các thành viên chia sẻ trạng thái pipeline.

    Args:
        base_dir: Thư mục gốc chứa progress.json (ví dụ: "data" hoặc đường dẫn Drive).
    """
    global _PROGRESS_FILE
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    _PROGRESS_FILE = str(Path(base_dir) / "progress.json")


def load_progress() -> dict:
    """Nạp toàn bộ trạng thái từ progress.json. Trả về {} nếu chưa có file."""
    if Path(_PROGRESS_FILE).exists():
        with open(_PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_progress(progress: dict) -> None:
    with open(_PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def is_done(dataset: str, step: str) -> bool:
    """
    Kiểm tra xem bước `step` của `dataset` đã hoàn thành chưa.

    Args:
        dataset: Tên dataset (snake_case, ví dụ "student_performance").
        step:    Tên bước ("eda", "classification", "regression", "clustering").

    Returns:
        True nếu trạng thái là "DONE" → nên bỏ qua bước này.
    """
    return load_progress().get(dataset, {}).get(step) == "DONE"


def mark_done(dataset: str, step: str) -> None:
    """
    Đánh dấu bước `step` của `dataset` là DONE kèm timestamp.

    Args:
        dataset: Tên dataset.
        step:    Tên bước.
    """
    progress = load_progress()
    if dataset not in progress:
        progress[dataset] = {}
    progress[dataset][step] = "DONE"
    progress[dataset][f"{step}_completed_at"] = datetime.now().isoformat()
    _save_progress(progress)


def mark_failed(dataset: str, step: str, reason: str = "") -> None:
    """Ghi trạng thái FAILED để dễ debug khi script bị lỗi."""
    progress = load_progress()
    if dataset not in progress:
        progress[dataset] = {}
    progress[dataset][step] = "FAILED"
    progress[dataset][f"{step}_failed_at"] = datetime.now().isoformat()
    if reason:
        progress[dataset][f"{step}_reason"] = reason
    _save_progress(progress)


def reset_step(dataset: str, step: Optional[str] = None) -> None:
    """
    Xóa trạng thái của một bước (hoặc toàn bộ dataset) để chạy lại.

    Args:
        dataset: Tên dataset.
        step:    Tên bước cần reset. Nếu None → reset toàn bộ dataset.
    """
    progress = load_progress()
    if dataset not in progress:
        return
    if step is None:
        del progress[dataset]
    else:
        progress[dataset].pop(step, None)
        progress[dataset].pop(f"{step}_completed_at", None)
        progress[dataset].pop(f"{step}_failed_at", None)
        progress[dataset].pop(f"{step}_reason", None)
    _save_progress(progress)
