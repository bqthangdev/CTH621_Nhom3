"""
Data Layer — Validator
Kiểm tra ràng buộc đầu vào cho từng nhóm dữ liệu A, B, C.
Ném ValueError với thông báo rõ ràng nếu dữ liệu không hợp lệ.
"""

import pandas as pd
from typing import Optional

from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Group A — Tabular
# ─────────────────────────────────────────────────────────────────────────────

def validate_group_a(df: pd.DataFrame, config: dict) -> None:
    """
    Kiểm tra ràng buộc cho dữ liệu nhóm A (Tabular).

    Ràng buộc:
        - Số samples > min_samples (mặc định 100)
        - Số features > min_features (mặc định 5)
        - Phải có ít nhất 2 kiểu dữ liệu khác nhau trong các cột

    Args:
        df:     DataFrame cần kiểm tra.
        config: Dict cấu hình, đọc từ params.yaml['validation']['group_a'].

    Raises:
        ValueError: Nếu bất kỳ ràng buộc nào bị vi phạm.
    """
    cfg = config.get("validation", {}).get("group_a", {})
    min_samples = cfg.get("min_samples", 100)
    min_features = cfg.get("min_features", 5)

    errors = []

    n_samples, n_features = df.shape
    if n_samples <= min_samples:
        errors.append(f"Số samples ({n_samples}) phải > {min_samples}.")
    if n_features <= min_features:
        errors.append(f"Số features ({n_features}) phải > {min_features}.")

    dtypes = df.dtypes.astype(str).unique()
    if len(dtypes) < 2:
        errors.append(
            f"Chỉ có 1 kiểu dữ liệu ({dtypes[0]}). Nhóm A yêu cầu ít nhất 2 kiểu khác nhau."
        )

    if errors:
        msg = "Validation Group A thất bại:\n  " + "\n  ".join(errors)
        logger.error(msg)
        raise ValueError(msg)

    logger.info(
        f"[VALIDATE] Group A OK — {n_samples} samples, {n_features} features, "
        f"dtypes: {list(dtypes)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Group B — Time Series
# ─────────────────────────────────────────────────────────────────────────────

def validate_group_b(df: pd.DataFrame, config: dict) -> None:
    """
    Kiểm tra ràng buộc cho dữ liệu nhóm B (Time Series).

    Ràng buộc:
        - Số samples > min_samples (mặc định 500)
        - Index phải là DatetimeIndex (đã gọi load_timeseries trước)

    Args:
        df:     DataFrame với DatetimeIndex.
        config: Dict cấu hình, đọc từ params.yaml['validation']['group_b'].

    Raises:
        ValueError: Nếu bất kỳ ràng buộc nào bị vi phạm.
    """
    cfg = config.get("validation", {}).get("group_b", {})
    min_samples = cfg.get("min_samples", 500)

    errors = []

    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(
            "Index không phải DatetimeIndex. Hãy dùng load_timeseries() để nạp dữ liệu."
        )

    if len(df) <= min_samples:
        errors.append(f"Số samples ({len(df)}) phải > {min_samples}.")

    if errors:
        msg = "Validation Group B thất bại:\n  " + "\n  ".join(errors)
        logger.error(msg)
        raise ValueError(msg)

    logger.info(
        f"[VALIDATE] Group B OK — {len(df)} samples | "
        f"{df.index.min()} → {df.index.max()}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Group C — Multimedia
# ─────────────────────────────────────────────────────────────────────────────

def validate_group_c(items: list, subtype: str, config: dict) -> None:
    """
    Kiểm tra ràng buộc cho dữ liệu nhóm C (Multimedia).

    Args:
        items:   list[dict] trả về từ loader (load_images / load_audio / load_video).
                 Với subtype='text' truyền pd.DataFrame thay vì list.
        subtype: Một trong "image" | "audio" | "video" | "text".
        config:  Dict cấu hình từ params.yaml['validation']['group_c'][subtype].

    Raises:
        ValueError: Nếu bất kỳ ràng buộc nào bị vi phạm.
    """
    cfg = config.get("validation", {}).get("group_c", {}).get(subtype, {})
    errors = []

    if subtype == "image":
        min_samples = cfg.get("min_samples", 500)
        min_w = cfg.get("min_width", 256)
        min_h = cfg.get("min_height", 256)

        if len(items) < min_samples:
            errors.append(f"Số ảnh ({len(items)}) phải >= {min_samples}.")

        small = [
            r["path"] for r in items
            if r["width"] < min_w or r["height"] < min_h
        ]
        if small:
            errors.append(
                f"{len(small)} ảnh có kích thước nhỏ hơn {min_w}x{min_h}px. "
                f"Ví dụ: {small[:3]}"
            )

    elif subtype == "audio":
        min_samples = cfg.get("min_samples", 200)
        if len(items) < min_samples:
            errors.append(f"Số audio ({len(items)}) phải >= {min_samples}.")

    elif subtype == "video":
        min_samples = cfg.get("min_samples", 50)
        if len(items) < min_samples:
            errors.append(f"Số video ({len(items)}) phải >= {min_samples}.")

    elif subtype == "text":
        # items là pd.DataFrame khi subtype='text'
        min_samples = cfg.get("min_samples", 1000)
        n = len(items) if not isinstance(items, pd.DataFrame) else len(items)
        if n < min_samples:
            errors.append(f"Số văn bản ({n}) phải >= {min_samples}.")
    else:
        errors.append(f"Subtype không hợp lệ: '{subtype}'. Chọn image|audio|video|text.")

    if errors:
        msg = f"Validation Group C ({subtype}) thất bại:\n  " + "\n  ".join(errors)
        logger.error(msg)
        raise ValueError(msg)

    n = len(items)
    logger.info(f"[VALIDATE] Group C ({subtype}) OK — {n} samples")
