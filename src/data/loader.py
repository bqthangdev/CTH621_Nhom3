"""
Data Layer — Loader
Nạp dữ liệu từ nhiều định dạng khác nhau: CSV/Parquet (tabular & time series),
ảnh, audio, video, và văn bản.
Tất cả hàm trả về pd.DataFrame hoặc list[dict] chuẩn hóa.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Group A — Tabular
# ─────────────────────────────────────────────────────────────────────────────

def load_tabular(path: str) -> pd.DataFrame:
    """
    Nạp dữ liệu dạng bảng từ CSV hoặc Parquet.

    Args:
        path: Đường dẫn tới file .csv hoặc .parquet.

    Returns:
        pd.DataFrame chứa toàn bộ dữ liệu.
    """
    path = Path(path)
    logger.info(f"[LOAD] Tabular — {path}")
    if not path.exists():
        raise FileNotFoundError(f"File không tồn tại: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    logger.info(f"[LOAD] Kích thước: {df.shape} | Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Group B — Time Series
# ─────────────────────────────────────────────────────────────────────────────

def load_timeseries(path: str, datetime_col: str) -> pd.DataFrame:
    """
    Nạp dữ liệu chuỗi thời gian, chuyển cột thời gian thành DatetimeIndex.

    Args:
        path:         Đường dẫn tới file .csv hoặc .parquet.
        datetime_col: Tên cột chứa giá trị thời gian.

    Returns:
        pd.DataFrame với DatetimeIndex đã được sắp xếp tăng dần.
    """
    path = Path(path)
    logger.info(f"[LOAD] TimeSeries — {path} (datetime_col='{datetime_col}')")
    if not path.exists():
        raise FileNotFoundError(f"File không tồn tại: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col).sort_index()
    logger.info(f"[LOAD] Kích thước: {df.shape} | Phạm vi: {df.index.min()} → {df.index.max()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Group C — Images
# ─────────────────────────────────────────────────────────────────────────────

def load_images(dir_path: str, extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")) -> list:
    """
    Nạp toàn bộ ảnh từ thư mục, trích xuất metadata và pixel array.

    Args:
        dir_path:   Đường dẫn thư mục chứa ảnh.
        extensions: Tuple các đuôi file ảnh hợp lệ.

    Returns:
        list[dict] mỗi phần tử gồm:
            - path (str): đường dẫn file
            - width (int): chiều rộng
            - height (int): chiều cao
            - channels (int): số kênh màu
            - array (np.ndarray): pixel array dạng uint8
            - label (str): tên thư mục con (nếu có) dùng làm nhãn
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Cần cài Pillow: pip install Pillow")

    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Thư mục không tồn tại: {dir_path}")

    records = []
    image_files = [
        p for p in dir_path.rglob("*")
        if p.suffix.lower() in extensions
    ]
    logger.info(f"[LOAD] Images — tìm thấy {len(image_files)} file trong {dir_path}")

    def _image_features(arr: np.ndarray) -> dict:
        arr_float = arr.astype("float32") / 255.0
        gray = (
            0.299 * arr_float[:, :, 0]
            + 0.587 * arr_float[:, :, 1]
            + 0.114 * arr_float[:, :, 2]
        )
        edge_y = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
        edge_x = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0

        features = {
            "brightness_mean": float(gray.mean()),
            "brightness_std": float(gray.std()),
            "edge_density": float(edge_x + edge_y),
        }
        for idx, channel in enumerate(("red", "green", "blue")):
            values = arr_float[:, :, idx]
            features[f"{channel}_mean"] = float(values.mean())
            features[f"{channel}_std"] = float(values.std())
            hist, _ = np.histogram(values, bins=8, range=(0.0, 1.0), density=False)
            hist = hist.astype("float32") / max(1, hist.sum())
            for bin_idx, value in enumerate(hist):
                features[f"{channel}_hist_{bin_idx}"] = float(value)
        return features

    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)
            record = {
                "path": str(img_path),
                "label": img_path.parent.name,
                "width": img.width,
                "height": img.height,
                "channels": arr.shape[2] if arr.ndim == 3 else 1,
                "array": arr,
            }
            record.update(_image_features(arr))
            records.append(record)
        except Exception as e:
            logger.warning(f"[LOAD] Bỏ qua ảnh lỗi {img_path}: {e}")

    logger.info(f"[LOAD] Nạp thành công {len(records)} ảnh")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Group C — Audio
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(dir_path: str, extensions: tuple = (".wav", ".mp3", ".flac", ".ogg")) -> list:
    """
    Nạp toàn bộ file audio từ thư mục bằng librosa.

    Args:
        dir_path:   Đường dẫn thư mục chứa audio.
        extensions: Tuple các đuôi file audio hợp lệ.

    Returns:
        list[dict] mỗi phần tử gồm:
            - path (str): đường dẫn file
            - label (str): tên thư mục con
            - waveform (np.ndarray): dạng sóng
            - sample_rate (int): tần số lấy mẫu
            - duration (float): thời lượng (giây)
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("Cần cài librosa: pip install librosa")

    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Thư mục không tồn tại: {dir_path}")

    records = []
    audio_files = [
        p for p in dir_path.rglob("*")
        if p.suffix.lower() in extensions
    ]
    logger.info(f"[LOAD] Audio — tìm thấy {len(audio_files)} file trong {dir_path}")

    for audio_path in audio_files:
        try:
            waveform, sr = librosa.load(str(audio_path), sr=None)
            records.append({
                "path": str(audio_path),
                "label": audio_path.parent.name,
                "waveform": waveform,
                "sample_rate": sr,
                "duration": len(waveform) / sr,
            })
        except Exception as e:
            logger.warning(f"[LOAD] Bỏ qua audio lỗi {audio_path}: {e}")

    logger.info(f"[LOAD] Nạp thành công {len(records)} audio")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Group C — Video
# ─────────────────────────────────────────────────────────────────────────────

def load_video(dir_path: str, extensions: tuple = (".mp4", ".avi", ".mov", ".mkv")) -> list:
    """
    Nạp metadata của các file video từ thư mục bằng OpenCV.

    Args:
        dir_path:   Đường dẫn thư mục chứa video.
        extensions: Tuple các đuôi file video hợp lệ.

    Returns:
        list[dict] mỗi phần tử gồm:
            - path (str): đường dẫn file
            - label (str): tên thư mục con
            - fps (float): frames per second
            - frame_count (int): tổng số frame
            - width (int), height (int)
            - duration (float): thời lượng (giây)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("Cần cài opencv-python: pip install opencv-python")

    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Thư mục không tồn tại: {dir_path}")

    records = []
    video_files = [
        p for p in dir_path.rglob("*")
        if p.suffix.lower() in extensions
    ]
    logger.info(f"[LOAD] Video — tìm thấy {len(video_files)} file trong {dir_path}")

    for vid_path in video_files:
        try:
            cap = cv2.VideoCapture(str(vid_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            records.append({
                "path": str(vid_path),
                "label": vid_path.parent.name,
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": frame_count / fps if fps > 0 else 0.0,
            })
        except Exception as e:
            logger.warning(f"[LOAD] Bỏ qua video lỗi {vid_path}: {e}")

    logger.info(f"[LOAD] Nạp thành công {len(records)} video")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Group C — Text
# ─────────────────────────────────────────────────────────────────────────────

def load_text(path: str, text_col: Optional[str] = None, label_col: Optional[str] = None) -> pd.DataFrame:
    """
    Nạp dữ liệu văn bản từ CSV.

    Args:
        path:      Đường dẫn tới file .csv.
        text_col:  Tên cột chứa nội dung văn bản (tự phát hiện nếu None).
        label_col: Tên cột nhãn (tùy chọn).

    Returns:
        pd.DataFrame với ít nhất cột 'text' và 'label' (nếu có).
    """
    path = Path(path)
    logger.info(f"[LOAD] Text — {path}")
    if not path.exists():
        raise FileNotFoundError(f"File không tồn tại: {path}")

    df = pd.read_csv(path)

    # Tự phát hiện cột text nếu không truyền vào
    if text_col is None:
        str_cols = df.select_dtypes(include="object").columns.tolist()
        if not str_cols:
            raise ValueError("Không tìm thấy cột kiểu string trong file text.")
        text_col = max(str_cols, key=lambda c: df[c].str.len().mean())
        logger.info(f"[LOAD] Tự phát hiện cột text: '{text_col}'")

    df = df.rename(columns={text_col: "text"})
    if label_col and label_col in df.columns:
        df = df.rename(columns={label_col: "label"})

    logger.info(f"[LOAD] Kích thước text dataset: {df.shape}")
    return df
