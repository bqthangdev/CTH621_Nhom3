"""
Presentation — CLI Entry Point
Giao diện dòng lệnh cho pipeline CTH621.
Đọc toàn bộ cấu hình từ configs/params.yaml.

Cách dùng:
    python src/presentation/run_pipeline.py --task eda --dataset student_performance --config configs/params.yaml
    python src/presentation/run_pipeline.py --task classification --dataset student_performance --algo logistic --config configs/params.yaml
    python src/presentation/run_pipeline.py --task all --dataset stock_prices --config configs/params.yaml
"""

import argparse
import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

# Thêm root vào sys.path để import src.*
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.infrastructure.checkpoint import init_checkpoint, is_done, mark_done, mark_failed
from src.infrastructure.logger import get_logger

logger = get_logger("run_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Nạp cấu hình từ file .yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_environment(requirements_file: str = "requirements.txt") -> None:
    """
    Kiểm tra phiên bản thư viện so với requirements.txt.
    Cảnh báo nếu có sự khác biệt — tránh lỗi giữa các thành viên nhóm.
    """
    req_path = Path(requirements_file)
    if not req_path.exists():
        logger.warning(f"[ENV] Không tìm thấy {requirements_file} — bỏ qua kiểm tra môi trường.")
        return

    with open(req_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    mismatches = []
    for req in lines:
        if "==" in req:
            pkg, expected = req.split("==", 1)
            try:
                installed = importlib.metadata.version(pkg.strip())
                if installed.strip() != expected.strip():
                    mismatches.append(f"  {pkg}: expected {expected}, installed {installed}")
            except importlib.metadata.PackageNotFoundError:
                mismatches.append(f"  {pkg}: KHÔNG TÌM THẤY")

    if mismatches:
        logger.warning(
            "[ENV] Phát hiện khác biệt môi trường:\n" + "\n".join(mismatches)
        )
    else:
        logger.info("[ENV] Môi trường hợp lệ — tất cả thư viện đúng phiên bản.")


# ─────────────────────────────────────────────────────────────────────────────
# Task Runners
# ─────────────────────────────────────────────────────────────────────────────

def _load_group_c_dataframe(dataset_name: str, config: dict):
    """Load a multimedia dataset and return lightweight metadata as a DataFrame."""
    import pandas as pd
    from src.data.loader import load_audio, load_images, load_text, load_video
    from src.data.validator import validate_group_c

    ds_cfg = config.get("datasets", {}).get(dataset_name, {})
    subtype = ds_cfg.get("subtype")

    if subtype == "image":
        items = load_images(ds_cfg.get("dir", ""))
    elif subtype == "audio":
        try:
            items = load_audio(ds_cfg.get("dir", ""))
        except ImportError as exc:
            import numpy as np
            import wave

            logger.warning(f"[LOAD] {exc}. Falling back to WAV metadata only.")
            audio_dir = Path(ds_cfg.get("dir", ""))
            if not audio_dir.exists():
                raise FileNotFoundError(f"Thư mục audio không tồn tại: {audio_dir}")
            items = []
            def _wav_to_float(wav) -> tuple[np.ndarray, int]:
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                raw = wav.readframes(wav.getnframes())
                if sample_width == 1:
                    data = (np.frombuffer(raw, dtype=np.uint8).astype("float32") - 128.0) / 128.0
                elif sample_width == 2:
                    data = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
                elif sample_width == 4:
                    data = np.frombuffer(raw, dtype="<i4").astype("float32") / 2147483648.0
                else:
                    data = np.array([], dtype="float32")
                if channels > 1 and data.size:
                    data = data.reshape(-1, channels).mean(axis=1)
                return data, sample_rate

            def _audio_features(waveform: np.ndarray, sr: int) -> dict:
                if waveform.size == 0 or sr <= 0:
                    return {
                        "rms_mean": 0.0,
                        "rms_std": 0.0,
                        "zero_crossing_rate": 0.0,
                        "amplitude_mean": 0.0,
                        "amplitude_std": 0.0,
                        "amplitude_max": 0.0,
                        "spectral_centroid_hz": 0.0,
                        "spectral_bandwidth_hz": 0.0,
                        "spectral_rolloff_85_hz": 0.0,
                    }
                y = np.asarray(waveform, dtype="float32")
                abs_y = np.abs(y)
                rms = np.sqrt(np.mean(np.square(y)))
                zcr = np.mean(np.abs(np.diff(np.signbit(y).astype("int8"))))
                spectrum = np.abs(np.fft.rfft(y))
                freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
                total_energy = spectrum.sum()
                if total_energy > 0:
                    centroid = float((freqs * spectrum).sum() / total_energy)
                    bandwidth = float(np.sqrt((((freqs - centroid) ** 2) * spectrum).sum() / total_energy))
                    cumulative = np.cumsum(spectrum)
                    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * total_energy))
                    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
                else:
                    centroid = bandwidth = rolloff = 0.0
                return {
                    "rms_mean": float(rms),
                    "rms_std": float(np.std(np.square(y))),
                    "zero_crossing_rate": float(zcr),
                    "amplitude_mean": float(abs_y.mean()),
                    "amplitude_std": float(abs_y.std()),
                    "amplitude_max": float(abs_y.max()),
                    "spectral_centroid_hz": centroid,
                    "spectral_bandwidth_hz": bandwidth,
                    "spectral_rolloff_85_hz": rolloff,
                }

            for audio_path in audio_dir.rglob("*.wav"):
                sample_rate = 0
                duration = 0.0
                try:
                    with wave.open(str(audio_path), "rb") as wav:
                        waveform, sample_rate = _wav_to_float(wav)
                        frames = len(waveform)
                        duration = frames / sample_rate if sample_rate else 0.0
                except wave.Error as wave_exc:
                    logger.warning(f"[LOAD] Bỏ qua audio lỗi {audio_path}: {wave_exc}")
                    continue
                record = {
                    "path": str(audio_path),
                    "label": audio_path.parent.name,
                    "sample_rate": sample_rate,
                    "duration": duration,
                    "file_size_bytes": audio_path.stat().st_size,
                }
                record.update(_audio_features(waveform, sample_rate))
                items.append(record)
    elif subtype == "video":
        items = load_video(ds_cfg.get("dir", ""))
    elif subtype == "text":
        df = load_text(
            ds_cfg.get("file", ""),
            text_col=ds_cfg.get("text_col"),
            label_col=ds_cfg.get("label_col"),
        )
        validate_group_c(df, subtype, config)
        return df
    else:
        raise ValueError(f"Subtype Group C không hợp lệ cho '{dataset_name}': {subtype}")

    validate_group_c(items, subtype, config)
    rows = [
        {k: v for k, v in item.items() if k not in {"array", "waveform"}}
        for item in items
    ]
    df = pd.DataFrame(rows)

    metadata_files = ds_cfg.get("metadata_files", [])
    if metadata_files and "path" in df.columns:
        meta_frames = []
        for meta_file in metadata_files:
            meta_path = Path(meta_file)
            if meta_path.exists():
                meta_frames.append(pd.read_csv(meta_path))
            else:
                logger.warning(f"[LOAD] Metadata file không tồn tại: {meta_file}")

        if meta_frames:
            base_dir = Path(ds_cfg.get("dir", "."))
            if not base_dir.is_absolute():
                base_dir = ROOT / base_dir
            base_dir = base_dir.resolve()

            def _relative_media_path(path_value: str) -> str:
                media_path = Path(path_value)
                if not media_path.is_absolute():
                    media_path = ROOT / media_path
                try:
                    return media_path.resolve().relative_to(base_dir).as_posix()
                except ValueError:
                    return str(path_value).replace("\\", "/")

            metadata = pd.concat(meta_frames, ignore_index=True)
            if "fname" in metadata.columns:
                metadata = metadata.copy()
                df = df.copy()
                metadata.loc[:, "fname"] = metadata["fname"].astype(str).str.replace("\\", "/", regex=False)
                df.loc[:, "fname"] = df["path"].map(_relative_media_path)
                merge_cols = [
                    c for c in ["fname", "dataset", "label", "sublabel"]
                    if c in metadata.columns
                ]
                df = df.drop(
                    columns=[c for c in merge_cols if c != "fname" and c in df.columns],
                    errors="ignore",
                )
                df = df.merge(metadata[merge_cols], on="fname", how="left")

    if "label" in df.columns and df["label"].isna().any() and "path" in df.columns:
        fallback = df["path"].map(lambda p: Path(str(p)).stem.split("__", 1)[0])
        df.loc[:, "label"] = df["label"].fillna(fallback)

    return df


def run_eda_task(dataset_name: str, config: dict) -> None:
    """Chạy EDA pipeline cho một dataset."""
    from src.data.loader import load_tabular, load_timeseries
    from src.data.validator import validate_group_a, validate_group_b
    from src.domain.eda import run_eda_pipeline

    if is_done(dataset_name, "eda"):
        logger.info(f"[SKIP] EDA đã hoàn thành trước đó — {dataset_name}")
        return

    ds_cfg = config.get("datasets", {}).get(dataset_name, {})
    group = ds_cfg.get("type", "A")
    file_path = ds_cfg.get("file", "")

    try:
        if group == "A":
            df = load_tabular(file_path)
            validate_group_a(df, config)
        elif group == "B":
            datetime_col = ds_cfg.get("datetime_col", "date")
            df = load_timeseries(file_path, datetime_col)
            validate_group_b(df, config)
        else:
            df = _load_group_c_dataframe(dataset_name, config)
            logger.info(f"[EDA] Group C metadata loaded via CLI — {dataset_name}")

        run_eda_pipeline(df, dataset_name, group, config)
        mark_done(dataset_name, "eda")
        logger.info(f"[DONE] EDA — {dataset_name}")
    except Exception as e:
        mark_failed(dataset_name, "eda", str(e))
        logger.error(f"[FAIL] EDA — {dataset_name}: {e}", exc_info=True)
        raise


def run_classification_task(
    dataset_name: str, config: dict, algo_filter: Optional[str] = None
) -> None:
    """Chạy classification pipeline."""
    from src.domain.classification import run_classification

    step = "classification"
    if is_done(dataset_name, step):
        logger.info(f"[SKIP] Classification đã hoàn thành — {dataset_name}")
        return

    interim_path = Path(config.get("base_data_dir", "data")) / "interim" / f"{dataset_name}_transformed.parquet"
    if not interim_path.exists():
        logger.warning(f"[CLASSIFY] Chưa có dữ liệu transformed. Chạy EDA trước.")
        run_eda_task(dataset_name, config)

    try:
        import pandas as pd
        df = pd.read_parquet(interim_path)
        run_classification(df, dataset_name, config, algo_filter=algo_filter)
        mark_done(dataset_name, step)
        logger.info(f"[DONE] Classification — {dataset_name}")
    except Exception as e:
        mark_failed(dataset_name, step, str(e))
        logger.error(f"[FAIL] Classification — {dataset_name}: {e}", exc_info=True)
        raise


def run_regression_task(
    dataset_name: str, config: dict, algo_filter: Optional[str] = None
) -> None:
    """Chạy regression pipeline."""
    from src.data.loader import load_timeseries
    from src.domain.regression import run_regression

    step = "regression"
    if is_done(dataset_name, step):
        logger.info(f"[SKIP] Regression đã hoàn thành — {dataset_name}")
        return

    ds_cfg = config.get("datasets", {}).get(dataset_name, {})
    interim_path = Path(config.get("base_data_dir", "data")) / "interim" / f"{dataset_name}_transformed.parquet"

    try:
        import pandas as pd
        if interim_path.exists():
            df = pd.read_parquet(interim_path)
        else:
            datetime_col = ds_cfg.get("datetime_col", "date")
            df = load_timeseries(ds_cfg.get("file", ""), datetime_col)

        run_regression(df, dataset_name, config, algo_filter=algo_filter)
        mark_done(dataset_name, step)
        logger.info(f"[DONE] Regression — {dataset_name}")
    except Exception as e:
        mark_failed(dataset_name, step, str(e))
        logger.error(f"[FAIL] Regression — {dataset_name}: {e}", exc_info=True)
        raise


def run_clustering_task(dataset_name: str, config: dict) -> None:
    """Chạy clustering pipeline (luôn chạy cả 3: K-Means + Hierarchical + DBSCAN)."""
    from src.domain.clustering import run_clustering

    step = "clustering"
    if is_done(dataset_name, step):
        logger.info(f"[SKIP] Clustering đã hoàn thành — {dataset_name}")
        return

    interim_path = Path(config.get("base_data_dir", "data")) / "interim" / f"{dataset_name}_transformed.parquet"

    try:
        import pandas as pd
        if interim_path.exists():
            df = pd.read_parquet(interim_path)
        else:
            ds_cfg = config.get("datasets", {}).get(dataset_name, {})
            if ds_cfg.get("type") == "C":
                df = _load_group_c_dataframe(dataset_name, config)
            else:
                df = pd.read_csv(ds_cfg.get("file", ""))

        run_clustering(df, dataset_name, config)
        mark_done(dataset_name, step)
        logger.info(f"[DONE] Clustering — {dataset_name}")
    except Exception as e:
        mark_failed(dataset_name, step, str(e))
        logger.error(f"[FAIL] Clustering — {dataset_name}: {e}", exc_info=True)
        raise


def _run_all_for_dataset(
    dataset_name: str,
    config: dict,
    algo_filter: Optional[str] = None,
) -> None:
    """Chạy toàn bộ pipeline phù hợp cho một dataset dựa theo type (A/B/C)."""
    ds_group = config.get("datasets", {}).get(dataset_name, {}).get("type", "A")
    run_eda_task(dataset_name, config)
    if ds_group == "A":
        run_classification_task(dataset_name, config, algo_filter=algo_filter)
        run_clustering_task(dataset_name, config)
    elif ds_group == "B":
        run_regression_task(dataset_name, config, algo_filter=algo_filter)
        run_clustering_task(dataset_name, config)
    elif ds_group == "C":
        run_clustering_task(dataset_name, config)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="CTH621 Data Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Chạy từng task với dataset cụ thể:
  python src/presentation/run_pipeline.py --task eda --dataset student_performance
  python src/presentation/run_pipeline.py --task classification --dataset student_performance
  python src/presentation/run_pipeline.py --task regression --dataset stock_prices
  python src/presentation/run_pipeline.py --task clustering --dataset image_dataset

  # Chạy 1 algo cụ thể (override per-dataset config):
  python src/presentation/run_pipeline.py --task classification --dataset student_performance --algo logistic
  python src/presentation/run_pipeline.py --task regression --dataset stock_prices --algo xgboost

  # Chạy toàn bộ pipeline cho 1 dataset:
  python src/presentation/run_pipeline.py --task all --dataset student_performance

  # Chạy toàn bộ pipeline cho TẤT CẢ datasets trong config:
  python src/presentation/run_pipeline.py --task all --dataset all
        """,
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["eda", "classification", "regression", "clustering", "all"],
        help="Task cần chạy.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Tên dataset (phải có trong params.yaml datasets section).",
    )
    parser.add_argument(
        "--algo",
        default=None,
        help="Thuật toán cụ thể (tùy chọn, mặc định chạy tất cả trong config).",
    )
    parser.add_argument(
        "--config",
        default="configs/params.yaml",
        help="Đường dẫn tới file cấu hình .yaml (mặc định: configs/params.yaml).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset trạng thái checkpoint của dataset trước khi chạy.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    log_level = config.get("log_level", "INFO")

    global logger
    logger = get_logger("run_pipeline", level=log_level)
    logger.info(f"[START] task={args.task} | dataset={args.dataset} | config={args.config}")

    # Khởi tạo đường dẫn progress.json theo base_data_dir (hỗ trợ shared drive)
    init_checkpoint(config.get("base_data_dir", "."))
    logger.info(
        f"[CHECKPOINT] progress.json → "
        f"{config.get('base_data_dir', '.')}/progress.json"
    )

    # Kiểm tra môi trường khi khởi động
    validate_environment()

    if args.reset:
        from src.infrastructure.checkpoint import reset_step
        reset_step(args.dataset)
        logger.info(f"[RESET] Đã xóa checkpoint cho '{args.dataset}'")

    # Điều hướng task
    # --dataset all: lặp qua tất cả datasets trong config
    datasets_to_run: list
    if args.dataset == "all":
        datasets_to_run = list(config.get("datasets", {}).keys())
        logger.info(f"[ALL-DATASETS] Sẽ chạy {len(datasets_to_run)} dataset(s): {datasets_to_run}")
    else:
        if args.dataset not in config.get("datasets", {}):
            logger.error(
                f"[ERROR] Dataset '{args.dataset}' không tìm thấy trong params.yaml. "
                f"Các dataset hợp lệ: {list(config.get('datasets', {}).keys())}"
            )
            raise SystemExit(1)
        datasets_to_run = [args.dataset]

    for ds_name in datasets_to_run:
        logger.info(f"[START-DS] Bắt đầu dataset={ds_name} | task={args.task}")
        if args.task == "eda":
            run_eda_task(ds_name, config)
        elif args.task == "classification":
            run_classification_task(ds_name, config, algo_filter=args.algo)
        elif args.task == "regression":
            run_regression_task(ds_name, config, algo_filter=args.algo)
        elif args.task == "clustering":
            run_clustering_task(ds_name, config)
        elif args.task == "all":
            _run_all_for_dataset(ds_name, config, algo_filter=args.algo)
        logger.info(f"[DONE-DS] Hoàn thành dataset={ds_name}")

    logger.info(f"[FINISH] Pipeline hoàn thành — {args.dataset}")


if __name__ == "__main__":
    main()
