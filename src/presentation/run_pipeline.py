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

from src.infrastructure.checkpoint import is_done, mark_done, mark_failed
from src.infrastructure.logger import get_logger


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
            logger.error(f"[EDA] Nhóm C không hỗ trợ qua CLI task 'eda'. Dùng notebook.")
            return

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
