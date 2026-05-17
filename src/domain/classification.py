"""
Domain — Classification
Hỗ trợ ≥2 thuật toán, ≥3 target columns, đánh giá đầy đủ metrics,
log imbalance, lưu model .joblib, append kết quả vào summary_results.csv.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

SUMMARY_FILE = "summary_results.csv"
SUMMARY_FIELDS = [
    "timestamp", "dataset", "task", "algorithm", "target_col",
    "split", "n_train", "n_test",
    "accuracy", "precision", "recall", "f1",
    "mae", "rmse", "r2", "silhouette_score", "n_clusters", "notes",
]

ALGO_MAP = {
    "logistic": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "svm": SVC,
}


def _append_summary(row: dict) -> None:
    row.setdefault("timestamp", datetime.now().isoformat())
    write_header = not Path(SUMMARY_FILE).exists()
    with open(SUMMARY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


class ClassificationPipeline:
    """
    Pipeline phân loại tổng quát.
    Nhận **kwargs để map toàn bộ hyperparameter từ params.yaml vào estimator.
    """

    def __init__(self, algorithm: str, random_state: int, **kwargs):
        """
        Args:
            algorithm:    Tên thuật toán: "logistic" | "decision_tree" | "svm".
            random_state: Đọc từ params.yaml — KHÔNG hardcode.
            **kwargs:     Hyperparameter bổ sung từ params.yaml.
        """
        if algorithm not in ALGO_MAP:
            raise ValueError(
                f"Thuật toán '{algorithm}' không hợp lệ. Chọn: {list(ALGO_MAP.keys())}"
            )
        EstimatorClass = ALGO_MAP[algorithm]
        # Chỉ truyền random_state nếu estimator hỗ trợ
        try:
            self.model = EstimatorClass(random_state=random_state, **kwargs)
        except TypeError:
            self.model = EstimatorClass(**kwargs)
        self.algorithm = algorithm
        self.random_state = random_state

    def run(
        self,
        df: pd.DataFrame,
        target_col: str,
        dataset_name: str,
        config: dict,
        feature_cols: Optional[list] = None,
    ) -> dict:
        """
        Huấn luyện, đánh giá và lưu kết quả.

        Args:
            df:           DataFrame đã tiền xử lý (chỉ numeric features).
            target_col:   Cột mục tiêu.
            dataset_name: Tên dataset.
            config:       Dict từ params.yaml.
            feature_cols: Danh sách feature dùng để huấn luyện (mặc định: tất cả trừ target).

        Returns:
            dict kết quả metrics.
        """
        clf_cfg = config.get("classification", {})
        test_size = clf_cfg.get("test_size", 0.2)
        stratify_flag = clf_cfg.get("stratify", True)

        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target_col]

        X = df[feature_cols].select_dtypes(include="number")
        y = df[target_col]

        # Log phân phối nhãn — kiểm tra imbalance
        dist = y.value_counts(normalize=True)
        logger.info(f"[CLASSIFY] Phân phối nhãn '{target_col}':\n{dist.to_string()}")
        if dist.min() < 0.1:
            logger.warning(
                f"[CLASSIFY] Imbalanced data phát hiện ở '{target_col}': "
                f"nhãn thiểu số chỉ chiếm {dist.min()*100:.1f}%"
            )

        stratify_y = y if stratify_flag else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_y,
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        logger.info(
            f"[CLASSIFY] {self.algorithm} | target={target_col} | "
            f"Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}"
        )
        logger.info(f"[CLASSIFY] Confusion Matrix:\n{cm}")

        # Lưu model
        out_dir = os.path.join(
            config.get("base_output_dir", "outputs"),
            dataset_name, "models"
        )
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(out_dir, f"clf_{self.algorithm}_{target_col}.joblib")
        joblib.dump(self.model, model_path)
        logger.info(f"[CLASSIFY] Model đã lưu → {model_path}")

        result = {
            "dataset": dataset_name,
            "task": "classification",
            "algorithm": self.algorithm,
            "target_col": target_col,
            "split": f"train={1 - test_size:.0%}/test={test_size:.0%}",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
        _append_summary(result)
        return result


def run_classification(
    df: pd.DataFrame,
    dataset_name: str,
    config: dict,
    algo_filter: Optional[str] = None,
) -> list:
    """
    Chạy classification cho TẤT CẢ target columns và algorithms được cấu hình.

    Logic ưu tiên cho algorithms:
      1. CLI --algo   → chỉ chạy 1 algo đó
      2. datasets[name].classification.algorithms → chỉ chạy algo được liệt kê
      3. Global classification.algorithms → fallback, chạy tất cả

    Hyperparameter: dataset-level overrides global (shallow merge).

    Args:
        df:           DataFrame đã tiền xử lý.
        dataset_name: Tên dataset.
        config:       Dict từ params.yaml (cần có datasets[dataset_name]).
        algo_filter:  Nếu có, chỉ chạy algo này (từ CLI --algo).

    Returns:
        list[dict] kết quả metrics của từng run.
    """
    from src.infrastructure.config_resolver import resolve_algorithms

    ds_cfg = config.get("datasets", {}).get(dataset_name, {})
    target_columns = ds_cfg.get("classification", {}).get("target_columns", [])
    if not target_columns:
        raise ValueError(
            f"Không tìm thấy classification.target_columns cho '{dataset_name}' trong params.yaml."
        )
    if len(target_columns) < 3:
        logger.warning(
            f"[CLASSIFY] Chỉ có {len(target_columns)} target column(s). "
            "Guidelines yêu cầu ít nhất 3."
        )

    task_cfg, algo_map = resolve_algorithms("classification", dataset_name, config, algo_filter)
    if not algo_map:
        logger.warning(f"[CLASSIFY] Không có algorithm nào được cấu hình cho '{dataset_name}'.")
        return []

    random_state = config.get("random_state", 42)
    merged_config = {**config, "classification": task_cfg}

    logger.info(
        f"[CLASSIFY] Dataset={dataset_name} | Algorithms: {list(algo_map.keys())} | "
        f"Targets: {target_columns}"
    )

    all_results = []
    out_dir = os.path.join(config.get("base_output_dir", "outputs"), dataset_name, "ml", "classification")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for target_col in target_columns:
        for algo, algo_kwargs in algo_map.items():
            logger.info(f"[CLASSIFY] Bắt đầu: algo={algo}, target={target_col}")
            try:
                pipeline = ClassificationPipeline(algo, random_state=random_state, **algo_kwargs)
                result = pipeline.run(df, target_col, dataset_name, merged_config)
                all_results.append(result)
            except Exception as e:
                logger.error(f"[CLASSIFY] Lỗi {algo}/{target_col}: {e}", exc_info=True)

    logger.info(f"[CLASSIFY] Hoàn thành — {len(all_results)} run(s)")
    return all_results
