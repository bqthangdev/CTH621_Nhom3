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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from src.infrastructure.logger import get_logger

logger = get_logger(__name__)

SUMMARY_FILE = "summary_results.csv"
SUMMARY_FIELDS = [
    "timestamp", "dataset", "task", "algorithm", "target_col",
    "split", "n_train", "n_test",
    "accuracy", "precision", "recall", "f1",
    "cv_accuracy_mean", "cv_accuracy_std", "cv_f1_mean", "cv_f1_std",
    "mae", "rmse", "r2",
    "cv_mae_mean", "cv_mae_std", "cv_rmse_mean", "cv_rmse_std", "cv_r2_mean", "cv_r2_std",
    "silhouette_score", "cv_silhouette_mean", "cv_silhouette_std",
    "n_clusters", "notes",
]

ALGO_MAP = {
    "logistic": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "svm": SVC,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "knn": KNeighborsClassifier,
    "naive_bayes": GaussianNB,
}


def _append_summary(row: dict) -> None:
    row.setdefault("timestamp", datetime.now().isoformat())
    write_header = not Path(SUMMARY_FILE).exists()
    with open(SUMMARY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _plot_confusion_matrix(
    cm: np.ndarray, labels: list, algo: str, target_col: str, out_dir: str
) -> None:
    """Confusion matrix dạng heatmap — trực quan hóa số lượng TP/FP/TN/FN."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.3), max(4, len(labels) * 1.1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="gray",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Confusion Matrix — {algo} / {target_col}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(out_dir, f"cm_{algo}_{target_col}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"[CLASSIFY] Confusion matrix → {path}")


def _plot_roc_curve(
    model, X_test: pd.DataFrame, y_test: pd.Series,
    algo: str, target_col: str, out_dir: str
) -> None:
    """ROC/AUC curve cho phân loại nhị phân — đánh giá khả năng phân tách nhãn."""
    if y_test.nunique() != 2:
        return  # chỉ hỗ trợ binary
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            return
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#4C72B0", linewidth=2,
                label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        ax.fill_between(fpr, tpr, alpha=0.08, color="#4C72B0")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {algo} / {target_col}")
        ax.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(out_dir, f"roc_{algo}_{target_col}.png")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info(f"[CLASSIFY] ROC curve (AUC={roc_auc:.3f}) → {path}")
    except Exception as e:
        logger.warning(f"[CLASSIFY] Bỏ qua ROC curve ({algo}/{target_col}): {e}")


def _plot_feature_importance(
    model, feature_names: list, algo: str, target_col: str, out_dir: str
) -> None:
    """Feature importance bar chart — top-20 features có ảnh hưởng nhất đến dự báo."""
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef[0] if coef.ndim > 1 else coef)
    if importances is None or len(importances) == 0:
        return
    n_show = min(20, len(feature_names))
    indices = np.argsort(importances)[::-1][:n_show]
    top_names = [feature_names[i] for i in indices]
    top_vals  = importances[indices]
    fig, ax = plt.subplots(figsize=(9, max(4, n_show * 0.42)))
    ax.barh(range(n_show), top_vals[::-1], color="#4C72B0", edgecolor="black", height=0.7)
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {algo} / {target_col} (Top {n_show})")
    plt.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, f"feat_imp_{algo}_{target_col}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"[CLASSIFY] Feature importance → {path}")


class ClassificationPipeline:
    """
    Pipeline phân loại tổng quát.
    Nhận **kwargs để map toàn bộ hyperparameter từ params.yaml vào estimator.
    """

    def __init__(self, algo_name: str, random_state: int, **kwargs):
        """
        Args:
            algorithm:    Tên thuật toán: "logistic" | "decision_tree" | "svm".
            random_state: Đọc từ params.yaml — KHÔNG hardcode.
            **kwargs:     Hyperparameter bổ sung từ params.yaml.
        """
        if algo_name not in ALGO_MAP:
            raise ValueError(
                f"Thuật toán '{algo_name}' không hợp lệ. Chọn: {list(ALGO_MAP.keys())}"
            )
        EstimatorClass = ALGO_MAP[algo_name]
        # Chỉ truyền random_state nếu estimator hỗ trợ
        try:
            self.model = EstimatorClass(random_state=random_state, **kwargs)
        except TypeError:
            self.model = EstimatorClass(**kwargs)
        self.algorithm = algo_name
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

        # Compute model path trước khi split — dùng để kiểm tra checkpoint
        out_dir = os.path.join(
            config.get("base_output_dir", "outputs"),
            dataset_name, "models"
        )
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(out_dir, f"clf_{self.algorithm}_{target_col}.joblib")

        stratify_y = y if stratify_flag else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_y,
        )

        # Resume từ checkpoint nếu model đã tồn tại — bỏ qua bước fit
        if Path(model_path).exists():
            logger.info(f"[CLASSIFY] Nạp model từ checkpoint → {model_path}")
            self.model = joblib.load(model_path)
            if hasattr(self.model, "n_jobs"):
                self.model.set_params(n_jobs=1)
        else:
            self.model.fit(X_train, y_train)
            joblib.dump(self.model, model_path)
            logger.info(f"[CLASSIFY] Model đã lưu → {model_path}")

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

        # ── Trực quan hóa ────────────────────────────────────────────────────────
        viz_dir = os.path.join(
            config.get("base_output_dir", "outputs"),
            dataset_name, "ml", "classification"
        )
        _plot_confusion_matrix(
            cm, [str(l) for l in sorted(y.unique().tolist())],
            self.algorithm, target_col, viz_dir
        )
        _plot_roc_curve(self.model, X_test, y_test, self.algorithm, target_col, viz_dir)
        _plot_feature_importance(self.model, list(X.columns), self.algorithm, target_col, viz_dir)

        # ── StratifiedKFold CV (song song với Hold-out) ───────────────────────────
        n_splits = clf_cfg.get("cv_n_splits", 5)
        cv_strategy = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        cv_results = cross_validate(
            clone(self.model), X, y,
            cv=cv_strategy,
            scoring={
                "accuracy":  "accuracy",
                "f1":        "f1_weighted",
                "precision": "precision_weighted",
                "recall":    "recall_weighted",
            },
            return_train_score=False,
            error_score=0.0,
        )
        cv_acc_mean = round(float(np.mean(cv_results["test_accuracy"])), 4)
        cv_acc_std  = round(float(np.std(cv_results["test_accuracy"])),  4)
        cv_f1_mean  = round(float(np.mean(cv_results["test_f1"])),       4)
        cv_f1_std   = round(float(np.std(cv_results["test_f1"])),        4)
        logger.info(
            f"[CLASSIFY] {n_splits}-Fold StratifiedKFold | "
            f"Acc={cv_acc_mean}±{cv_acc_std} | F1={cv_f1_mean}±{cv_f1_std}"
        )

        result = {
            "dataset": dataset_name,
            "task": "classification",
            "algorithm": self.algorithm,
            "target_col": target_col,
            "split": f"train={1 - test_size:.0%}/test={test_size:.0%}",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy":         round(acc, 4),
            "precision":        round(prec, 4),
            "recall":           round(rec, 4),
            "f1":               round(f1, 4),
            "cv_accuracy_mean": cv_acc_mean,
            "cv_accuracy_std":  cv_acc_std,
            "cv_f1_mean":       cv_f1_mean,
            "cv_f1_std":        cv_f1_std,
        }
        _append_summary(result)
        from src.infrastructure import tracker
        tracker.log_run(
            config,
            run_name=f"{dataset_name}/{self.algorithm}/{target_col}",
            params={
                "algorithm": self.algorithm,
                "target_col": target_col,
                "test_size": test_size,
                "cv_n_splits": n_splits,
            },
            metrics={
                "accuracy":         round(acc, 4),
                "precision":        round(prec, 4),
                "recall":           round(rec, 4),
                "f1":               round(f1, 4),
                "cv_accuracy_mean": cv_acc_mean,
                "cv_f1_mean":       cv_f1_mean,
            },
            tags={"task": "classification", "dataset": dataset_name},
        )
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
