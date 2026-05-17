"""
Domain — Regression (Time Series)
KHÔNG dùng random split — chỉ dùng chronological split.
Hỗ trợ: Linear Regression, ARIMA/SARIMA, XGBoost với rolling-window features.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.domain.classification import _append_summary
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering — Rolling Window
# ─────────────────────────────────────────────────────────────────────────────

def build_lag_features(df: pd.DataFrame, target_col: str, lags: list, rolling_windows: list) -> pd.DataFrame:
    """
    Tạo lag features và rolling mean features cho XGBoost.

    Args:
        df:              DataFrame với DatetimeIndex.
        target_col:      Tên cột mục tiêu.
        lags:            List lag (ví dụ [1, 7, 14]).
        rolling_windows: List window size (ví dụ [7, 30]).

    Returns:
        DataFrame đã thêm các cột lag và rolling, đã dropna.
    """
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    for w in rolling_windows:
        df[f"rolling_mean_{w}"] = df[target_col].rolling(w).mean()
    df = df.dropna()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Chronological Split (BẮT BUỘC — không dùng random)
# ─────────────────────────────────────────────────────────────────────────────

def chronological_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Chia dữ liệu time series theo thứ tự thời gian.
    Tuyệt đối KHÔNG dùng random_state hay shuffle.

    Args:
        df:          DataFrame đã sort theo thời gian.
        train_ratio: Tỷ lệ train (mặc định 0.8).

    Returns:
        (train_df, test_df)
    """
    split_idx = int(len(df) * train_ratio)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    logger.info(
        f"[REGRESSION] Chronological split — "
        f"Train: {len(train)} ({train.index.min()} → {train.index.max()}) | "
        f"Test: {len(test)} ({test.index.min()} → {test.index.max()})"
    )
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# Regression Pipelines
# ─────────────────────────────────────────────────────────────────────────────

def _eval_metrics(y_true, y_pred, algo: str, dataset_name: str, target_col: str, config: dict) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    logger.info(f"[REGRESSION] {algo} | MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")
    result = {
        "dataset": dataset_name, "task": "regression",
        "algorithm": algo, "target_col": target_col,
        "split": f"chronological_{config.get('regression', {}).get('train_ratio', 0.8):.0%}",
        "mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4),
    }
    _append_summary(result)
    return result


def _plot_predictions(y_true, y_pred, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true.values, label="Thực tế", linewidth=1, color="#4C72B0")
    ax.plot(y_pred, label="Dự đoán", linewidth=1, color="#DD8452", linestyle="--")
    ax.set_title(title)
    ax.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"[REGRESSION] Đã lưu biểu đồ dự đoán → {out_path}")


class RegressionPipeline:
    """
    Pipeline hồi quy tổng quát cho dữ liệu chuỗi thời gian.
    Nhận **kwargs để map hyperparameter từ params.yaml.
    """

    def __init__(self, algorithm: str, random_state: int, **kwargs):
        """
        Args:
            algorithm:    "linear_regression" | "xgboost" | "arima"
            random_state: Đọc từ params.yaml — KHÔNG hardcode.
            **kwargs:     Hyperparameter từ params.yaml.
        """
        self.algorithm = algorithm
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None

    def run(
        self,
        df: pd.DataFrame,
        target_col: str,
        dataset_name: str,
        config: dict,
    ) -> dict:
        """
        Huấn luyện, đánh giá và lưu mô hình.

        Args:
            df:           DataFrame với DatetimeIndex (đã tiền xử lý).
            target_col:   Cột mục tiêu.
            dataset_name: Tên dataset.
            config:       Dict từ params.yaml.
        """
        reg_cfg = config.get("regression", {})
        train_ratio = reg_cfg.get("train_ratio", 0.8)
        out_dir = os.path.join(
            config.get("base_output_dir", "outputs"),
            dataset_name, "ml", "regression"
        )
        model_dir = os.path.join(config.get("base_output_dir", "outputs"), dataset_name, "models")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        if self.algorithm == "linear_regression":
            return self._run_linear(df, target_col, dataset_name, config, train_ratio, out_dir, model_dir)
        elif self.algorithm == "xgboost":
            return self._run_xgboost(df, target_col, dataset_name, config, reg_cfg, train_ratio, out_dir, model_dir)
        elif self.algorithm == "arima":
            return self._run_arima(df, target_col, dataset_name, config, reg_cfg, train_ratio, out_dir, model_dir)
        else:
            raise ValueError(f"Thuật toán không hợp lệ: '{self.algorithm}'")

    def _run_linear(self, df, target_col, dataset_name, config, train_ratio, out_dir, model_dir):
        train, test = chronological_split(df[[target_col]], train_ratio)
        # Dùng timestep index làm feature đơn giản
        X_train = np.arange(len(train)).reshape(-1, 1)
        X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
        self.model = LinearRegression()
        self.model.fit(X_train, train[target_col])
        y_pred = self.model.predict(X_test)
        _plot_predictions(test[target_col], y_pred, f"Linear Regression — {target_col}",
                          os.path.join(out_dir, f"linear_{target_col}.png"))
        joblib.dump(self.model, os.path.join(model_dir, f"reg_linear_{target_col}.joblib"))
        return _eval_metrics(test[target_col], y_pred, self.algorithm, dataset_name, target_col, config)

    def _run_xgboost(self, df, target_col, dataset_name, config, reg_cfg, train_ratio, out_dir, model_dir):
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("Cần cài xgboost: pip install xgboost")

        lags = reg_cfg.get("lags", [1, 7, 14])
        windows = reg_cfg.get("rolling_windows", [7, 30])
        df_feat = build_lag_features(df[[target_col]], target_col, lags, windows)

        feature_cols = [c for c in df_feat.columns if c != target_col]
        train, test = chronological_split(df_feat, train_ratio)

        early_stop = self.kwargs.pop("early_stopping_rounds", None)
        kwargs = {k: v for k, v in self.kwargs.items() if k != "early_stopping_rounds"}
        self.model = XGBRegressor(
            random_state=self.random_state,
            **kwargs,
        )
        fit_params = {}
        if early_stop:
            fit_params = {
                "eval_set": [(test[feature_cols], test[target_col])],
                "verbose": False,
            }
            self.model.set_params(early_stopping_rounds=early_stop)

        self.model.fit(train[feature_cols], train[target_col], **fit_params)
        y_pred = self.model.predict(test[feature_cols])
        _plot_predictions(test[target_col], y_pred, f"XGBoost — {target_col}",
                          os.path.join(out_dir, f"xgb_{target_col}.png"))
        joblib.dump(self.model, os.path.join(model_dir, f"reg_xgb_{target_col}.joblib"))
        return _eval_metrics(test[target_col], y_pred, "xgboost", dataset_name, target_col, config)

    def _run_arima(self, df, target_col, dataset_name, config, reg_cfg, train_ratio, out_dir, model_dir):
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ImportError("Cần cài statsmodels: pip install statsmodels")

        # order và seasonal_order được lấy từ self.kwargs (đã merge global + dataset overrides)
        order = tuple(self.kwargs.get("order", [1, 1, 1]))
        seasonal_order = tuple(self.kwargs.get("seasonal_order", [0, 0, 0, 0]))

        series = df[target_col]
        split_idx = int(len(series) * train_ratio)
        train_s, test_s = series.iloc[:split_idx], series.iloc[split_idx:]

        self.model = SARIMAX(train_s, order=order, seasonal_order=seasonal_order)
        fitted = self.model.fit(disp=False)
        y_pred = fitted.forecast(steps=len(test_s))

        _plot_predictions(test_s, y_pred.values, f"ARIMA — {target_col}",
                          os.path.join(out_dir, f"arima_{target_col}.png"))
        fitted.save(os.path.join(model_dir, f"reg_arima_{target_col}.pkl"))
        return _eval_metrics(test_s, y_pred.values, "arima", dataset_name, target_col, config)


def run_regression(
    df: pd.DataFrame,
    dataset_name: str,
    config: dict,
    algo_filter: Optional[str] = None,
) -> list:
    """
    Chạy regression với algorithms được cấu hình trong params.yaml.

    Logic ưu tiên cho algorithms:
      1. CLI --algo   → chỉ chạy 1 algo đó
      2. datasets[name].regression.algorithms → chỉ chạy algo được liệt kê
      3. Global regression.algorithms → fallback, chạy tất cả

    Hyperparameter (n_estimators, order, lags, …): dataset-level overrides global.

    Args:
        df:           DataFrame với DatetimeIndex.
        dataset_name: Tên dataset.
        config:       Dict từ params.yaml.
        algo_filter:  Nếu có, chỉ chạy algo này (từ CLI --algo).
    """
    from src.infrastructure.config_resolver import resolve_algorithms

    target_col = config.get("datasets", {}).get(dataset_name, {}).get("target_col")
    if not target_col:
        raise ValueError(f"Không tìm thấy target_col cho '{dataset_name}' trong params.yaml.")

    task_cfg, algo_map = resolve_algorithms("regression", dataset_name, config, algo_filter)
    if not algo_map:
        logger.warning(f"[REGRESSION] Không có algorithm nào được cấu hình cho '{dataset_name}'.")
        return []

    random_state = config.get("random_state", 42)
    merged_config = {**config, "regression": task_cfg}

    logger.info(
        f"[REGRESSION] Dataset={dataset_name} | Algorithms: {list(algo_map.keys())} | "
        f"Target: {target_col} | lags={task_cfg.get('lags')} | "
        f"rolling_windows={task_cfg.get('rolling_windows')}"
    )

    all_results = []
    for algo, algo_kwargs in algo_map.items():
        logger.info(f"[REGRESSION] Bắt đầu: algo={algo}, target={target_col}")
        try:
            pipeline = RegressionPipeline(algo, random_state=random_state, **algo_kwargs)
            result = pipeline.run(df, target_col, dataset_name, merged_config)
            all_results.append(result)
        except Exception as e:
            logger.error(f"[REGRESSION] Lỗi {algo}: {e}", exc_info=True)

    logger.info(f"[REGRESSION] Hoàn thành — {len(all_results)} run(s)")
    return all_results
