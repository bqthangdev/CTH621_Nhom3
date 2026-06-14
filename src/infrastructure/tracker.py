"""
Infrastructure — Experiment Tracker

Tích hợp W&B hoặc MLflow dựa theo cấu hình tracking.provider trong params.yaml.
Gọi tracker.log_run() trong domain modules sau khi tính xong metrics.
Nếu provider = "none" → no-op, không cần cài thư viện tracking.

Cách dùng:
    from src.infrastructure import tracker
    tracker.log_run(
        config,
        run_name="student_performance/logistic/Grade",
        params={"algorithm": "logistic", "C": 0.5},
        metrics={"accuracy": 0.92, "f1": 0.91},
        tags={"task": "classification", "dataset": "student_performance"},
    )
"""

from __future__ import annotations

from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


def log_run(
    config: dict,
    run_name: str,
    params: dict,
    metrics: dict,
    tags: dict | None = None,
) -> None:
    """
    Log một experiment run lên tracking server được cấu hình.

    Args:
        config:   Dict từ params.yaml (đọc tracking.provider).
        run_name: Tên run (ví dụ: "student_performance/logistic/Grade").
        params:   Siêu tham số của run (algorithm, target_col, …).
        metrics:  Kết quả đo lường (accuracy, rmse, silhouette_score, …).
        tags:     Nhãn phụ trợ (task, dataset, …).
    """
    provider = (config.get("tracking") or {}).get("provider", "none")
    if provider == "wandb":
        _log_wandb(config, run_name, params, metrics, tags or {})
    elif provider == "mlflow":
        _log_mlflow(config, run_name, params, metrics, tags or {})
    else:
        logger.debug(f"[TRACKER] provider=none — bỏ qua logging cho run '{run_name}'")


def _log_wandb(
    config: dict, run_name: str, params: dict, metrics: dict, tags: dict
) -> None:
    """Log metrics lên Weights & Biases."""
    try:
        import wandb  # type: ignore
    except ImportError:
        logger.warning("[TRACKER] wandb chưa được cài. Chạy: pip install wandb")
        return

    tc = config.get("tracking", {})
    run = wandb.init(
        project=tc.get("wandb_project", "CTH621"),
        name=run_name,
        config=params,
        tags=list(tags.values()) if tags else [],
        reinit=True,
    )
    wandb.log(metrics)
    run.finish()
    logger.info(f"[TRACKER] W&B logged run '{run_name}': {metrics}")


def _log_mlflow(
    config: dict, run_name: str, params: dict, metrics: dict, tags: dict
) -> None:
    """Log metrics lên MLflow tracking server."""
    try:
        import mlflow  # type: ignore
    except ImportError:
        logger.warning("[TRACKER] mlflow chưa được cài. Chạy: pip install mlflow")
        return

    tc = config.get("tracking", {})
    mlflow.set_tracking_uri(tc.get("mlflow_tracking_uri", "http://localhost:5000"))
    mlflow.set_experiment(tc.get("wandb_project", "CTH621"))
    with mlflow.start_run(run_name=run_name, tags=tags):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
    logger.info(f"[TRACKER] MLflow logged run '{run_name}': {metrics}")
