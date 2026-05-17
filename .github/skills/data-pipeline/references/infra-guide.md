# Infrastructure Guide — CTH621

## Logger Setup (`src/infrastructure/logger.py`)

```python
import logging
import os
from datetime import datetime

def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Returns a configured logger writing to logs/pipeline.log.
    NEVER use print() in batch pipeline — use this logger instead.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, "pipeline.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    # Console handler (INFO and above only)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
```

---

## Checkpoint Manager (`src/infrastructure/checkpoint.py`)

```python
import json
import os
from pathlib import Path

PROGRESS_FILE = "progress.json"

def load_progress() -> dict:
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}

def is_done(dataset: str, step: str) -> bool:
    """Returns True if step is already marked DONE — skip processing."""
    progress = load_progress()
    return progress.get(dataset, {}).get(step) == "DONE"

def mark_done(dataset: str, step: str) -> None:
    """Marks a step as DONE in progress.json."""
    progress = load_progress()
    if dataset not in progress:
        progress[dataset] = {}
    progress[dataset][step] = "DONE"
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

# Usage pattern in any pipeline step:
# if is_done(dataset_name, "eda"):
#     logger.info(f"[SKIP] EDA already done for {dataset_name}")
# else:
#     run_eda(...)
#     mark_done(dataset_name, "eda")
```

---

## Summary Results (`summary_results.csv`)

All ML metrics must be appended (not overwritten) here:

```python
import csv
from pathlib import Path

SUMMARY_FILE = "summary_results.csv"
SUMMARY_FIELDS = ["timestamp", "dataset", "task", "algorithm", "target_col",
                  "split", "accuracy", "precision", "recall", "f1",
                  "mae", "rmse", "r2", "silhouette_score", "n_clusters", "notes"]

def append_summary(row: dict) -> None:
    """Appends one result row to summary_results.csv."""
    write_header = not Path(SUMMARY_FILE).exists()
    with open(SUMMARY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        row.setdefault("timestamp", datetime.now().isoformat())
        writer.writerow(row)
```

---

## PM2 Config (`ecosystem.config.js`)

For long-running batch jobs on Linux servers:

```javascript
module.exports = {
  apps: [
    {
      name: "cth621-pipeline",
      script: "src/presentation/run_pipeline.py",
      interpreter: "python3",
      args: "--task all --config configs/params.yaml",
      autorestart: true,
      watch: false,
      max_memory_restart: "4G",
      log_file: "logs/pm2_pipeline.log",
      out_file: "logs/pm2_out.log",
      error_file: "logs/pm2_err.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss"
    }
  ]
};
```

---

## W&B / MLflow Integration

### Weights & Biases (preferred for team sharing)
```python
import wandb

def init_experiment(config: dict, project: str = "CTH621") -> None:
    wandb.init(
        project=project,
        config=config,
        name=f"{config['dataset']}_{config['task']}_{config['algorithm']}"
    )

def log_metrics(metrics: dict) -> None:
    wandb.log(metrics)

def save_artifact(file_path: str, artifact_type: str = "model") -> None:
    artifact = wandb.Artifact(name=Path(file_path).stem, type=artifact_type)
    artifact.add_file(file_path)
    wandb.log_artifact(artifact)
```

### MLflow (alternative)
```python
import mlflow

def log_run(config: dict, metrics: dict, model_path: str) -> None:
    with mlflow.start_run(run_name=f"{config['dataset']}_{config['algorithm']}"):
        mlflow.log_params(config)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(model_path)
```

---

## Environment Validation (Team Sync)

Add this check to `run_pipeline.py` startup:

```python
import importlib.metadata
import yaml

def validate_environment(requirements_file: str = "requirements.txt") -> None:
    """Warns if installed package versions differ from requirements.txt."""
    with open(requirements_file) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    mismatches = []
    for req in lines:
        if "==" in req:
            pkg, expected = req.split("==")
            try:
                installed = importlib.metadata.version(pkg.strip())
                if installed != expected.strip():
                    mismatches.append(f"{pkg}: expected {expected}, got {installed}")
            except importlib.metadata.PackageNotFoundError:
                mismatches.append(f"{pkg}: NOT INSTALLED")
    if mismatches:
        logger.warning("Environment mismatch detected:\n" + "\n".join(mismatches))
```
