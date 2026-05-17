"""
Infrastructure — Config Resolver

Merge global và dataset-specific algorithm configs từ params.yaml.

Logic ưu tiên (từ cao đến thấp):
  1. --algo CLI flag       → chỉ chạy 1 algo đó
  2. datasets[name][task].algorithms → chỉ chạy các algo được liệt kê
  3. Global [task].algorithms        → fallback, chạy tất cả algo toàn cục

Hyperparameter merge: dataset params override global params (shallow merge 1 cấp).
"""

from __future__ import annotations

from typing import Optional, Tuple


def resolve_algorithms(
    task: str,
    dataset_name: str,
    config: dict,
    algo_filter: Optional[str] = None,
) -> Tuple[dict, dict]:
    """
    Trả về (task_cfg, algo_map) sau khi merge global + dataset overrides.

    Args:
        task:         "classification" | "regression"
        dataset_name: Tên dataset trong config["datasets"]
        config:       Dict toàn bộ từ params.yaml
        algo_filter:  Nếu có, chỉ trả về algo này (từ CLI --algo)

    Returns:
        task_cfg: Dict tham số task-level đã merge (test_size, train_ratio, lags, …)
        algo_map: {algo_name: {merged_hyperparams}} — chỉ các algo sẽ được chạy
    """
    global_task: dict = config.get(task, {}) or {}
    ds_task: dict = (
        (config.get("datasets", {}) or {}).get(dataset_name, {}) or {}
    ).get(task, {}) or {}

    # Merge task-level settings — dataset overrides global
    _exclude = {"algorithms", "target_columns", "label_columns", "target_col"}
    task_cfg: dict = {
        **global_task,
        **{k: v for k, v in ds_task.items() if k not in _exclude},
    }

    # Resolve algorithm list
    global_algos: dict = global_task.get("algorithms", {}) or {}
    ds_algos = ds_task.get("algorithms", None)

    if ds_algos is not None:
        # Dataset defines exactly which algorithms to run
        algo_map: dict = {}
        for algo_name, ds_params in ds_algos.items():
            global_params = dict(global_algos.get(algo_name, {}) or {})
            ds_params = dict(ds_params or {})
            # Dataset-level params override global — shallow merge
            algo_map[algo_name] = {**global_params, **ds_params}
    else:
        # Fallback: run all globally defined algorithms unchanged
        algo_map = {k: dict(v or {}) for k, v in global_algos.items()}

    # CLI --algo filter: restrict to a single algorithm
    if algo_filter:
        if algo_filter in algo_map:
            algo_map = {algo_filter: algo_map[algo_filter]}
        else:
            # Algo not in current dataset config → try global fallback
            global_params = dict(global_algos.get(algo_filter, {}) or {})
            algo_map = {algo_filter: global_params}

    return task_cfg, algo_map


def resolve_clustering_config(dataset_name: str, config: dict) -> dict:
    """
    Merge global clustering config với per-dataset overrides.

    Các key bị loại trừ: label_columns (không phải tham số clustering).

    Args:
        dataset_name: Tên dataset trong config["datasets"]
        config:       Dict toàn bộ từ params.yaml

    Returns:
        Dict clustering config đã merge — dùng làm config["clustering"] khi chạy
    """
    global_cl: dict = dict(config.get("clustering", {}) or {})
    ds_cl: dict = dict(
        ((config.get("datasets", {}) or {}).get(dataset_name, {}) or {})
        .get("clustering", {}) or {}
    )
    _exclude = {"label_columns"}
    ds_overrides = {k: v for k, v in ds_cl.items() if k not in _exclude}
    return {**global_cl, **ds_overrides}
