"""Resolve additive, versioned artifact paths without touching legacy outputs."""

from pathlib import Path


def output_policy(config: dict) -> dict:
    """Return the configured output-retention policy."""
    return config.get("output_policy", {}) or {}


def dataset_output_root(config: dict, dataset_name: str) -> Path:
    """Return the dataset artifact root, optionally isolated by output version."""
    root = Path(config.get("base_output_dir", "outputs")) / dataset_name
    policy = output_policy(config)
    if policy.get("preserve_existing_outputs", False):
        version = str(policy.get("version_subdir", "enhanced_v2")).strip()
        if version:
            root = root / version
    return root


def transformed_interim_path(config: dict, dataset_name: str) -> Path:
    """Return a versioned transformed-data path when legacy preservation is on."""
    interim_dir = Path(config.get("base_data_dir", "data")) / "interim"
    policy = output_policy(config)
    suffix = ""
    if policy.get("preserve_existing_outputs", False):
        version = str(policy.get("version_subdir", "enhanced_v2")).strip()
        if version:
            safe_version = "".join(
                char if char.isalnum() or char in {"-", "_"} else "_"
                for char in version
            )
            suffix = f"_{safe_version}"
    return interim_dir / f"{dataset_name}_transformed{suffix}.parquet"
