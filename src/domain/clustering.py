"""
Domain — Clustering
Triển khai đồng thời K-Means, Hierarchical Clustering, DBSCAN.
BẮT BUỘC drop label columns trước khi fit.
Xuất Elbow curve, Silhouette scores, Dendrogram, và label files.
"""

import os
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.domain.classification import _append_summary
from src.infrastructure.logger import get_logger

logger = get_logger(__name__)


def _get_scaler(method: str):
    return StandardScaler() if method == "standard" else MinMaxScaler()


def _savefig(fig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"[CLUSTER] Đã lưu biểu đồ → {path}")


class ClusteringPipeline:
    """
    Pipeline clustering tổng quát.
    Chạy K-Means, Hierarchical, DBSCAN trên cùng feature matrix.
    """

    def __init__(self, random_state: int, **kwargs):
        """
        Args:
            random_state: Đọc từ params.yaml — KHÔNG hardcode.
            **kwargs:     Các tham số bổ sung (không dùng trong constructor này).
        """
        self.random_state = random_state

    def run(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        config: dict,
        label_columns: Optional[list] = None,
    ) -> dict:
        """
        Chạy toàn bộ clustering pipeline.

        Args:
            df:             DataFrame nguồn.
            dataset_name:   Tên dataset.
            config:         Dict từ params.yaml.
            label_columns:  Cột nhãn cần drop trước khi fit (BẮT BUỘC).

        Returns:
            dict kết quả {algorithm: {labels, silhouette_score, n_clusters}}.
        """
        cl_cfg = config.get("clustering", {})
        out_dir = os.path.join(
            config.get("base_output_dir", "outputs"),
            dataset_name, "ml", "clustering"
        )
        model_dir = os.path.join(
            config.get("base_output_dir", "outputs"),
            dataset_name, "models"
        )
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Drop label columns — BẮT BUỘC
        if label_columns is None:
            label_columns = config.get("datasets", {}).get(dataset_name, {}).get(
                "clustering", {}
            ).get("label_columns", [])

        feature_cols = [c for c in df.columns if c not in label_columns]
        if not feature_cols:
            raise ValueError("Không còn feature sau khi drop label columns!")
        logger.info(
            f"[CLUSTER] Drop {len(label_columns)} label col(s): {label_columns}. "
            f"Feature count: {len(feature_cols)}"
        )

        X = df[feature_cols].select_dtypes(include="number").dropna()
        scaler_method = cl_cfg.get("scaler", "standard")
        scaler = _get_scaler(scaler_method)
        X_scaled = scaler.fit_transform(X)

        results = {}
        results["kmeans"] = self._run_kmeans(X_scaled, X.index, cl_cfg, out_dir, model_dir, dataset_name, config)
        results["hierarchical"] = self._run_hierarchical(X_scaled, X.index, cl_cfg, out_dir, dataset_name, config)
        results["dbscan"] = self._run_dbscan(X_scaled, X.index, cl_cfg, out_dir, dataset_name, config)

        logger.info(f"[CLUSTER] Hoàn thành clustering cho {dataset_name}")
        return results

    # ── K-Means ──────────────────────────────────────────────────────────────

    def _run_kmeans(self, X_scaled, index, cl_cfg, out_dir, model_dir, dataset_name, config):
        max_k = cl_cfg.get("max_k", 10)
        k_range = range(2, max_k + 1)
        inertias, sil_scores = [], []

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X_scaled, labels))

        # Elbow curve
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(list(k_range), inertias, "bo-")
        axes[0].set_title("Elbow Method")
        axes[0].set_xlabel("K")
        axes[0].set_ylabel("Inertia")
        axes[1].plot(list(k_range), sil_scores, "rs-")
        axes[1].set_title("Silhouette Score")
        axes[1].set_xlabel("K")
        axes[1].set_ylabel("Score")
        _savefig(fig, os.path.join(out_dir, "kmeans_elbow_silhouette.png"))

        best_k = list(k_range)[int(np.argmax(sil_scores))]
        logger.info(f"[CLUSTER] K-Means: best_k={best_k}, silhouette={max(sil_scores):.4f}")

        km_final = KMeans(n_clusters=best_k, random_state=self.random_state, n_init=10)
        final_labels = km_final.fit_predict(X_scaled)
        joblib.dump(km_final, os.path.join(model_dir, "cluster_kmeans.joblib"))

        labels_df = pd.DataFrame({"kmeans_cluster": final_labels}, index=index)
        labels_df.to_csv(os.path.join(out_dir, "kmeans_labels.csv"))

        result = {"n_clusters": best_k, "silhouette_score": round(max(sil_scores), 4)}
        _append_summary({
            "dataset": dataset_name, "task": "clustering", "algorithm": "kmeans",
            "n_clusters": best_k, "silhouette_score": result["silhouette_score"],
        })
        return result

    # ── Hierarchical ─────────────────────────────────────────────────────────

    def _run_hierarchical(self, X_scaled, index, cl_cfg, out_dir, dataset_name, config):
        method = cl_cfg.get("linkage_method", "ward")
        n_clusters = cl_cfg.get("n_clusters", 3)

        Z = linkage(X_scaled, method=method)

        # Dendrogram (lấy mẫu tối đa 200 điểm để tránh quá tải)
        fig, ax = plt.subplots(figsize=(14, 6))
        dendrogram(
            Z, ax=ax, truncate_mode="lastp", p=min(50, len(X_scaled)),
            leaf_rotation=90, leaf_font_size=8,
        )
        ax.set_title(f"Dendrogram — {dataset_name} ({method} linkage)")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Distance")
        _savefig(fig, os.path.join(out_dir, "dendrogram.png"))

        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        try:
            sil = silhouette_score(X_scaled, labels)
        except Exception:
            sil = np.nan
        logger.info(f"[CLUSTER] Hierarchical: n_clusters={n_clusters}, silhouette={sil:.4f}")

        labels_df = pd.DataFrame({"hierarchical_cluster": labels}, index=index)
        labels_df.to_csv(os.path.join(out_dir, "hierarchical_labels.csv"))

        _append_summary({
            "dataset": dataset_name, "task": "clustering", "algorithm": "hierarchical",
            "n_clusters": n_clusters, "silhouette_score": round(float(sil), 4) if not np.isnan(sil) else "",
        })
        return {"n_clusters": n_clusters, "silhouette_score": sil}

    # ── DBSCAN ───────────────────────────────────────────────────────────────

    def _run_dbscan(self, X_scaled, index, cl_cfg, out_dir, dataset_name, config):
        eps = cl_cfg.get("dbscan_eps", 0.5)
        min_samples = cl_cfg.get("dbscan_min_samples", 5)

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = int((labels == -1).sum())
        logger.info(
            f"[CLUSTER] DBSCAN: eps={eps}, min_samples={min_samples} → "
            f"{n_clusters} clusters, {n_outliers} outliers ({n_outliers/len(labels)*100:.1f}%)"
        )

        try:
            sil = silhouette_score(X_scaled, labels) if n_clusters > 1 else np.nan
        except Exception:
            sil = np.nan

        labels_df = pd.DataFrame({"dbscan_cluster": labels}, index=index)
        labels_df.to_csv(os.path.join(out_dir, "dbscan_labels.csv"))

        _append_summary({
            "dataset": dataset_name, "task": "clustering", "algorithm": "dbscan",
            "n_clusters": n_clusters,
            "silhouette_score": round(float(sil), 4) if not np.isnan(sil) else "",
            "notes": f"outliers={n_outliers}",
        })
        return {"n_clusters": n_clusters, "n_outliers": n_outliers, "silhouette_score": sil}


def run_clustering(df: pd.DataFrame, dataset_name: str, config: dict) -> dict:
    """
    Khởi tạo và chạy ClusteringPipeline cho một dataset.

    Luôn chạy đủ 3 giải thuật: K-Means, Hierarchical, DBSCAN.
    Per-dataset overrides cho tham số clustering (max_k, n_clusters,
    dbscan_eps, dbscan_min_samples, linkage_method, scaler).

    Args:
        df:           DataFrame nguồn.
        dataset_name: Tên dataset.
        config:       Dict từ params.yaml.
    """
    from src.infrastructure.config_resolver import resolve_clustering_config

    cl_cfg = resolve_clustering_config(dataset_name, config)
    random_state = config.get("random_state", 42)
    label_columns = (
        ((config.get("datasets", {}) or {}).get(dataset_name, {}) or {})
        .get("clustering", {})
        .get("label_columns", [])
    )
    logger.info(
        f"[CLUSTER] Dataset={dataset_name} | "
        f"max_k={cl_cfg.get('max_k')} | n_clusters={cl_cfg.get('n_clusters')} | "
        f"dbscan_eps={cl_cfg.get('dbscan_eps')} | label_cols={label_columns}"
    )
    merged_config = {**config, "clustering": cl_cfg}
    pipeline = ClusteringPipeline(random_state=random_state)
    return pipeline.run(df, dataset_name, merged_config, label_columns=label_columns)
