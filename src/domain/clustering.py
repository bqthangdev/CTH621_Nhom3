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
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
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


def _plot_cluster_pca(
    X_scaled: np.ndarray, labels: np.ndarray,
    algo_name: str, dataset_name: str, out_dir: str
) -> None:
    """PCA 2D scatter — chiếu các cluster xuống 2D để trực quan hóa cấu trúc phân cụm."""
    try:
        from sklearn.decomposition import PCA
        n_components = min(2, X_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_2d = pca.fit_transform(X_scaled)
        explained = pca.explained_variance_ratio_

        unique_labels = sorted(set(labels))
        n_clusters = len([l for l in unique_labels if l != -1])
        palette = sns.color_palette("tab10", max(n_clusters, 1))
        color_map = {}  # label → color
        ci = 0
        for lbl in unique_labels:
            color_map[lbl] = "#aaaaaa" if lbl == -1 else palette[ci % len(palette)]
            if lbl != -1:
                ci += 1

        fig, ax = plt.subplots(figsize=(8, 6))
        for lbl in unique_labels:
            mask = labels == lbl
            label_name = "Outlier" if lbl == -1 else f"Cluster {lbl}"
            marker = "x" if lbl == -1 else "o"
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1] if n_components > 1 else np.zeros(mask.sum()),
                c=[color_map[lbl]] * mask.sum(),
                label=label_name, marker=marker,
                alpha=0.55, s=18, linewidths=0.3,
            )

        x_label = f"PC1 ({explained[0]*100:.1f}%)"
        y_label = f"PC2 ({explained[1]*100:.1f}%)" if n_components > 1 else "PC2"
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"Cluster Scatter (PCA 2D) — {algo_name} / {dataset_name}")
        ax.legend(loc="best", markerscale=1.8, fontsize=8,
                  ncol=max(1, len(unique_labels) // 8))
        plt.tight_layout()
        _savefig(fig, os.path.join(out_dir, f"scatter_pca_{algo_name}.png"))
    except Exception as e:
        logger.warning(f"[CLUSTER] Bỏ qua PCA scatter ({algo_name}): {e}")


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
        results["gaussian_mixture"] = self._run_gaussian_mixture(X_scaled, X.index, cl_cfg, out_dir, dataset_name, config)

        logger.info(f"[CLUSTER] Hoàn thành clustering cho {dataset_name}")
        return results

    # ── K-Means ──────────────────────────────────────────────────────────────

    def _run_kmeans(self, X_scaled, index, cl_cfg, out_dir, model_dir, dataset_name, config):
        max_k = cl_cfg.get("max_k", 10)
        kmeans_init = cl_cfg.get("kmeans_init", "k-means++")
        kmeans_n_init = cl_cfg.get("kmeans_n_init", 10)
        kmeans_max_iter = cl_cfg.get("kmeans_max_iter", 300)
        k_range = range(2, max_k + 1)
        inertias, sil_scores = [], []

        for k in k_range:
            km = KMeans(
                n_clusters=k,
                init=kmeans_init,
                n_init=kmeans_n_init,
                max_iter=kmeans_max_iter,
                random_state=self.random_state,
            )
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

        km_final = KMeans(
            n_clusters=best_k,
            init=kmeans_init,
            n_init=kmeans_n_init,
            max_iter=kmeans_max_iter,
            random_state=self.random_state,
        )
        final_labels = km_final.fit_predict(X_scaled)
        joblib.dump(km_final, os.path.join(model_dir, "cluster_kmeans.joblib"))

        labels_df = pd.DataFrame({"kmeans_cluster": final_labels}, index=index)
        labels_df.to_csv(os.path.join(out_dir, "kmeans_labels.csv"))
        _plot_cluster_pca(X_scaled, final_labels, "kmeans", dataset_name, out_dir)

        # ── Subsampling CV: đánh giá độ ổn định silhouette ───────────────────────
        cv_n_splits = cl_cfg.get("cv_n_splits", 5)
        rng = np.random.RandomState(self.random_state)
        n_total = len(X_scaled)
        cv_sil_list = []
        for _ in range(cv_n_splits):
            idx = rng.choice(n_total, size=int(0.8 * n_total), replace=False)
            km_cv = KMeans(
                n_clusters=best_k, init=kmeans_init,
                n_init=kmeans_n_init, max_iter=kmeans_max_iter,
                random_state=self.random_state,
            )
            km_cv.fit(X_scaled[idx])
            labels_cv = km_cv.predict(X_scaled)
            try:
                cv_sil_list.append(silhouette_score(X_scaled, labels_cv))
            except Exception:
                cv_sil_list.append(np.nan)
        cv_sil_mean = round(float(np.nanmean(cv_sil_list)), 4)
        cv_sil_std  = round(float(np.nanstd(cv_sil_list)),  4)
        logger.info(
            f"[CLUSTER] K-Means subsampling CV ({cv_n_splits}x) | "
            f"Silhouette={cv_sil_mean}±{cv_sil_std}"
        )

        result = {
            "n_clusters": best_k,
            "silhouette_score": round(max(sil_scores), 4),
            "cv_silhouette_mean": cv_sil_mean,
            "cv_silhouette_std":  cv_sil_std,
        }
        _append_summary({
            "dataset": dataset_name, "task": "clustering", "algorithm": "kmeans",
            "n_clusters": best_k, "silhouette_score": result["silhouette_score"],
            "cv_silhouette_mean": cv_sil_mean, "cv_silhouette_std": cv_sil_std,
        })
        from src.infrastructure import tracker
        tracker.log_run(
            config,
            run_name=f"{dataset_name}/kmeans",
            params={"algorithm": "kmeans", "n_clusters": best_k},
            metrics={
                "silhouette_score":   round(max(sil_scores), 4),
                "cv_silhouette_mean": cv_sil_mean,
                "cv_silhouette_std":  cv_sil_std,
            },
            tags={"task": "clustering", "dataset": dataset_name},
        )
        return result

    # ── Hierarchical ─────────────────────────────────────────────────────────

    def _run_hierarchical(self, X_scaled, index, cl_cfg, out_dir, dataset_name, config):
        method = cl_cfg.get("linkage_method", "ward")
        n_clusters = cl_cfg.get("n_clusters", 3)
        # ward chỉ hỗ trợ euclidean — bỏ qua metric config nếu dùng ward
        metric = cl_cfg.get("hierarchical_metric", "euclidean")
        effective_metric = "euclidean" if method == "ward" else metric

        Z = linkage(X_scaled, method=method, metric=effective_metric)

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
        _plot_cluster_pca(X_scaled, labels, "hierarchical", dataset_name, out_dir)

        _append_summary({
            "dataset": dataset_name, "task": "clustering", "algorithm": "hierarchical",
            "n_clusters": n_clusters, "silhouette_score": round(float(sil), 4) if not np.isnan(sil) else "",
        })
        from src.infrastructure import tracker
        tracker.log_run(
            config,
            run_name=f"{dataset_name}/hierarchical",
            params={"algorithm": "hierarchical", "n_clusters": n_clusters},
            metrics={"silhouette_score": round(float(sil), 4) if not np.isnan(sil) else 0.0},
            tags={"task": "clustering", "dataset": dataset_name},
        )
        return {"n_clusters": n_clusters, "silhouette_score": sil}

    # ── DBSCAN ───────────────────────────────────────────────────────────────

    def _run_dbscan(self, X_scaled, index, cl_cfg, out_dir, dataset_name, config):
        eps = cl_cfg.get("dbscan_eps", 0.5)
        min_samples = cl_cfg.get("dbscan_min_samples", 5)
        metric = cl_cfg.get("dbscan_metric", "euclidean")

        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
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
        _plot_cluster_pca(X_scaled, labels, "dbscan", dataset_name, out_dir)

        _append_summary({
            "dataset": dataset_name, "task": "clustering", "algorithm": "dbscan",
            "n_clusters": n_clusters,
            "silhouette_score": round(float(sil), 4) if not np.isnan(sil) else "",
            "notes": f"outliers={n_outliers}",
        })
        from src.infrastructure import tracker
        tracker.log_run(
            config,
            run_name=f"{dataset_name}/dbscan",
            params={"algorithm": "dbscan", "n_clusters": n_clusters, "n_outliers": n_outliers},
            metrics={"silhouette_score": round(float(sil), 4) if not np.isnan(sil) else 0.0},
            tags={"task": "clustering", "dataset": dataset_name},
        )
        return {"n_clusters": n_clusters, "n_outliers": n_outliers, "silhouette_score": sil}


    # ── Gaussian Mixture Model ───────────────────────────────────────────────

    def _run_gaussian_mixture(self, X_scaled, index, cl_cfg, out_dir, dataset_name, config):
        n_components = cl_cfg.get("gmm_n_components", 3)
        covariance_type = cl_cfg.get("gmm_covariance_type", "full")

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=cl_cfg.get("gmm_max_iter", 100),
            n_init=cl_cfg.get("gmm_n_init", 3),
            tol=cl_cfg.get("gmm_tol", 0.001),
            random_state=self.random_state,
        )
        labels = gmm.fit_predict(X_scaled)

        try:
            sil = silhouette_score(X_scaled, labels) if n_components > 1 else np.nan
        except Exception:
            sil = np.nan

        sil_str = f"{sil:.4f}" if not np.isnan(sil) else "nan"
        logger.info(
            f"[CLUSTER] GMM: n_components={n_components}, "
            f"covariance={covariance_type}, silhouette={sil_str}"
        )

        labels_df = pd.DataFrame({"gmm_cluster": labels}, index=index)
        labels_df.to_csv(os.path.join(out_dir, "gmm_labels.csv"))
        _plot_cluster_pca(X_scaled, labels, "gmm", dataset_name, out_dir)

        # ── Subsampling CV: đánh giá độ ổn định silhouette ───────────────────────
        cv_n_splits = cl_cfg.get("cv_n_splits", 5)
        rng = np.random.RandomState(self.random_state)
        n_total = len(X_scaled)
        cv_sil_list_gmm = []
        for _ in range(cv_n_splits):
            idx = rng.choice(n_total, size=int(0.8 * n_total), replace=False)
            gmm_cv = GaussianMixture(
                n_components=n_components, covariance_type=covariance_type,
                max_iter=cl_cfg.get("gmm_max_iter", 100),
                n_init=cl_cfg.get("gmm_n_init", 3),
                tol=cl_cfg.get("gmm_tol", 0.001),
                random_state=self.random_state,
            )
            gmm_cv.fit(X_scaled[idx])
            labels_cv = gmm_cv.predict(X_scaled)
            try:
                cv_sil_list_gmm.append(
                    silhouette_score(X_scaled, labels_cv) if n_components > 1 else np.nan
                )
            except Exception:
                cv_sil_list_gmm.append(np.nan)
        cv_sil_mean = round(float(np.nanmean(cv_sil_list_gmm)), 4)
        cv_sil_std  = round(float(np.nanstd(cv_sil_list_gmm)),  4)
        logger.info(
            f"[CLUSTER] GMM subsampling CV ({cv_n_splits}x) | "
            f"Silhouette={cv_sil_mean}±{cv_sil_std}"
        )

        _append_summary({
            "dataset": dataset_name, "task": "clustering", "algorithm": "gaussian_mixture",
            "n_clusters": n_components,
            "silhouette_score": round(float(sil), 4) if not np.isnan(sil) else "",
            "cv_silhouette_mean": cv_sil_mean, "cv_silhouette_std": cv_sil_std,
        })
        from src.infrastructure import tracker
        tracker.log_run(
            config,
            run_name=f"{dataset_name}/gaussian_mixture",
            params={"algorithm": "gaussian_mixture", "n_components": n_components},
            metrics={
                "silhouette_score":   round(float(sil), 4) if not np.isnan(sil) else 0.0,
                "cv_silhouette_mean": cv_sil_mean,
                "cv_silhouette_std":  cv_sil_std,
            },
            tags={"task": "clustering", "dataset": dataset_name},
        )
        return {
            "n_clusters": n_components, "silhouette_score": sil,
            "cv_silhouette_mean": cv_sil_mean, "cv_silhouette_std": cv_sil_std,
        }


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
