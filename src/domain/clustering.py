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


def _safe_silhouette(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    max_samples: Optional[int] = None,
    random_state: int = 42,
) -> float:
    labels = np.asarray(labels)
    unique_labels = set(labels.tolist())
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return np.nan

    X_eval = X_scaled
    labels_eval = labels
    if max_samples and len(labels) > max_samples:
        rng = np.random.RandomState(random_state)
        sample_idx = rng.choice(len(labels), size=max_samples, replace=False)
        X_eval = X_scaled[sample_idx]
        labels_eval = labels[sample_idx]
        if len(set(labels_eval.tolist())) < 2:
            return np.nan

    try:
        return float(silhouette_score(X_eval, labels_eval))
    except Exception:
        return np.nan


def _nanmean(values: list) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan


def _sample_rows(
    X_scaled: np.ndarray,
    index,
    max_samples: Optional[int],
    random_state: int,
    algo_name: str,
):
    if not max_samples or len(X_scaled) <= max_samples:
        return X_scaled, index, np.arange(len(X_scaled)), False

    rng = np.random.RandomState(random_state)
    sample_idx = np.sort(rng.choice(len(X_scaled), size=max_samples, replace=False))
    logger.info(
        f"[CLUSTER] {algo_name} uses {max_samples}/{len(X_scaled)} sampled rows "
        f"to avoid excessive memory usage."
    )
    return X_scaled[sample_idx], index[sample_idx], sample_idx, True


def _savefig(fig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"[CLUSTER] Đã lưu biểu đồ → {path}")


def _plot_cluster_pca(
    X_scaled: np.ndarray, labels: np.ndarray,
    algo_name: str, dataset_name: str, out_dir: str,
    max_samples: Optional[int] = None, random_state: int = 42,
) -> None:
    """PCA 2D scatter — chiếu các cluster xuống 2D để trực quan hóa cấu trúc phân cụm."""
    try:
        from sklearn.decomposition import PCA
        labels = np.asarray(labels)
        if max_samples and len(X_scaled) > max_samples:
            rng = np.random.RandomState(random_state)
            sample_idx = np.sort(rng.choice(len(X_scaled), size=max_samples, replace=False))
            X_scaled = X_scaled[sample_idx]
            labels = labels[sample_idx]
            logger.info(
                f"[CLUSTER] PCA scatter for {algo_name} uses "
                f"{max_samples} sampled rows."
            )

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

def _export_cluster_profile(
    X_features: pd.DataFrame,
    labels: np.ndarray,
    algo_name: str,
    out_dir: str,
    top_n_log: int = 8,
) -> None:
    """Export descriptive statistics per cluster for interpretation."""
    labels = np.asarray(labels)
    if len(X_features) != len(labels):
        logger.warning(
            f"[CLUSTER] Skip profile for {algo_name}: feature rows "
            f"({len(X_features)}) != labels ({len(labels)})."
        )
        return

    profile_df = X_features.copy()
    profile_df.loc[:, "cluster"] = labels

    sizes = (
        profile_df["cluster"]
        .value_counts(dropna=False)
        .rename_axis("cluster")
        .reset_index(name="n_samples")
        .sort_values("cluster")
    )
    sizes_path = os.path.join(out_dir, f"cluster_sizes_{algo_name}.csv")
    sizes.to_csv(sizes_path, index=False, encoding="utf-8")

    feature_cols = [c for c in profile_df.columns if c != "cluster"]
    rows = []
    grouped = profile_df.groupby("cluster", dropna=False)
    for cluster_id, group in grouped:
        for feature in feature_cols:
            values = group[feature]
            row = {"cluster": cluster_id, "feature": feature, "n_samples": len(group)}
            row["mean"] = values.mean()
            row["std"] = values.std()
            row["median"] = values.median()
            row["min"] = values.min()
            row["max"] = values.max()
            rows.append(row)

    profile_path = os.path.join(out_dir, f"cluster_profile_{algo_name}.csv")
    pd.DataFrame(rows).to_csv(profile_path, index=False, encoding="utf-8")

    global_means = X_features.mean(numeric_only=True)
    cluster_means = grouped[feature_cols].mean(numeric_only=True)
    if not cluster_means.empty:
        diff = cluster_means.subtract(global_means, axis=1).abs()
        for cluster_id in diff.index:
            top_features = diff.loc[cluster_id].sort_values(ascending=False).head(top_n_log)
            summary = ", ".join(f"{name}={value:.4g}" for name, value in top_features.items())
            n_samples = int(sizes.loc[sizes["cluster"].eq(cluster_id), "n_samples"].iloc[0])
            logger.info(
                f"[CLUSTER] {algo_name} cluster={cluster_id} | n={n_samples} | "
                f"top differentiating features: {summary}"
            )

    logger.info(
        f"[CLUSTER] Cluster sizes -> {sizes_path}; "
        f"Cluster profile -> {profile_path}"
    )


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
        if X.empty:
            raise ValueError("KhÃ´ng cÃ³ feature sá»‘ há»£p lá»‡ Ä‘á»ƒ clustering.")

        variable_cols = X.columns[X.nunique(dropna=False) > 1].tolist()
        dropped_constant = [c for c in X.columns if c not in variable_cols]
        if dropped_constant:
            logger.info(
                f"[CLUSTER] Drop {len(dropped_constant)} constant feature(s): "
                f"{dropped_constant[:10]}"
            )
        X = X[variable_cols]
        if X.empty:
            raise ValueError("Táº¥t cáº£ feature sá»‘ Ä‘á»u háº±ng; cáº§n trá»‹ch xuáº¥t feature giÃ u thÃ´ng tin hÆ¡n.")

        scaler_method = cl_cfg.get("scaler", "standard")
        scaler = _get_scaler(scaler_method)
        X_scaled = scaler.fit_transform(X)

        results = {}
        results["kmeans"] = self._run_kmeans(X_scaled, X, cl_cfg, out_dir, model_dir, dataset_name, config)
        results["hierarchical"] = self._run_hierarchical(X_scaled, X, cl_cfg, out_dir, dataset_name, config)
        results["dbscan"] = self._run_dbscan(X_scaled, X, cl_cfg, out_dir, dataset_name, config)
        results["gaussian_mixture"] = self._run_gaussian_mixture(X_scaled, X, cl_cfg, out_dir, dataset_name, config)

        logger.info(f"[CLUSTER] Hoàn thành clustering cho {dataset_name}")
        return results

    # ── K-Means ──────────────────────────────────────────────────────────────

    def _run_kmeans(self, X_scaled, X_features, cl_cfg, out_dir, model_dir, dataset_name, config):
        index = X_features.index
        max_k = cl_cfg.get("max_k", 10)
        kmeans_init = cl_cfg.get("kmeans_init", "k-means++")
        kmeans_n_init = cl_cfg.get("kmeans_n_init", 10)
        kmeans_max_iter = cl_cfg.get("kmeans_max_iter", 300)
        silhouette_max_samples = cl_cfg.get("silhouette_max_samples")
        max_k = min(max_k, len(X_scaled) - 1)
        if max_k < 2:
            raise ValueError("Cáº§n Ã­t nháº¥t 3 samples há»£p lá»‡ Ä‘á»ƒ cháº¡y K-Means.")
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
            sil_scores.append(
                _safe_silhouette(
                    X_scaled, labels,
                    max_samples=silhouette_max_samples,
                    random_state=self.random_state,
                )
            )

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

        valid_scores = np.asarray(sil_scores, dtype=float)
        if np.isfinite(valid_scores).any():
            best_k = list(k_range)[int(np.nanargmax(valid_scores))]
            best_silhouette = float(np.nanmax(valid_scores))
        else:
            best_k = min(cl_cfg.get("n_clusters", 2), max_k)
            best_silhouette = np.nan
        best_silhouette_str = f"{best_silhouette:.4f}" if not np.isnan(best_silhouette) else "nan"
        logger.info(f"[CLUSTER] K-Means: best_k={best_k}, silhouette={best_silhouette_str}")

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
        _export_cluster_profile(X_features, final_labels, "kmeans", out_dir)
        _plot_cluster_pca(
            X_scaled, final_labels, "kmeans", dataset_name, out_dir,
            max_samples=cl_cfg.get("plot_max_samples"),
            random_state=self.random_state,
        )

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
                cv_sil_list.append(
                    _safe_silhouette(
                        X_scaled, labels_cv,
                        max_samples=silhouette_max_samples,
                        random_state=self.random_state,
                    )
                )
            except Exception:
                cv_sil_list.append(np.nan)
        cv_sil_mean_raw = _nanmean(cv_sil_list)
        cv_sil_std_raw = float(np.nanstd(cv_sil_list)) if np.isfinite(cv_sil_list).any() else np.nan
        cv_sil_mean = round(cv_sil_mean_raw, 4) if not np.isnan(cv_sil_mean_raw) else np.nan
        cv_sil_std  = round(cv_sil_std_raw,  4) if not np.isnan(cv_sil_std_raw) else np.nan
        logger.info(
            f"[CLUSTER] K-Means subsampling CV ({cv_n_splits}x) | "
            f"Silhouette={cv_sil_mean}±{cv_sil_std}"
        )

        result = {
            "n_clusters": best_k,
            "silhouette_score": round(best_silhouette, 4) if not np.isnan(best_silhouette) else np.nan,
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
                "silhouette_score":   result["silhouette_score"] if not np.isnan(result["silhouette_score"]) else 0.0,
                "cv_silhouette_mean": cv_sil_mean if not np.isnan(cv_sil_mean) else 0.0,
                "cv_silhouette_std":  cv_sil_std if not np.isnan(cv_sil_std) else 0.0,
            },
            tags={"task": "clustering", "dataset": dataset_name},
        )
        return result

    # ── Hierarchical ─────────────────────────────────────────────────────────

    def _run_hierarchical(self, X_scaled, X_features, cl_cfg, out_dir, dataset_name, config):
        index = X_features.index
        method = cl_cfg.get("linkage_method", "ward")
        n_clusters = cl_cfg.get("n_clusters", 3)
        max_samples = cl_cfg.get("hierarchical_max_samples")
        # ward chỉ hỗ trợ euclidean — bỏ qua metric config nếu dùng ward
        metric = cl_cfg.get("hierarchical_metric", "euclidean")
        effective_metric = "euclidean" if method == "ward" else metric

        X_h = X_scaled
        index_h = index
        X_h_features = X_features
        if max_samples and len(X_scaled) > max_samples:
            rng = np.random.RandomState(self.random_state)
            sample_idx = np.sort(rng.choice(len(X_scaled), size=max_samples, replace=False))
            X_h = X_scaled[sample_idx]
            index_h = index[sample_idx]
            X_h_features = X_features.iloc[sample_idx]
            logger.info(
                f"[CLUSTER] Hierarchical uses {max_samples}/{len(X_scaled)} sampled rows "
                f"to avoid O(n^2) linkage cost."
            )

        Z = linkage(X_h, method=method, metric=effective_metric)

        # Dendrogram (lấy mẫu tối đa 200 điểm để tránh quá tải)
        fig, ax = plt.subplots(figsize=(14, 6))
        dendrogram(
            Z, ax=ax, truncate_mode="lastp", p=min(50, len(X_h)),
            leaf_rotation=90, leaf_font_size=8,
        )
        ax.set_title(f"Dendrogram — {dataset_name} ({method} linkage)")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Distance")
        _savefig(fig, os.path.join(out_dir, "dendrogram.png"))

        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        sil = _safe_silhouette(
            X_h, labels,
            max_samples=cl_cfg.get("silhouette_max_samples"),
            random_state=self.random_state,
        )
        logger.info(f"[CLUSTER] Hierarchical: n_clusters={n_clusters}, silhouette={sil:.4f}")

        labels_df = pd.DataFrame({"hierarchical_cluster": labels}, index=index_h)
        labels_df.to_csv(os.path.join(out_dir, "hierarchical_labels.csv"))
        _export_cluster_profile(X_h_features, labels, "hierarchical", out_dir)
        _plot_cluster_pca(
            X_h, labels, "hierarchical", dataset_name, out_dir,
            max_samples=cl_cfg.get("plot_max_samples"),
            random_state=self.random_state,
        )

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

    def _run_dbscan(self, X_scaled, X_features, cl_cfg, out_dir, dataset_name, config):
        index = X_features.index
        eps = cl_cfg.get("dbscan_eps", 0.5)
        min_samples = cl_cfg.get("dbscan_min_samples", 5)
        metric = cl_cfg.get("dbscan_metric", "euclidean")
        max_samples = cl_cfg.get("dbscan_max_samples")

        X_db, index_db, sample_idx, sampled = _sample_rows(
            X_scaled, index, max_samples, self.random_state, "DBSCAN"
        )
        X_db_features = X_features.iloc[sample_idx] if sampled else X_features

        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = db.fit_predict(X_db)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = int((labels == -1).sum())
        logger.info(
            f"[CLUSTER] DBSCAN: eps={eps}, min_samples={min_samples} → "
            f"{n_clusters} clusters, {n_outliers} outliers ({n_outliers/len(labels)*100:.1f}%)"
        )

        sil = _safe_silhouette(
            X_db, labels,
            max_samples=cl_cfg.get("silhouette_max_samples"),
            random_state=self.random_state,
        ) if n_clusters > 1 else np.nan

        labels_df = pd.DataFrame({"dbscan_cluster": labels}, index=index_db)
        if sampled:
            labels_df["sampled_for_dbscan"] = True
        labels_df.to_csv(os.path.join(out_dir, "dbscan_labels.csv"))
        _export_cluster_profile(X_db_features, labels, "dbscan", out_dir)
        _plot_cluster_pca(
            X_db, labels, "dbscan", dataset_name, out_dir,
            max_samples=cl_cfg.get("plot_max_samples"),
            random_state=self.random_state,
        )

        _append_summary({
            "dataset": dataset_name, "task": "clustering", "algorithm": "dbscan",
            "n_clusters": n_clusters,
            "silhouette_score": round(float(sil), 4) if not np.isnan(sil) else "",
            "notes": f"outliers={n_outliers}" + (f";sampled={len(labels)}/{len(X_scaled)}" if sampled else ""),
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

    def _run_gaussian_mixture(self, X_scaled, X_features, cl_cfg, out_dir, dataset_name, config):
        index = X_features.index
        n_components = cl_cfg.get("gmm_n_components", 3)
        covariance_type = cl_cfg.get("gmm_covariance_type", "full")
        fit_max_samples = cl_cfg.get("gmm_fit_max_samples")
        cv_max_samples = cl_cfg.get("gmm_cv_max_samples", fit_max_samples)

        X_fit, _, _, sampled_fit = _sample_rows(
            X_scaled, index, fit_max_samples, self.random_state, "GMM fit"
        )

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=cl_cfg.get("gmm_max_iter", 100),
            n_init=cl_cfg.get("gmm_n_init", 3),
            tol=cl_cfg.get("gmm_tol", 0.001),
            random_state=self.random_state,
        )
        gmm.fit(X_fit)
        labels = gmm.predict(X_scaled)

        sil = _safe_silhouette(
            X_scaled, labels,
            max_samples=cl_cfg.get("silhouette_max_samples"),
            random_state=self.random_state,
        ) if n_components > 1 else np.nan

        sil_str = f"{sil:.4f}" if not np.isnan(sil) else "nan"
        logger.info(
            f"[CLUSTER] GMM: n_components={n_components}, "
            f"covariance={covariance_type}, silhouette={sil_str}"
        )

        labels_df = pd.DataFrame({"gmm_cluster": labels}, index=index)
        labels_df.to_csv(os.path.join(out_dir, "gmm_labels.csv"))
        _export_cluster_profile(X_features, labels, "gmm", out_dir)
        _plot_cluster_pca(
            X_scaled, labels, "gmm", dataset_name, out_dir,
            max_samples=cl_cfg.get("plot_max_samples"),
            random_state=self.random_state,
        )

        # ── Subsampling CV: đánh giá độ ổn định silhouette ───────────────────────
        cv_n_splits = cl_cfg.get("cv_n_splits", 5)
        rng = np.random.RandomState(self.random_state)
        n_total = len(X_scaled)
        cv_sample_size = int(0.8 * n_total)
        if cv_max_samples:
            cv_sample_size = min(cv_sample_size, int(cv_max_samples))
        cv_sil_list_gmm = []
        for _ in range(cv_n_splits):
            idx = rng.choice(n_total, size=cv_sample_size, replace=False)
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
                    _safe_silhouette(
                        X_scaled, labels_cv,
                        max_samples=cl_cfg.get("silhouette_max_samples"),
                        random_state=self.random_state,
                    ) if n_components > 1 else np.nan
                )
            except Exception:
                cv_sil_list_gmm.append(np.nan)
        cv_sil_mean_raw = _nanmean(cv_sil_list_gmm)
        cv_sil_std_raw = float(np.nanstd(cv_sil_list_gmm)) if np.isfinite(cv_sil_list_gmm).any() else np.nan
        cv_sil_mean = round(cv_sil_mean_raw, 4) if not np.isnan(cv_sil_mean_raw) else np.nan
        cv_sil_std  = round(cv_sil_std_raw,  4) if not np.isnan(cv_sil_std_raw) else np.nan
        logger.info(
            f"[CLUSTER] GMM subsampling CV ({cv_n_splits}x) | "
            f"Silhouette={cv_sil_mean}±{cv_sil_std}"
        )

        _append_summary({
            "dataset": dataset_name, "task": "clustering", "algorithm": "gaussian_mixture",
            "n_clusters": n_components,
            "silhouette_score": round(float(sil), 4) if not np.isnan(sil) else "",
            "cv_silhouette_mean": cv_sil_mean, "cv_silhouette_std": cv_sil_std,
            "notes": f"fit_sampled={len(X_fit)}/{len(X_scaled)}" if sampled_fit else "",
        })
        from src.infrastructure import tracker
        tracker.log_run(
            config,
            run_name=f"{dataset_name}/gaussian_mixture",
            params={"algorithm": "gaussian_mixture", "n_components": n_components},
            metrics={
                "silhouette_score":   round(float(sil), 4) if not np.isnan(sil) else 0.0,
                "cv_silhouette_mean": cv_sil_mean if not np.isnan(cv_sil_mean) else 0.0,
                "cv_silhouette_std":  cv_sil_std if not np.isnan(cv_sil_std) else 0.0,
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
