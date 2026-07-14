"""
Domain — EDA (Exploratory Data Analysis)
Phân loại biến, tính thống kê mô tả đầy đủ, trực quan hóa,
và xuất Pipeline 1 (raw) + Pipeline 2 (transformed).
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — an toàn cho server

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.infrastructure.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Phân loại biến
# ─────────────────────────────────────────────────────────────────────────────

def classify_variables(df: pd.DataFrame) -> dict:
    """
    Tự động phân loại từng cột thành:
        - qualitative: object, category, bool
        - quantitative_discrete: int với số giá trị duy nhất < 20
        - quantitative_continuous: float hoặc int với số giá trị >= 20

    Returns:
        dict với 3 keys, mỗi key là list tên cột.
    """
    qual, q_disc, q_cont = [], [], []
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == "object" or dtype.name == "category" or dtype == "bool":
            qual.append(col)
        elif pd.api.types.is_integer_dtype(dtype):
            if df[col].nunique() < 20:
                q_disc.append(col)
            else:
                q_cont.append(col)
        elif pd.api.types.is_float_dtype(dtype):
            q_cont.append(col)
        else:
            qual.append(col)  # fallback

    result = {
        "qualitative": qual,
        "quantitative_discrete": q_disc,
        "quantitative_continuous": q_cont,
    }
    logger.info(
        f"[EDA] Phân loại biến — "
        f"Qualitative: {len(qual)}, "
        f"Quant.Discrete: {len(q_disc)}, "
        f"Quant.Continuous: {len(q_cont)}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. Thống kê mô tả
# ─────────────────────────────────────────────────────────────────────────────

def compute_statistics(df: pd.DataFrame, var_types: dict) -> dict:
    """
    Tính thống kê mô tả đầy đủ theo loại biến.

    Quantitative: Mean, Median, Mode, P5/P25/P50/P75/P95,
                  Range, Variance, StdDev, CV, IQR
    Qualitative:  Absolute freq, Relative freq (%), Mode

    Args:
        df:        DataFrame nguồn.
        var_types: Kết quả từ classify_variables().

    Returns:
        dict {column_name: stats_dict}
    """
    stats = {}
    quant_cols = var_types["quantitative_continuous"] + var_types["quantitative_discrete"]

    for col in quant_cols:
        s = df[col].dropna()
        if s.empty:
            stats[col] = {
                "type": "quantitative",
                "count": 0,
                "null_count": df[col].isna().sum(),
                "mean": np.nan,
                "median": np.nan,
                "mode": None,
                "p5": np.nan,
                "p25": np.nan,
                "p50": np.nan,
                "p75": np.nan,
                "p95": np.nan,
                "range": np.nan,
                "variance": np.nan,
                "std": np.nan,
                "cv": np.nan,
                "iqr": np.nan,
            }
            continue
        p = np.percentile(s, [5, 25, 50, 75, 95])
        mean_val = s.mean()
        iqr = p[3] - p[1]
        stats[col] = {
            "type": "quantitative",
            "count": len(s),
            "null_count": df[col].isna().sum(),
            "mean": round(mean_val, 4),
            "median": round(s.median(), 4),
            "mode": s.mode().iloc[0] if not s.mode().empty else None,
            "p5": round(p[0], 4),
            "p25": round(p[1], 4),
            "p50": round(p[2], 4),
            "p75": round(p[3], 4),
            "p95": round(p[4], 4),
            "range": round(float(s.max() - s.min()), 4),
            "variance": round(s.var(), 4),
            "std": round(s.std(), 4),
            "cv": round((s.std() / mean_val * 100) if mean_val != 0 else np.nan, 4),
            "iqr": round(iqr, 4),
        }

    for col in var_types["qualitative"]:
        freq = df[col].value_counts(dropna=False)
        rel_freq = df[col].value_counts(normalize=True, dropna=False) * 100
        stats[col] = {
            "type": "qualitative",
            "count": df[col].notna().sum(),
            "null_count": df[col].isna().sum(),
            "mode": df[col].mode().iloc[0] if not df[col].mode().empty else None,
            "absolute_freq": freq.to_dict(),
            "relative_freq_pct": {k: round(v, 2) for k, v in rel_freq.items()},
        }

    logger.info(f"[EDA] Đã tính thống kê cho {len(stats)} biến")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trực quan hóa
# ─────────────────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"[EDA] Đã lưu biểu đồ → {path}")


def plot_histograms(df: pd.DataFrame, quant_cols: list, out_dir: str) -> None:
    """Histogram cho từng biến định lượng."""
    for col in quant_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        df[col].dropna().hist(ax=ax, bins=30, edgecolor="black", color="#4C72B0")
        ax.set_title(f"Histogram — {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Tần suất")
        _savefig(fig, os.path.join(out_dir, f"hist_{col}.png"))


def plot_boxplots(df: pd.DataFrame, quant_cols: list, out_dir: str) -> None:
    """Boxplot cho từng biến định lượng (phát hiện outliers)."""
    for col in quant_cols:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(x=df[col].dropna(), ax=ax, color="#4C72B0")
        ax.set_title(f"Boxplot — {col}")
        _savefig(fig, os.path.join(out_dir, f"boxplot_{col}.png"))


def plot_scatter_matrix(
    df: pd.DataFrame,
    quant_cols: list,
    out_dir: str,
    max_samples: int = 5000,
) -> None:
    """Scatter plot matrix (pairplot) cho các cặp biến định lượng."""
    if len(quant_cols) < 2:
        return
    cols = quant_cols[:6]  # giới hạn 6 cột để tránh biểu đồ quá lớn
    plot_df = df[cols].dropna()
    if len(plot_df) > max_samples:
        plot_df = plot_df.sample(max_samples, random_state=42)
    fig = sns.pairplot(plot_df).figure
    _savefig(fig, os.path.join(out_dir, "scatter_matrix.png"))


def plot_bar_charts(df: pd.DataFrame, qual_cols: list, out_dir: str) -> None:
    """Bar chart và Pie chart cho biến định tính."""
    for col in qual_cols:
        freq = df[col].value_counts()

        # Bar chart
        fig, ax = plt.subplots(figsize=(max(6, len(freq) * 0.8), 4))
        freq.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="black")
        ax.set_title(f"Bar Chart — {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Số lượng")
        plt.xticks(rotation=45, ha="right")
        _savefig(fig, os.path.join(out_dir, f"bar_{col}.png"))

        # Pie chart (chỉ khi ≤10 giá trị unique)
        if freq.nunique() <= 10:
            fig, ax = plt.subplots(figsize=(6, 6))
            freq.plot(kind="pie", ax=ax, autopct="%1.1f%%", startangle=90)
            ax.set_ylabel("")
            ax.set_title(f"Pie Chart — {col}")
            _savefig(fig, os.path.join(out_dir, f"pie_{col}.png"))


def plot_timeseries(df: pd.DataFrame, out_dir: str) -> None:
    """Line chart (time on x-axis) — BẮT BUỘC cho nhóm B."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(12, 4))
        df[col].plot(ax=ax, linewidth=1, color="#4C72B0")
        ax.set_title(f"Time Series — {col}")
        ax.set_xlabel("Thời gian")
        ax.set_ylabel(col)
        _savefig(fig, os.path.join(out_dir, f"timeseries_{col}.png"))


def plot_correlation_heatmap(df: pd.DataFrame, quant_cols: list, out_dir: str) -> None:
    """Correlation heatmap giữa các biến định lượng — phát hiện đa cộng tuyến."""
    cols = [c for c in quant_cols if c in df.columns][:20]  # giới hạn 20 cột
    if len(cols) < 2:
        return
    corr = df[cols].corr()
    n = len(cols)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.75), max(6, n * 0.65)))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # chỉ hiện nửa dưới tam giác
    sns.heatmap(
        corr, mask=mask, ax=ax,
        annot=(n <= 12), fmt=".2f", annot_kws={"size": 7},
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.3, square=True,
    )
    ax.set_title("Correlation Heatmap")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    _savefig(fig, os.path.join(out_dir, "correlation_heatmap.png"))


def plot_null_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    """Heatmap trực quan hóa vị trí giá trị null (chỉ vẽ khi có null)."""
    null_count = df.isnull().sum()
    cols_with_null = null_count[null_count > 0].index.tolist()
    if not cols_with_null:
        return
    sample = df[cols_with_null].isnull()
    if len(sample) > 300:
        sample = sample.sample(300, random_state=42)
    fig, ax = plt.subplots(figsize=(max(6, len(cols_with_null) * 0.9), 5))
    sns.heatmap(sample, cbar=False, ax=ax, cmap=["#4C72B0", "#f0f0f0"], yticklabels=False)
    ax.set_title("Missing Values Map  (xanh = có dữ liệu, trắng = null)")
    ax.set_xlabel("Cột")
    plt.xticks(rotation=45, ha="right")
    _savefig(fig, os.path.join(out_dir, "null_heatmap.png"))


def plot_class_distribution(df: pd.DataFrame, target_cols: list, out_dir: str) -> None:
    """Bar chart phân phối nhãn từng cột mục tiêu — phát hiện class imbalance."""
    for col in target_cols:
        if col not in df.columns:
            continue
        values = df[col].astype("string").fillna("<missing>")
        freq = values.value_counts()
        pct = values.value_counts(normalize=True) * 100
        n_classes = len(freq)
        palette = sns.color_palette("tab10", n_classes)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Count
        bars = axes[0].bar(freq.index.astype(str), freq.values,
                           color=palette, edgecolor="black")
        for bar, v in zip(bars, freq.values):
            axes[0].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(freq) * 0.01,
                         f"{v:,}", ha="center", va="bottom", fontsize=9)
        axes[0].set_title(f"Class Distribution — {col} (count)")
        axes[0].set_xlabel("Class")
        axes[0].set_ylabel("Số lượng")

        # Percentage + đường ngưỡng 10%
        bars2 = axes[1].bar(pct.index.astype(str), pct.values,
                            color=palette, edgecolor="black")
        for bar, v in zip(bars2, pct.values):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 1,
                         f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
        axes[1].axhline(10, color="red", linestyle="--", linewidth=1.2,
                        label="Ngưỡng 10% imbalance")
        axes[1].set_title(f"Class Distribution — {col} (%)")
        axes[1].set_xlabel("Class")
        axes[1].set_ylabel("Tỉ lệ (%)")
        axes[1].set_ylim(0, min(110, max(pct.values) * 1.25))
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        _savefig(fig, os.path.join(out_dir, f"class_dist_{col}.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pipeline tiền xử lý
# ─────────────────────────────────────────────────────────────────────────────

def plot_group_c_multimedia_overview(
    df: pd.DataFrame,
    subtype: Optional[str],
    out_dir: str,
    max_samples: int = 3000,
) -> None:
    """Group C visuals: overall label balance and per-class numeric feature profiles."""
    label_col = "label" if "label" in df.columns else None
    categorical_cols = [
        c for c in ["label", "dataset", "sublabel"]
        if c in df.columns
    ]
    if categorical_cols:
        plot_class_distribution(df, categorical_cols, out_dir)

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns.tolist()
        if df[c].nunique(dropna=True) > 1
    ]
    if not numeric_cols:
        return

    if label_col:
        counts = df[label_col].value_counts(dropna=False).rename_axis(label_col).reset_index(name="count")
        counts["relative_pct"] = (counts["count"] / counts["count"].sum() * 100).round(4)
        counts.to_csv(os.path.join(out_dir, "group_c_class_counts.csv"), index=False)

        profile_cols = numeric_cols[:18]
        profile = df.groupby(label_col, dropna=False)[profile_cols].mean()
        profile.to_csv(os.path.join(out_dir, "group_c_feature_profile_by_class.csv"))
        scaled_profile = (profile - profile.mean()) / profile.std(ddof=0).replace(0, np.nan)
        scaled_profile = scaled_profile.fillna(0.0)
        fig, ax = plt.subplots(figsize=(max(9, len(profile_cols) * 0.6), max(5, len(profile) * 0.45)))
        sns.heatmap(scaled_profile, ax=ax, cmap="coolwarm", center=0, linewidths=0.3)
        ax.set_title("Group C Feature Profile by Class")
        ax.set_xlabel("Numeric feature")
        ax.set_ylabel("Class")
        plt.xticks(rotation=45, ha="right")
        _savefig(fig, os.path.join(out_dir, "group_c_feature_profile_heatmap.png"))

        priority = []
        if subtype == "image":
            priority = [
                "brightness_mean", "brightness_std", "edge_density",
                "red_mean", "green_mean", "blue_mean",
            ]
        elif subtype == "audio":
            priority = [
                "duration", "rms_mean", "zero_crossing_rate",
                "spectral_centroid_hz", "spectral_bandwidth_hz",
                "spectral_rolloff_85_hz",
            ]
        box_cols = [c for c in priority if c in numeric_cols]
        box_cols += [c for c in numeric_cols if c not in box_cols][: max(0, 6 - len(box_cols))]
        for col in box_cols[:6]:
            plot_df = df[[label_col, col]].dropna()
            if len(plot_df) > max_samples:
                plot_df = plot_df.sample(max_samples, random_state=42)
            fig, ax = plt.subplots(figsize=(max(8, plot_df[label_col].nunique() * 0.85), 4.5))
            sns.boxplot(data=plot_df, x=label_col, y=col, ax=ax, color="#4C72B0")
            ax.set_title(f"Group C Class-Level Distribution - {col}")
            ax.set_xlabel("Class")
            ax.set_ylabel(col)
            plt.xticks(rotation=45, ha="right")
            _savefig(fig, os.path.join(out_dir, f"group_c_class_box_{col}.png"))

    if len(numeric_cols) >= 2:
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            pca_cols = numeric_cols[:30]
            pca_df = df[pca_cols].copy()
            pca_df = pca_df.fillna(pca_df.median(numeric_only=True))
            valid = pca_df.notna().all(axis=1)
            pca_df = pca_df[valid]
            labels = df.loc[pca_df.index, label_col].astype(str) if label_col else None
            if len(pca_df) > max_samples:
                sample_idx = pca_df.sample(max_samples, random_state=42).index
                pca_df = pca_df.loc[sample_idx]
                labels = labels.loc[sample_idx] if labels is not None else None
            X = StandardScaler().fit_transform(pca_df)
            coords = PCA(n_components=2, random_state=42).fit_transform(X)
            fig, ax = plt.subplots(figsize=(8, 6))
            if labels is not None:
                for label in sorted(labels.unique()):
                    mask = labels == label
                    ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.6, label=label)
                ax.legend(fontsize=8, loc="best", ncol=max(1, len(labels.unique()) // 8))
            else:
                ax.scatter(coords[:, 0], coords[:, 1], s=18, alpha=0.6, color="#4C72B0")
            ax.set_title("Group C Numeric Feature PCA")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            _savefig(fig, os.path.join(out_dir, "group_c_pca_by_class.png"))
        except Exception as exc:
            logger.warning(f"[EDA] Skip Group C PCA plot: {exc}")

    if subtype == "image" and label_col and "path" in df.columns:
        try:
            from PIL import Image

            classes = df[label_col].dropna().astype(str).value_counts().index.tolist()[:9]
            samples = []
            for cls in classes:
                cls_rows = df[df[label_col].astype(str) == cls].head(3)
                for _, row in cls_rows.iterrows():
                    samples.append((cls, row["path"]))
            if samples:
                ncols = 3
                nrows = int(np.ceil(len(samples) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
                axes = np.array(axes).reshape(-1)
                for ax, (cls, path_value) in zip(axes, samples):
                    img = Image.open(path_value).convert("RGB")
                    ax.imshow(img)
                    ax.set_title(str(cls), fontsize=9)
                    ax.axis("off")
                for ax in axes[len(samples):]:
                    ax.axis("off")
                plt.tight_layout()
                _savefig(fig, os.path.join(out_dir, "group_c_image_samples_by_class.png"))
        except Exception as exc:
            logger.warning(f"[EDA] Skip Group C image samples: {exc}")


def export_frequency_tables(df: pd.DataFrame, qual_cols: list, out_dir: str) -> None:
    """Export long-form absolute and relative frequency tables for categorical columns."""
    rows = []
    for col in qual_cols:
        freq = df[col].value_counts(dropna=False)
        pct = df[col].value_counts(normalize=True, dropna=False) * 100
        for value, count in freq.items():
            rows.append({
                "column": col,
                "value": value,
                "count": int(count),
                "relative_freq_pct": round(float(pct.loc[value]), 4),
            })

        col_table = pd.DataFrame(rows[-len(freq):])
        col_table.to_csv(os.path.join(out_dir, f"frequency_{col}.csv"), index=False)

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "frequency_tables.csv"), index=False)


def _iqr_bounds(df: pd.DataFrame, quant_cols: list, multiplier: float, exclude: Optional[list] = None) -> dict:
    bounds = {}
    skip = set(exclude or [])
    for col in quant_cols:
        if col in skip or col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - multiplier * iqr, q3 + multiplier * iqr)
    return bounds


def _count_outliers(df: pd.DataFrame, bounds: dict) -> dict:
    counts = {}
    for col, (lower, upper) in bounds.items():
        if col in df.columns:
            counts[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
    return counts


def export_before_after_summary(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    var_types_before: dict,
    var_types_after: dict,
    out_dir: str,
    protected_cols: Optional[list] = None,
    iqr_multiplier: float = 1.5,
) -> None:
    """Export a compact before/after audit for missing values, outliers, and schema changes."""
    quant_before = var_types_before["quantitative_continuous"] + var_types_before["quantitative_discrete"]
    quant_after = var_types_after["quantitative_continuous"] + var_types_after["quantitative_discrete"]
    bounds = _iqr_bounds(df_before, quant_before, iqr_multiplier, exclude=protected_cols)
    out_before = _count_outliers(df_before, bounds)
    out_after = _count_outliers(df_after, bounds)

    all_cols = sorted(set(df_before.columns) | set(df_after.columns))
    rows = []
    for col in all_cols:
        rows.append({
            "column": col,
            "in_before": col in df_before.columns,
            "in_after": col in df_after.columns,
            "dtype_before": str(df_before[col].dtype) if col in df_before.columns else "",
            "dtype_after": str(df_after[col].dtype) if col in df_after.columns else "",
            "null_before": int(df_before[col].isna().sum()) if col in df_before.columns else "",
            "null_after": int(df_after[col].isna().sum()) if col in df_after.columns else "",
            "outliers_before_iqr": out_before.get(col, ""),
            "outliers_after_iqr_raw_bounds": out_after.get(col, ""),
            "min_before": df_before[col].min() if col in quant_before and col in df_before.columns else "",
            "max_before": df_before[col].max() if col in quant_before and col in df_before.columns else "",
            "min_after": df_after[col].min() if col in quant_after and col in df_after.columns else "",
            "max_after": df_after[col].max() if col in quant_after and col in df_after.columns else "",
        })

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "before_after_summary.csv"), index=False)
    pd.DataFrame([{
        "rows_before": len(df_before),
        "cols_before": len(df_before.columns),
        "rows_after": len(df_after),
        "cols_after": len(df_after.columns),
        "total_null_before": int(df_before.isna().sum().sum()),
        "total_null_after": int(df_after.isna().sum().sum()),
        "total_outliers_before_iqr": int(sum(out_before.values())),
        "total_outliers_after_iqr_raw_bounds": int(sum(out_after.values())),
    }]).to_csv(os.path.join(out_dir, "before_after_overview.csv"), index=False)


def _fill_nulls_group_a(df: pd.DataFrame, var_types: dict) -> pd.DataFrame:
    df = df.copy()
    for col in var_types["quantitative_continuous"]:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in var_types["quantitative_discrete"]:
        df[col].fillna(df[col].median(), inplace=True)
    for col in var_types["qualitative"]:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col].fillna(mode_val.iloc[0], inplace=True)
    return df


def _cap_outliers_iqr(
    df: pd.DataFrame,
    quant_cols: list,
    multiplier: float = 1.5,
    exclude: Optional[list] = None,
) -> pd.DataFrame:
    df = df.copy()
    skip = set(exclude or [])
    for col in quant_cols:
        if col in skip:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def _normalize(
    df: pd.DataFrame,
    quant_cols: list,
    method: str = "min-max",
    exclude: Optional[list] = None,
) -> pd.DataFrame:
    df = df.copy()
    skip = set(exclude or [])
    for col in quant_cols:
        if col in skip:
            continue
        if method == "min-max":
            mn, mx = df[col].min(), df[col].max()
            if mx != mn:
                df[col] = (df[col] - mn) / (mx - mn)
        else:  # z-score
            mean, std = df[col].mean(), df[col].std()
            if std != 0:
                df[col] = (df[col] - mean) / std
    return df


def _fill_nulls_group_b(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    if numeric_cols:
        method = "time" if isinstance(df.index, pd.DatetimeIndex) else "linear"
        try:
            df[numeric_cols] = df[numeric_cols].interpolate(method=method)
        except ValueError:
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

    if non_numeric_cols:
        df[non_numeric_cols] = df[non_numeric_cols].ffill().bfill()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Hàm chính: run_eda_pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _export_dataframe(df: pd.DataFrame, out_dir: str, stem: str, group: str, eda_cfg: dict) -> None:
    index = group == "B"
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    xlsx_path = os.path.join(out_dir, f"{stem}.xlsx")
    df.to_csv(csv_path, index=index)

    excel_max_rows = int(eda_cfg.get("excel_max_rows", 1_048_576))
    excel_sample_rows = int(eda_cfg.get("excel_sample_rows", 100_000))
    if len(df) <= excel_max_rows:
        df.to_excel(xlsx_path, index=index)
        return

    sample_n = min(excel_sample_rows, len(df), excel_max_rows)
    sample_path = os.path.join(out_dir, f"{stem}_sample.xlsx")
    df.head(sample_n).to_excel(sample_path, index=index)
    logger.warning(
        f"[EDA] {stem}.xlsx skipped because {len(df)} rows exceed Excel limit "
        f"{excel_max_rows}. Wrote CSV and {sample_n}-row Excel sample instead."
    )


def run_eda_pipeline(
    df: pd.DataFrame,
    dataset_name: str,
    group: str,
    config: dict,
) -> pd.DataFrame:
    """
    Chạy toàn bộ EDA pipeline cho một dataset.

    Bước 1 — Pipeline Raw: báo cáo null, boxplot trước xử lý.
    Bước 2 — Pipeline Transformed: điền khuyết, xử lý outlier,
              chuẩn hóa, xuất .xlsx/.csv/.parquet.

    Args:
        df:           DataFrame nguồn (Group A) hoặc DatetimeIndex DataFrame (B).
        dataset_name: Tên dataset (snake_case).
        group:        "A" | "B" | "C"
        config:       Dict từ params.yaml.

    Returns:
        DataFrame đã biến đổi (transformed).
    """
    base_out = config.get("base_output_dir", "outputs")
    raw_dir = os.path.join(base_out, dataset_name, "eda", "raw")
    trans_dir = os.path.join(base_out, dataset_name, "eda", "transformed")
    interim_dir = os.path.join(config.get("base_data_dir", "data"), "interim")
    for d in [raw_dir, trans_dir, interim_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    eda_cfg = config.get("eda", {})
    ds_cfg = (config.get("datasets") or {}).get(dataset_name, {})
    iqr_mult = eda_cfg.get("iqr_multiplier", 1.5)

    # ── Drop columns không cần thiết (ví dụ: id) ────────────────────────────
    drop_cols = ds_cfg.get("drop_columns", [])
    if drop_cols:
        existing_drop = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=existing_drop)
        logger.info(f"[EDA] Đã drop {len(existing_drop)} cột: {existing_drop}")
    var_types = classify_variables(df)
    quant_cols = var_types["quantitative_continuous"] + var_types["quantitative_discrete"]

    # ── Pipeline 1: Raw ──────────────────────────────────────────────────────
    logger.info(f"[EDA] Pipeline 1 (raw) — {dataset_name}")

    null_report = df.isnull().sum().rename("null_count").to_frame()
    null_report["null_pct"] = (null_report["null_count"] / len(df) * 100).round(2)
    null_report.to_csv(os.path.join(raw_dir, "null_report.csv"))
    logger.info(f"[EDA] Null report → {raw_dir}/null_report.csv")

    _export_dataframe(df, raw_dir, "data_raw", group, eda_cfg)

    plot_histograms(df, quant_cols, raw_dir)
    plot_boxplots(df, quant_cols, raw_dir)
    if group in {"A", "B"}:
        plot_scatter_matrix(
            df,
            quant_cols,
            raw_dir,
            max_samples=eda_cfg.get("plot_max_samples", 5000),
        )
    plot_correlation_heatmap(df, quant_cols, raw_dir)
    plot_null_heatmap(df, raw_dir)
    _protect = ds_cfg.get("protect_columns", [])
    if _protect:
        plot_class_distribution(df, _protect, raw_dir)
    if group == "B":
        plot_timeseries(df, raw_dir)

    # Tính thống kê và lưu
    stats = compute_statistics(df, var_types)
    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv(os.path.join(raw_dir, "statistics.csv"))
    export_frequency_tables(df, var_types["qualitative"], raw_dir)
    if group == "C":
        plot_group_c_multimedia_overview(
            df,
            ds_cfg.get("subtype"),
            raw_dir,
            max_samples=eda_cfg.get("plot_max_samples", 3000),
        )

    # ── Pipeline 2: Transformed ──────────────────────────────────────────────
    logger.info(f"[EDA] Pipeline 2 (transformed) — {dataset_name}")

    outlier_method = eda_cfg.get("outlier_method", "iqr-cap")

    if group == "A":
        df_t = _fill_nulls_group_a(df, var_types)
        protect_cols = ds_cfg.get("protect_columns", [])
        outlier_method = eda_cfg.get("outlier_method", "iqr-cap")
        iqr_mult = eda_cfg.get("iqr_multiplier", 1.5)
        if outlier_method == "iqr-cap":
            df_t = _cap_outliers_iqr(df_t, quant_cols, multiplier=iqr_mult, exclude=protect_cols)
        norm_method = eda_cfg.get("normalization", "min-max")
        df_t = _normalize(df_t, quant_cols, method=norm_method, exclude=protect_cols)
        if protect_cols:
            logger.info(f"[EDA] Bảo vệ {len(protect_cols)} cột khỏi cap/normalize: {protect_cols}")

        # One-hot encode categorical columns nếu encode_categoricals = true
        if ds_cfg.get("encode_categoricals", False):
            qual_cols = classify_variables(df_t)["qualitative"]
            if qual_cols:
                df_t = pd.get_dummies(df_t, columns=qual_cols, drop_first=False)
                # Ép các cột bool mới về int để tương thích downstream
                bool_cols = df_t.select_dtypes(include="bool").columns.tolist()
                df_t[bool_cols] = df_t[bool_cols].astype(int)
                logger.info(
                    f"[EDA] One-hot encoded {len(qual_cols)} cột → "
                    f"shape sau encode: {df_t.shape}"
                )
    else:  # Group B & C
        df_t = _fill_nulls_group_b(df)
        if group == "B" and outlier_method == "iqr-cap":
            protected_b = [
                c for c in [ds_cfg.get("target_col")]
                if c
            ] + ds_cfg.get("clustering", {}).get("label_columns", [])
            numeric_cols_b = df_t.select_dtypes(include="number").columns.tolist()
            df_t = _cap_outliers_iqr(
                df_t,
                numeric_cols_b,
                multiplier=iqr_mult,
                exclude=protected_b,
            )
        transforms = eda_cfg.get("group_b_transforms", [])
        if "differencing" in transforms:
            df_t = df_t.diff().dropna()
        if "log_transform" in transforms:
            num_cols = df_t.select_dtypes(include="number").columns
            df_t[num_cols] = np.log1p(df_t[num_cols].clip(lower=0))

    _export_dataframe(df_t, trans_dir, "data_transformed", group, eda_cfg)
    df_t.to_parquet(
        os.path.join(interim_dir, f"{dataset_name}_transformed.parquet"),
        index=(group == "B"),
    )
    logger.info(f"[EDA] Đã lưu transformed data → {trans_dir}/")

    # Biểu đồ sau biến đổi
    var_types_t = classify_variables(df_t)
    quant_cols_t = var_types_t["quantitative_continuous"] + var_types_t["quantitative_discrete"]
    stats_t = compute_statistics(df_t, var_types_t)
    pd.DataFrame(stats_t).T.to_csv(os.path.join(trans_dir, "statistics.csv"))
    export_frequency_tables(df_t, var_types_t["qualitative"], trans_dir)
    if group == "C":
        plot_group_c_multimedia_overview(
            df_t,
            ds_cfg.get("subtype"),
            trans_dir,
            max_samples=eda_cfg.get("plot_max_samples", 3000),
        )

    protected_for_summary = ds_cfg.get("protect_columns", [])
    if group == "B":
        protected_for_summary = [
            c for c in [ds_cfg.get("target_col")]
            if c
        ] + ds_cfg.get("clustering", {}).get("label_columns", [])
    export_before_after_summary(
        df,
        df_t,
        var_types,
        var_types_t,
        trans_dir,
        protected_cols=protected_for_summary,
        iqr_multiplier=iqr_mult,
    )
    plot_histograms(df_t, quant_cols_t, trans_dir)
    plot_boxplots(df_t, quant_cols_t, trans_dir)
    if group == "A":
        plot_scatter_matrix(df_t, quant_cols_t, trans_dir)
        plot_bar_charts(df_t, var_types_t["qualitative"], trans_dir)
    elif group == "B":
        plot_scatter_matrix(
            df_t,
            quant_cols_t,
            trans_dir,
            max_samples=eda_cfg.get("plot_max_samples", 5000),
        )
        plot_timeseries(df_t, trans_dir)
    plot_correlation_heatmap(df_t, quant_cols_t, trans_dir)

    logger.info(f"[EDA] Hoàn thành — {dataset_name}")
    return df_t
