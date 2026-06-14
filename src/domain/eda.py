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


def plot_scatter_matrix(df: pd.DataFrame, quant_cols: list, out_dir: str) -> None:
    """Scatter plot matrix (pairplot) cho các cặp biến định lượng."""
    if len(quant_cols) < 2:
        return
    cols = quant_cols[:6]  # giới hạn 6 cột để tránh biểu đồ quá lớn
    fig = sns.pairplot(df[cols].dropna()).figure
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
        freq = df[col].value_counts().sort_index()
        pct  = df[col].value_counts(normalize=True).sort_index() * 100
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
    df = df.ffill().bfill()           # Forward Fill, rồi Backward Fill
    df = df.interpolate(method="linear")  # Lấp đầy khoảng trống còn lại
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Hàm chính: run_eda_pipeline
# ─────────────────────────────────────────────────────────────────────────────

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

    df.to_excel(os.path.join(raw_dir, "data_raw.xlsx"), index=(group == "B"))
    df.to_csv(os.path.join(raw_dir, "data_raw.csv"), index=(group == "B"))

    plot_boxplots(df, quant_cols, raw_dir)
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

    # ── Pipeline 2: Transformed ──────────────────────────────────────────────
    logger.info(f"[EDA] Pipeline 2 (transformed) — {dataset_name}")

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
        transforms = eda_cfg.get("group_b_transforms", [])
        if "differencing" in transforms:
            df_t = df_t.diff().dropna()
        if "log_transform" in transforms:
            num_cols = df_t.select_dtypes(include="number").columns
            df_t[num_cols] = np.log1p(df_t[num_cols].clip(lower=0))

    df_t.to_excel(os.path.join(trans_dir, "data_transformed.xlsx"), index=(group == "B"))
    df_t.to_csv(os.path.join(trans_dir, "data_transformed.csv"), index=(group == "B"))
    df_t.to_parquet(
        os.path.join(interim_dir, f"{dataset_name}_transformed.parquet"),
        index=(group == "B"),
    )
    logger.info(f"[EDA] Đã lưu transformed data → {trans_dir}/")

    # Biểu đồ sau biến đổi
    var_types_t = classify_variables(df_t)
    quant_cols_t = var_types_t["quantitative_continuous"] + var_types_t["quantitative_discrete"]
    plot_histograms(df_t, quant_cols_t, trans_dir)
    plot_boxplots(df_t, quant_cols_t, trans_dir)
    if group == "A":
        plot_scatter_matrix(df_t, quant_cols_t, trans_dir)
        plot_bar_charts(df_t, var_types_t["qualitative"], trans_dir)
    elif group == "B":
        plot_timeseries(df_t, trans_dir)
    plot_correlation_heatmap(df_t, quant_cols_t, trans_dir)

    logger.info(f"[EDA] Hoàn thành — {dataset_name}")
    return df_t
