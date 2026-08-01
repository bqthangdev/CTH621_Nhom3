"""
Domain — EDA (Exploratory Data Analysis)
Phân loại biến, tính thống kê mô tả đầy đủ, trực quan hóa,
và xuất Pipeline 1 (raw) + Pipeline 2 (transformed).
"""

import os
import textwrap
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
from src.infrastructure.output_paths import dataset_output_root, transformed_interim_path

logger = get_logger(__name__)
warnings.filterwarnings("ignore")
_FIGURE_DPI = 300


def _display_label(value, width: int = 26) -> str:
    """Wrap long labels so figures remain legible on an A4 page."""
    text = str(value).replace("_", " ")
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Phân loại biến
# ─────────────────────────────────────────────────────────────────────────────

def classify_variables(df: pd.DataFrame, overrides: Optional[dict] = None) -> dict:
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

    overrides = overrides or {}
    configured_qual = [c for c in overrides.get("categorical_columns", []) if c in df.columns]
    configured_cont = [c for c in overrides.get("continuous_columns", []) if c in df.columns]
    configured_disc = [c for c in overrides.get("discrete_columns", []) if c in df.columns]
    configured = set(configured_qual + configured_cont + configured_disc)

    # Configuration describes statistical meaning, which takes precedence over
    # storage dtype (for example Course and binary symptoms are coded integers).
    qual = [c for c in qual if c not in configured] + configured_qual
    q_disc = [c for c in q_disc if c not in configured] + configured_disc
    q_cont = [c for c in q_cont if c not in configured] + configured_cont

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
    fig.savefig(path, bbox_inches="tight", dpi=_FIGURE_DPI)
    plt.close(fig)
    logger.info(f"[EDA] Đã lưu biểu đồ → {path}")


def plot_histograms(df: pd.DataFrame, quant_cols: list, out_dir: str) -> None:
    """Histogram + density with reference statistics for numeric variables."""
    for col in quant_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(
            values, bins="auto", kde=values.nunique() > 5, ax=ax,
            color="#4C72B0", edgecolor="white", linewidth=0.5,
        )
        mean_value, median_value = values.mean(), values.median()
        ax.axvline(mean_value, color="#C44E52", linestyle="--", linewidth=1.4,
                   label=f"Mean = {mean_value:,.3g}")
        ax.axvline(median_value, color="#55A868", linestyle=":", linewidth=1.6,
                   label=f"Median = {median_value:,.3g}")
        ax.set_title(f"Distribution of {_display_label(col, 38)}")
        ax.set_xlabel(_display_label(col, 38))
        ax.set_ylabel("Number of observations")
        ax.legend(frameon=True, fontsize=8)
        _savefig(fig, os.path.join(out_dir, f"hist_{col}.png"))


def plot_boxplots(df: pd.DataFrame, quant_cols: list, out_dir: str) -> None:
    """Horizontal boxplot with a mean marker for outlier and skew inspection."""
    for col in quant_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(x=values, ax=ax, color="#4C72B0", width=0.42,
                    showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white",
                               "markeredgecolor": "#C44E52", "markersize": 6})
        ax.set_title(f"Spread and potential outliers — {_display_label(col, 34)}")
        ax.set_xlabel(_display_label(col, 38))
        ax.set_ylabel("")
        ax.text(0.99, 0.93, f"n = {len(values):,}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
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
    grid = sns.pairplot(
        plot_df, diag_kind="hist", corner=True, height=1.7, aspect=1,
        plot_kws={"s": 12, "alpha": 0.45, "linewidth": 0},
        diag_kws={"bins": 24, "edgecolor": "white"},
    )
    for ax in grid.axes.flatten():
        if ax is None:
            continue
        ax.set_xlabel(_display_label(ax.get_xlabel(), 18))
        ax.set_ylabel(_display_label(ax.get_ylabel(), 18))
    grid.figure.suptitle(
        f"Pairwise relationships among selected numeric variables (n={len(plot_df):,})",
        y=1.01, fontsize=12,
    )
    fig = grid.figure
    _savefig(fig, os.path.join(out_dir, "scatter_matrix.png"))


def plot_bar_charts(df: pd.DataFrame, qual_cols: list, out_dir: str) -> None:
    """Bar chart và Pie chart cho biến định tính."""
    for col in qual_cols:
        freq = df[col].astype("string").fillna("<missing>").value_counts()

        # Horizontal bars remain readable for high-cardinality categories on A4.
        if len(freq) > 12:
            fig, ax = plt.subplots(figsize=(9, min(8.5, max(4.5, len(freq) * 0.27))))
            ordered = freq.sort_values(ascending=True)
            bars = ordered.plot(kind="barh", ax=ax, color="#4C72B0", edgecolor="black")
            ax.set_yticklabels([_display_label(v, 22) for v in ordered.index])
            ax.set_xlabel("Number of observations")
            ax.set_ylabel(_display_label(col, 36))
            ax.bar_label(bars.containers[0], fmt="{:,.0f}", padding=2, fontsize=7)
            ax.margins(x=0.12)
        else:
            fig, ax = plt.subplots(figsize=(max(6, min(11, len(freq) * 0.75)), 4.5))
            bars = freq.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="black")
            ax.set_xlabel(_display_label(col, 36))
            ax.set_ylabel("Number of observations")
            ax.set_xticklabels([_display_label(v, 16) for v in freq.index], rotation=35, ha="right")
            ax.bar_label(bars.containers[0], fmt="{:,.0f}", padding=2, fontsize=7)
            ax.margins(y=0.14)
        ax.set_title(f"Category distribution — {_display_label(col, 36)}")
        _savefig(fig, os.path.join(out_dir, f"bar_{col}.png"))

        # Pie chart (chỉ khi ≤10 giá trị unique)
        if freq.nunique() <= 10 and len(freq) <= 20:
            fig, ax = plt.subplots(figsize=(7.2, 5.4))
            wedges, _, _ = ax.pie(
                freq.values, labels=None, autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
                startangle=90, pctdistance=0.78,
                colors=sns.color_palette("colorblind", len(freq)),
                textprops={"fontsize": 8},
            )
            ax.legend(wedges, [_display_label(v, 22) for v in freq.index],
                      title=_display_label(col, 24), loc="center left",
                      bbox_to_anchor=(1.02, 0.5), fontsize=8)
            ax.set_ylabel("")
            ax.set_title(f"Composition by {_display_label(col, 34)}")
            _savefig(fig, os.path.join(out_dir, f"pie_{col}.png"))


def _even_sample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    positions = np.linspace(0, len(df) - 1, max_points, dtype=int)
    return df.iloc[positions]


def _coerce_configured_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Convert currency/percentage-like configured measures to numeric safely."""
    result = df.copy()
    for col in columns:
        if col not in result.columns or pd.api.types.is_numeric_dtype(result[col]):
            continue
        original = result[col]
        cleaned = (
            original.astype("string")
            .str.replace(r"[$,%]", "", regex=True)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        non_null = int(original.notna().sum())
        if non_null and int(numeric.notna().sum()) >= max(1, int(non_null * 0.9)):
            result.loc[:, col] = numeric
            logger.info(f"[EDA] Coerced configured quantitative column to numeric: {col}")
    return result


def plot_timeseries(
    df: pd.DataFrame,
    out_dir: str,
    columns: Optional[list] = None,
    max_points: int = 12000,
    group_col: Optional[str] = None,
    stage: str = "Raw",
) -> None:
    """Publication-sized time-series views, optionally separated by entity."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    columns = [c for c in (columns or numeric_cols) if c in numeric_cols]
    for col in columns:
        if group_col and group_col in df.columns:
            groups = df[group_col].dropna().value_counts().index.tolist()[:4]
            if not groups:
                continue
            fig, axes = plt.subplots(len(groups), 1, figsize=(11, 2.4 * len(groups)), sharex=False)
            axes = np.atleast_1d(axes)
            for ax, group_value in zip(axes, groups):
                sample = _even_sample(df[df[group_col] == group_value][[col]], max_points)
                ax.plot(sample.index, sample[col], linewidth=0.8, color="#4C72B0")
                ax.set_ylabel(col)
                ax.set_title(f"{group_col}={group_value}", loc="left", fontsize=10)
            axes[-1].set_xlabel("Time")
            fig.suptitle(f"{stage} time series — {col} (separated by {group_col})", y=1.01)
            plt.tight_layout()
        else:
            sample = _even_sample(df[[col]], max_points)
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.plot(sample.index, sample[col], linewidth=0.9, color="#4C72B0")
            ax.set_title(f"{stage} time series — {col}")
            ax.set_xlabel("Time")
            ax.set_ylabel(col)
            ax.grid(alpha=0.25)
        _savefig(fig, os.path.join(out_dir, f"timeseries_{col}.png"))


def plot_temporal_diagnostics(
    df: pd.DataFrame,
    target_col: str,
    out_dir: str,
    seasonal_period: Optional[int] = None,
    max_points: int = 12000,
) -> None:
    """Calendar profiles, rolling trend, and autocorrelation for time-series targets."""
    if target_col not in df.columns or not isinstance(df.index, pd.DatetimeIndex):
        return
    series = df[target_col].dropna().sort_index()
    if series.empty:
        return

    if len(series) > max_points:
        sample_positions = np.linspace(0, len(series) - 1, max_points, dtype=int)
    else:
        sample_positions = np.arange(len(series))
    sample = series.iloc[sample_positions]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(sample.index, sample.values, color="#4C72B0", linewidth=0.7, alpha=0.65, label="Observed")
    window = max(2, int(seasonal_period or max(2, len(series) // 100)))
    rolling = series.rolling(window, min_periods=max(2, window // 4)).mean()
    # Position-based selection supports legitimate duplicate timestamps.
    rolling_sample = rolling.iloc[sample_positions]
    ax.plot(rolling_sample.index, rolling_sample.values, color="#DD8452", linewidth=1.5,
            label=f"Rolling mean ({window})")
    ax.set_title(f"Trend and rolling level — {target_col}")
    ax.set_xlabel("Time")
    ax.set_ylabel(target_col)
    ax.legend()
    _savefig(fig, os.path.join(out_dir, f"temporal_trend_{target_col}.png"))

    profile = pd.DataFrame({"value": series})
    profile["hour"] = profile.index.hour
    profile["day_of_week"] = profile.index.dayofweek
    profile["month"] = profile.index.month
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    profile.groupby("hour")["value"].mean().plot(ax=axes[0], marker="o", color="#4C72B0")
    axes[0].set_title("Mean profile by hour")
    axes[0].set_xlabel("Hour")
    profile.groupby("day_of_week")["value"].mean().plot(ax=axes[1], marker="o", color="#55A868")
    axes[1].set_title("Mean profile by day of week")
    axes[1].set_xlabel("Day (Mon=0)")
    profile.groupby("month")["value"].mean().plot(ax=axes[2], marker="o", color="#C44E52")
    axes[2].set_title("Mean profile by month")
    axes[2].set_xlabel("Month")
    for ax in axes:
        ax.set_ylabel(target_col)
        ax.grid(alpha=0.25)
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, f"seasonal_profiles_{target_col}.png"))

    if profile["hour"].nunique() > 1 and profile["day_of_week"].nunique() > 1:
        pivot = profile.pivot_table(index="day_of_week", columns="hour", values="value", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(11, 4.5))
        sns.heatmap(pivot, cmap="viridis", ax=ax, cbar_kws={"label": f"Mean {target_col}"})
        ax.set_title(f"Calendar heatmap — {target_col}")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Day of week (Mon=0)")
        _savefig(fig, os.path.join(out_dir, f"calendar_heatmap_{target_col}.png"))

    try:
        from statsmodels.graphics.tsaplots import plot_acf

        acf_values = series.iloc[: min(len(series), 50000)]
        nlags = min(max(20, int(seasonal_period or 24) * 3), max(1, len(acf_values) // 4))
        fig, ax = plt.subplots(figsize=(10, 4.5))
        plot_acf(acf_values, lags=nlags, ax=ax, zero=False, alpha=0.05)
        ax.set_title(f"Autocorrelation — {target_col}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        _savefig(fig, os.path.join(out_dir, f"acf_{target_col}.png"))
    except ImportError:
        values = series.iloc[: min(len(series), 50000)].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) > 3:
            values = values - values.mean()
            denom = float(np.dot(values, values))
            nlags = min(max(20, int(seasonal_period or 24) * 3), max(1, len(values) // 4))
            correlations = [1.0]
            for lag in range(1, nlags + 1):
                correlations.append(
                    float(np.dot(values[:-lag], values[lag:]) / denom) if denom else 0.0
                )
            fig, ax = plt.subplots(figsize=(10, 4.5))
            markerline, stemlines, baseline = ax.stem(range(nlags + 1), correlations)
            plt.setp(markerline, markersize=3)
            plt.setp(stemlines, linewidth=0.8)
            baseline.set_linewidth(0.8)
            confidence = 1.96 / np.sqrt(len(values))
            ax.axhspan(-confidence, confidence, color="#4C72B0", alpha=0.15)
            ax.set_title(f"Autocorrelation — {target_col}")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Autocorrelation")
            _savefig(fig, os.path.join(out_dir, f"acf_{target_col}.png"))
    except Exception as exc:
        logger.warning(f"[EDA] Skip ACF for {target_col}: {exc}")


def plot_correlation_heatmap(
    df: pd.DataFrame, quant_cols: list, out_dir: str, max_features: int = 18,
) -> None:
    """Correlation heatmap giữa các biến định lượng — phát hiện đa cộng tuyến."""
    cols = [c for c in quant_cols if c in df.columns][:max_features]
    if len(cols) < 2:
        return
    corr = df[cols].corr().fillna(0)
    corr.index = [_display_label(value, 18) for value in corr.index]
    corr.columns = [_display_label(value, 18) for value in corr.columns]
    n = len(cols)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.75), max(6, n * 0.65)))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # chỉ hiện nửa dưới tam giác
    sns.heatmap(
        corr, mask=mask, ax=ax,
        annot=(n <= 12), fmt=".2f", annot_kws={"size": 7},
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.3, square=True,
    )
    ax.set_title("Pearson correlation between numeric variables")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    _savefig(fig, os.path.join(out_dir, "correlation_heatmap.png"))

    if 3 <= n <= 30:
        try:
            cluster_grid = sns.clustermap(
                corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                linewidths=0.15, figsize=(max(8, n * 0.62), max(7, n * 0.58)),
                cbar_kws={"label": "Pearson r"},
            )
            cluster_grid.ax_heatmap.set_xlabel("Variables grouped by correlation pattern")
            cluster_grid.ax_heatmap.set_ylabel("Variables grouped by correlation pattern")
            cluster_grid.fig.suptitle("Hierarchically clustered correlation map", y=1.01)
            _savefig(
                cluster_grid.fig,
                os.path.join(out_dir, "correlation_clustered_heatmap.png"),
            )
        except Exception as exc:
            logger.warning(f"[EDA] Skip clustered correlation heatmap: {exc}")


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


def plot_missingness_summary(df: pd.DataFrame, out_dir: str, max_features: int = 20) -> None:
    """Ranked missing-value percentages with counts for report-ready interpretation."""
    counts = df.isna().sum().sort_values(ascending=False).head(max_features)
    percentages = counts / max(len(df), 1) * 100
    labels = [_display_label(value, 24) for value in counts.index]
    fig_height = max(4.2, len(counts) * 0.32)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    colors = ["#C44E52" if value > 0 else "#B8B8B8" for value in percentages]
    bars = ax.barh(labels[::-1], percentages.values[::-1], color=colors[::-1])
    max_pct = max(float(percentages.max()), 1.0)
    ax.set_xlim(0, max_pct * 1.22)
    for bar, pct, count in zip(bars, percentages.values[::-1], counts.values[::-1]):
        ax.text(bar.get_width() + max_pct * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{int(count):,} ({pct:.1f}%)", va="center", fontsize=8)
    ax.set_title(f"Missing-data profile (n={len(df):,})")
    ax.set_xlabel("Missing observations (%)")
    ax.set_ylabel("Variable")
    _savefig(fig, os.path.join(out_dir, "missingness_summary.png"))


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


def plot_categorical_distributions(
    df: pd.DataFrame,
    categorical_cols: list,
    out_dir: str,
    max_categories: int = 20,
    stage: str = "Raw",
) -> None:
    """Horizontal count charts with both n and %, robust to long category labels."""
    for col in categorical_cols:
        if col not in df.columns:
            continue
        values = df[col].astype("string").fillna("<missing>")
        freq = values.value_counts(dropna=False)
        if freq.empty:
            continue
        if len(freq) > max_categories:
            top = freq.head(max_categories - 1)
            freq = pd.concat([top, pd.Series({"Other": int(freq.iloc[max_categories - 1:].sum())})])
        pct = freq / len(values) * 100
        height = min(10, max(3.5, len(freq) * 0.38))
        fig, ax = plt.subplots(figsize=(9, height))
        bars = ax.barh(freq.index.astype(str), freq.values, color=sns.color_palette("colorblind")[0])
        ax.invert_yaxis()
        offset = max(freq.max() * 0.01, 0.5)
        for bar, count, share in zip(bars, freq.values, pct.values):
            ax.text(count + offset, bar.get_y() + bar.get_height() / 2,
                    f"{int(count):,} ({share:.1f}%)", va="center", fontsize=8)
        ax.set_xlim(0, freq.max() * 1.28)
        ax.set_title(f"{stage} categorical distribution — {col}")
        ax.set_xlabel("Number of observations")
        ax.set_ylabel(col)
        ax.grid(axis="x", alpha=0.25)
        _savefig(fig, os.path.join(out_dir, f"categorical_{col}.png"))


def plot_legacy_categorical_compatibility(
    df: pd.DataFrame,
    columns: list,
    out_dir: str,
    measure_col: str = "duration",
) -> None:
    """Keep legacy filenames while replacing invalid categorical hist/box plots."""
    for col in columns:
        if col not in df.columns:
            continue
        counts = df[col].fillna("Missing").astype(str).value_counts()
        if counts.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(counts.index, counts.values, color="#4C72B0")
        ax.set_title(f"Category frequency — {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Number of observations")
        ax.tick_params(axis="x", rotation=35)
        for bar, value in zip(bars, counts.values):
            ax.annotate(
                f"{int(value):,}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=8,
            )
        _savefig(fig, os.path.join(out_dir, f"hist_{col}.png"))

        if measure_col in df.columns and pd.api.types.is_numeric_dtype(df[measure_col]):
            plot_df = df[[col, measure_col]].dropna()
            if not plot_df.empty:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                sns.boxplot(data=plot_df, x=col, y=measure_col, color="#55A868", ax=ax)
                ax.set_title(f"{measure_col} distribution by {col}")
                ax.set_xlabel(col)
                ax.set_ylabel(measure_col)
                ax.tick_params(axis="x", rotation=35)
                _savefig(fig, os.path.join(out_dir, f"boxplot_{col}.png"))


def plot_target_relationships(
    df: pd.DataFrame,
    target_cols: list,
    numeric_cols: list,
    categorical_cols: list,
    out_dir: str,
    max_categories: int = 12,
    max_samples: int = 5000,
) -> None:
    """Class-conditioned distributions and normalized categorical relationships."""
    for target in [c for c in target_cols if c in df.columns]:
        target_values = df[target].astype("string").fillna("<missing>")
        if target_values.nunique() > max_categories:
            continue

        for col in [c for c in numeric_cols if c in df.columns][:6]:
            plot_df = df[[target, col]].dropna()
            if plot_df.empty:
                continue
            if len(plot_df) > max_samples:
                plot_df = plot_df.sample(max_samples, random_state=42)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            sns.violinplot(data=plot_df, x=target, y=col, inner="quartile", cut=0,
                           palette="colorblind", hue=target, legend=False, ax=axes[0])
            axes[0].set_title(f"Distribution of {col} by {target}")
            axes[0].tick_params(axis="x", rotation=30)
            sns.ecdfplot(data=plot_df, x=col, hue=target, palette="colorblind", ax=axes[1])
            axes[1].set_title(f"Cumulative distribution of {col}")
            axes[1].set_ylabel("Cumulative proportion")
            plt.tight_layout()
            _savefig(fig, os.path.join(out_dir, f"target_numeric_{target}_{col}.png"))

        for col in [c for c in categorical_cols if c in df.columns and c != target][:6]:
            values = df[col].astype("string").fillna("<missing>")
            top_levels = values.value_counts().head(max_categories).index
            plot_df = pd.DataFrame({col: values, target: target_values})
            plot_df = plot_df[plot_df[col].isin(top_levels)]
            table = pd.crosstab(plot_df[col], plot_df[target], normalize="index") * 100
            if table.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, max(4, len(table) * 0.4)))
            table.plot(kind="barh", stacked=True, ax=ax, colormap="tab10", edgecolor="white")
            ax.set_title(f"Class composition of {target} within {col}")
            ax.set_xlabel("Percentage within category (%)")
            ax.set_ylabel(col)
            ax.legend(title=target, bbox_to_anchor=(1.02, 1), loc="upper left")
            ax.grid(axis="x", alpha=0.25)
            plt.tight_layout()
            _savefig(fig, os.path.join(out_dir, f"target_categorical_{target}_{col}.png"))


def plot_before_after_comparison(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    columns: list,
    out_dir: str,
    max_samples: int = 10000,
) -> None:
    """Side-by-side distribution audit so preprocessing effects are visible."""
    for col in [c for c in columns if c in df_before.columns and c in df_after.columns][:8]:
        before = df_before[col].dropna()
        after = df_after[col].dropna()
        if before.empty or after.empty:
            continue
        if len(before) > max_samples:
            before = before.sample(max_samples, random_state=42)
        if len(after) > max_samples:
            after = after.sample(max_samples, random_state=42)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        sns.histplot(before, stat="density", bins=30, element="step", fill=False,
                     color="#4C72B0", label="Raw", ax=axes[0])
        sns.histplot(after, stat="density", bins=30, element="step", fill=False,
                     color="#DD8452", label="Transformed", ax=axes[0])
        axes[0].set_title(f"Distribution before/after — {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Density")
        axes[0].legend()
        box_df = pd.concat([
            pd.DataFrame({"stage": "Raw", "value": before}),
            pd.DataFrame({"stage": "Transformed", "value": after}),
        ], ignore_index=True)
        sns.boxplot(data=box_df, x="stage", y="value", hue="stage", legend=False,
                    palette="colorblind", ax=axes[1])
        axes[1].set_title(f"Range and outliers — {col}")
        axes[1].set_xlabel("Processing stage")
        axes[1].set_ylabel(col)
        plt.tight_layout()
        _savefig(fig, os.path.join(out_dir, f"before_after_{col}.png"))

    common = [c for c in df_before.columns if c in df_after.columns]
    if common:
        missing = pd.DataFrame({
            "Raw": df_before[common].isna().sum(),
            "Transformed": df_after[common].isna().sum(),
        })
        missing = missing[missing.max(axis=1) > 0].sort_values("Raw", ascending=False).head(20)
        if not missing.empty:
            fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.35)))
            missing.plot(kind="barh", ax=ax, color=["#4C72B0", "#DD8452"])
            ax.invert_yaxis()
            ax.set_title("Missing values before and after preprocessing")
            ax.set_xlabel("Number of missing observations")
            ax.set_ylabel("Variable")
            _savefig(fig, os.path.join(out_dir, "before_after_missing_values.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pipeline tiền xử lý
# ─────────────────────────────────────────────────────────────────────────────

def plot_group_c_multimedia_overview(
    df: pd.DataFrame,
    subtype: Optional[str],
    out_dir: str,
    max_samples: int = 3000,
    viz_cfg: Optional[dict] = None,
) -> None:
    """Group C visuals: overall label balance and per-class numeric feature profiles."""
    viz_cfg = viz_cfg or {}
    max_classes = int(viz_cfg.get("multimedia_max_classes", 12))
    label_col = "label" if "label" in df.columns else None
    categorical_cols = [
        c for c in ["label", "dataset", "sublabel"]
        if c in df.columns and df[c].notna().any() and df[c].nunique(dropna=True) <= max_classes
    ]
    if categorical_cols:
        plot_class_distribution(df, categorical_cols, out_dir)

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns.tolist()
        if df[c].nunique(dropna=True) > 1
    ]
    if not numeric_cols:
        return

    valid_label = label_col and df[label_col].nunique(dropna=True) <= max_classes
    if valid_label:
        counts = df[label_col].value_counts(dropna=False).rename_axis(label_col).reset_index(name="count")
        counts["relative_pct"] = (counts["count"] / counts["count"].sum() * 100).round(4)
        counts.to_csv(os.path.join(out_dir, "group_c_class_counts.csv"), index=False)

        profile_cols = numeric_cols[:18]
        profile = df.groupby(label_col, dropna=False)[profile_cols].mean()
        profile.to_csv(os.path.join(out_dir, "group_c_feature_profile_by_class.csv"))
        scaled_profile = (profile - profile.mean()) / profile.std(ddof=0).replace(0, np.nan)
        scaled_profile = scaled_profile.fillna(0.0)
        fig, ax = plt.subplots(figsize=(max(9, len(profile_cols) * 0.55), max(4.5, len(profile) * 0.42)))
        sns.heatmap(scaled_profile, ax=ax, cmap="coolwarm", center=0, linewidths=0.3,
                    cbar_kws={"label": "Standardized class mean (z-score)"})
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
            fig, ax = plt.subplots(figsize=(min(12, max(8, plot_df[label_col].nunique() * 0.85)), 4.5))
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
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X)
            explained = pca.explained_variance_ratio_ * 100
            fig, ax = plt.subplots(figsize=(8, 6))
            if labels is not None:
                for label in sorted(labels.unique()):
                    mask = labels == label
                    ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.6, label=label)
                ax.legend(fontsize=8, loc="best", ncol=max(1, len(labels.unique()) // 8))
            else:
                ax.scatter(coords[:, 0], coords[:, 1], s=18, alpha=0.6, color="#4C72B0")
            ax.set_title("Group C Numeric Feature PCA")
            ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
            ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
            _savefig(fig, os.path.join(out_dir, "group_c_pca_by_class.png"))
        except Exception as exc:
            logger.warning(f"[EDA] Skip Group C PCA plot: {exc}")

    if subtype == "image" and valid_label and "path" in df.columns:
        try:
            from PIL import Image

            classes = df[label_col].dropna().astype(str).value_counts().index.tolist()[:max_classes]
            samples_per_class = int(viz_cfg.get("multimedia_samples_per_class", 2))
            samples = []
            for cls in classes:
                cls_rows = df[df[label_col].astype(str) == cls].head(samples_per_class)
                for _, row in cls_rows.iterrows():
                    samples.append((cls, row["path"]))
            if samples:
                ncols = min(6, max(2, samples_per_class * 3))
                nrows = int(np.ceil(len(samples) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
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

    if subtype == "image":
        rgb_cols = {
            "Red": [f"red_hist_{i}" for i in range(8)],
            "Green": [f"green_hist_{i}" for i in range(8)],
            "Blue": [f"blue_hist_{i}" for i in range(8)],
        }
        if all(all(c in df.columns for c in cols) for cols in rgb_cols.values()):
            fig, ax = plt.subplots(figsize=(9, 4.8))
            colors = {"Red": "#C44E52", "Green": "#55A868", "Blue": "#4C72B0"}
            bins = np.arange(8)
            for channel, cols in rgb_cols.items():
                means = df[cols].mean().values
                ax.plot(bins, means, marker="o", linewidth=2, color=colors[channel], label=channel)
            ax.set_title("Mean normalized RGB intensity profile")
            ax.set_xlabel("Intensity bin (dark to bright)")
            ax.set_ylabel("Mean proportion of pixels")
            ax.set_xticks(bins)
            ax.legend()
            ax.grid(alpha=0.25)
            _savefig(fig, os.path.join(out_dir, "group_c_rgb_profile.png"))

        if {"width", "height"}.issubset(df.columns):
            dimensions = df[["width", "height"]].dropna().copy()
            dimensions["aspect_ratio"] = dimensions["width"] / dimensions["height"].replace(0, np.nan)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            axes[0].scatter(dimensions["width"], dimensions["height"], s=12, alpha=0.35,
                            color="#4C72B0")
            axes[0].set_title("Image dimensions")
            axes[0].set_xlabel("Width (pixels)")
            axes[0].set_ylabel("Height (pixels)")
            sns.histplot(dimensions["aspect_ratio"].dropna(), bins=30, kde=True,
                         color="#55A868", ax=axes[1])
            axes[1].set_title("Aspect-ratio distribution")
            axes[1].set_xlabel("Width / height")
            plt.tight_layout()
            _savefig(fig, os.path.join(out_dir, "group_c_image_dimensions.png"))

    if subtype == "audio" and "path" in df.columns and valid_label:
        try:
            from scipy.io import wavfile
            from scipy.signal import spectrogram

            preferred = viz_cfg.get("audio_example_classes", [])
            available = df[label_col].dropna().astype(str).unique().tolist()
            classes = [c for c in preferred if c in available]
            classes += [c for c in available if c not in classes]
            classes = classes[:4]
            if classes:
                fig, axes = plt.subplots(len(classes), 2, figsize=(12, 2.8 * len(classes)))
                axes = np.atleast_2d(axes)
                for row_idx, cls in enumerate(classes):
                    path_value = df[df[label_col].astype(str) == cls]["path"].iloc[0]
                    sr, y = wavfile.read(str(path_value))
                    y = np.asarray(y)
                    if y.ndim > 1:
                        y = y.mean(axis=1)
                    if np.issubdtype(y.dtype, np.integer):
                        scale = max(abs(np.iinfo(y.dtype).min), np.iinfo(y.dtype).max)
                        y = y.astype("float32") / max(1, scale)
                    else:
                        y = y.astype("float32")
                    y = y[: int(sr * 8)]
                    times = np.arange(len(y)) / sr
                    axes[row_idx, 0].plot(times, y, linewidth=0.7, color="#4C72B0")
                    axes[row_idx, 0].set_title(f"{cls}: waveform")
                    axes[row_idx, 0].set_xlabel("Time (s)")
                    axes[row_idx, 0].set_ylabel("Amplitude")
                    frequencies, spec_times, power = spectrogram(
                        y, fs=sr, nperseg=min(1024, max(64, len(y))), scaling="spectrum"
                    )
                    power_db = 10 * np.log10(np.maximum(power, np.finfo(float).eps))
                    axes[row_idx, 1].pcolormesh(
                        spec_times, frequencies, power_db, shading="auto", cmap="magma"
                    )
                    axes[row_idx, 1].set_title(f"{cls}: STFT spectrogram")
                    axes[row_idx, 1].set_xlabel("Time (s)")
                    axes[row_idx, 1].set_ylabel("Frequency (Hz)")
                plt.tight_layout()
                _savefig(fig, os.path.join(out_dir, "group_c_audio_waveform_spectrogram.png"))
        except Exception as exc:
            logger.warning(f"[EDA] Skip Group C audio examples: {exc}")


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
        col_path = os.path.join(out_dir, f"frequency_{col}.csv")
        # Some scientific datasets use '/' in a column label. On Windows that
        # becomes a nested path; create its parent to retain the legacy artifact
        # naming convention without aborting the complete pipeline.
        Path(col_path).parent.mkdir(parents=True, exist_ok=True)
        col_table.to_csv(col_path, index=False)

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


def _fill_nulls_group_b(df: pd.DataFrame, group_col: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    if numeric_cols and group_col and group_col in df.columns:
        fill_cols = [c for c in numeric_cols if c != group_col]
        df[fill_cols] = df.groupby(group_col, sort=False)[fill_cols].transform(
            lambda values: values.interpolate(method="linear").ffill().bfill()
        )
    elif numeric_cols:
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
    global _FIGURE_DPI
    _FIGURE_DPI = int((config.get("visualization", {}) or {}).get("dpi", 300))
    dataset_root = dataset_output_root(config, dataset_name)
    raw_dir = str(dataset_root / "eda" / "raw")
    trans_dir = str(dataset_root / "eda" / "transformed")
    interim_dir = os.path.join(config.get("base_data_dir", "data"), "interim")
    for d in [raw_dir, trans_dir, interim_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    eda_cfg = config.get("eda", {})
    ds_cfg = (config.get("datasets") or {}).get(dataset_name, {})
    viz_cfg = {
        **(config.get("visualization", {}) or {}),
        **(ds_cfg.get("visualization", {}) or {}),
    }
    retain_legacy_plots = bool(viz_cfg.get("retain_legacy_plot_catalog", True))
    sns.set_theme(style=viz_cfg.get("style", "whitegrid"), palette=viz_cfg.get("colorblind_palette", "colorblind"))
    iqr_mult = eda_cfg.get("iqr_multiplier", 1.5)

    df = _coerce_configured_numeric(df, viz_cfg.get("continuous_columns", []))

    # ── Drop columns không cần thiết (ví dụ: id) ────────────────────────────
    drop_cols = ds_cfg.get("drop_columns", [])
    if drop_cols:
        existing_drop = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=existing_drop)
        logger.info(f"[EDA] Đã drop {len(existing_drop)} cột: {existing_drop}")
    var_types = classify_variables(df, viz_cfg)
    quant_cols = var_types["quantitative_continuous"] + var_types["quantitative_discrete"]
    legacy_quant_cols = df.select_dtypes(include="number").columns.tolist()
    continuous_plot_cols = legacy_quant_cols if retain_legacy_plots else (
        var_types["quantitative_continuous"] + [
            c for c in var_types["quantitative_discrete"]
            if df[c].nunique(dropna=True) > 20
        ]
    )

    # ── Pipeline 1: Raw ──────────────────────────────────────────────────────
    logger.info(f"[EDA] Pipeline 1 (raw) — {dataset_name}")

    null_report = df.isnull().sum().rename("null_count").to_frame()
    null_report["null_pct"] = (null_report["null_count"] / len(df) * 100).round(2)
    null_report.to_csv(os.path.join(raw_dir, "null_report.csv"))
    logger.info(f"[EDA] Null report → {raw_dir}/null_report.csv")

    _export_dataframe(df, raw_dir, "data_raw", group, eda_cfg)

    plot_histograms(df, continuous_plot_cols, raw_dir)
    plot_boxplots(df, continuous_plot_cols, raw_dir)
    if group in {"A", "B"}:
        plot_scatter_matrix(
            df,
            var_types["quantitative_continuous"] + var_types["quantitative_discrete"],
            raw_dir,
            max_samples=eda_cfg.get("plot_max_samples", 5000),
        )
    plot_correlation_heatmap(
        df, quant_cols, raw_dir,
        max_features=int(viz_cfg.get("max_heatmap_features", 18)),
    )
    plot_null_heatmap(df, raw_dir)
    plot_missingness_summary(
        df, raw_dir, max_features=int(viz_cfg.get("max_heatmap_features", 18)),
    )
    plot_categorical_distributions(
        df,
        var_types["qualitative"] + [
            c for c in var_types["quantitative_discrete"]
            if df[c].nunique(dropna=True) <= 20
        ],
        raw_dir,
        max_categories=int(viz_cfg.get("max_categories", 20)),
        stage="Raw",
    )
    plot_legacy_categorical_compatibility(
        df,
        viz_cfg.get("legacy_categorical_plot_aliases", []),
        raw_dir,
    )
    _protect = ds_cfg.get("protect_columns", [])
    if _protect:
        plot_class_distribution(df, _protect, raw_dir)
    if group == "B":
        plot_timeseries(
            df,
            raw_dir,
            columns=None if retain_legacy_plots else viz_cfg.get("timeseries_columns"),
            max_points=int(viz_cfg.get("max_line_points", 12000)),
            group_col=viz_cfg.get("series_group_col"),
            stage="Raw",
        )
        diagnostic_target = viz_cfg.get("diagnostic_target")
        if diagnostic_target:
            plot_temporal_diagnostics(
                df,
                diagnostic_target,
                raw_dir,
                seasonal_period=viz_cfg.get("seasonal_period"),
                max_points=int(viz_cfg.get("max_line_points", 12000)),
            )
    if group == "A":
        plot_target_relationships(
            df,
            viz_cfg.get("target_columns", _protect),
            viz_cfg.get("relationship_numeric", var_types["quantitative_continuous"]),
            viz_cfg.get("relationship_categorical", var_types["qualitative"]),
            raw_dir,
            max_categories=int(viz_cfg.get("max_categories", 20)),
            max_samples=int(viz_cfg.get("max_scatter_points", 5000)),
        )

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
            viz_cfg=viz_cfg,
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
        df_t = _fill_nulls_group_b(df, group_col=viz_cfg.get("series_group_col"))
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
    df_t.to_parquet(transformed_interim_path(config, dataset_name), index=(group == "B"))
    logger.info(f"[EDA] Đã lưu transformed data → {trans_dir}/")

    # Biểu đồ sau biến đổi
    var_types_t = classify_variables(df_t, viz_cfg)
    quant_cols_t = var_types_t["quantitative_continuous"] + var_types_t["quantitative_discrete"]
    legacy_quant_cols_t = df_t.select_dtypes(include="number").columns.tolist()
    stats_t = compute_statistics(df_t, var_types_t)
    pd.DataFrame(stats_t).T.to_csv(os.path.join(trans_dir, "statistics.csv"))
    export_frequency_tables(df_t, var_types_t["qualitative"], trans_dir)
    plot_legacy_categorical_compatibility(
        df_t,
        viz_cfg.get("legacy_categorical_plot_aliases", []),
        trans_dir,
    )
    if group == "C":
        plot_group_c_multimedia_overview(
            df_t,
            ds_cfg.get("subtype"),
            trans_dir,
            max_samples=eda_cfg.get("plot_max_samples", 3000),
            viz_cfg=viz_cfg,
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
    plot_before_after_comparison(
        df,
        df_t,
        viz_cfg.get("continuous_columns", var_types["quantitative_continuous"]),
        trans_dir,
        max_samples=int(viz_cfg.get("max_scatter_points", 5000)),
    )
    continuous_plot_cols_t = legacy_quant_cols_t if retain_legacy_plots else (
        var_types_t["quantitative_continuous"] + [
            c for c in var_types_t["quantitative_discrete"]
            if df_t[c].nunique(dropna=True) > 20
        ]
    )
    plot_histograms(df_t, continuous_plot_cols_t, trans_dir)
    plot_boxplots(df_t, continuous_plot_cols_t, trans_dir)
    if group == "A":
        plot_scatter_matrix(
            df_t,
            var_types_t["quantitative_continuous"] + var_types_t["quantitative_discrete"],
            trans_dir,
        )
        if retain_legacy_plots:
            plot_bar_charts(df_t, var_types_t["qualitative"], trans_dir)
        plot_categorical_distributions(
            df_t,
            var_types_t["qualitative"],
            trans_dir,
            max_categories=int(viz_cfg.get("max_categories", 20)),
            stage="Transformed",
        )
        if _protect:
            plot_class_distribution(df_t, _protect, trans_dir)
    elif group == "B":
        plot_scatter_matrix(
            df_t,
            var_types_t["quantitative_continuous"] + var_types_t["quantitative_discrete"],
            trans_dir,
            max_samples=eda_cfg.get("plot_max_samples", 5000),
        )
        plot_timeseries(
            df_t,
            trans_dir,
            columns=None if retain_legacy_plots else viz_cfg.get("timeseries_columns"),
            max_points=int(viz_cfg.get("max_line_points", 12000)),
            group_col=viz_cfg.get("series_group_col"),
            stage="Transformed",
        )
    plot_correlation_heatmap(
        df_t, quant_cols_t, trans_dir,
        max_features=int(viz_cfg.get("max_heatmap_features", 18)),
    )
    plot_null_heatmap(df_t, trans_dir)
    plot_missingness_summary(
        df_t, trans_dir, max_features=int(viz_cfg.get("max_heatmap_features", 18)),
    )

    logger.info(f"[EDA] Hoàn thành — {dataset_name}")
    return df_t
