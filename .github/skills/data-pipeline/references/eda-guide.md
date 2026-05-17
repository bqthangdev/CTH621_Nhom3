# EDA Guide — CTH621

## 1. Variable Classification (Auto-detect)

```python
def classify_variables(df: pd.DataFrame) -> dict:
    """
    Returns dict with keys: 'qualitative', 'quantitative_continuous', 'quantitative_discrete'
    Rules:
    - object / category / bool → Qualitative
    - int with nunique < 20 → Quantitative Discrete
    - float or int with nunique >= 20 → Quantitative Continuous
    """
```

## 2. Descriptive Statistics

### Quantitative (Groups A & B)
Compute ALL of the following — missing any is a bug:
- Mean, Median, Mode
- Percentiles: P5, P25, P50, P75, P95
- Range (max - min)
- Variance, Standard Deviation
- CV = (std / mean) * 100
- IQR = P75 - P25

### Qualitative (Group A only)
- Absolute frequency table
- Relative frequency (%)
- Mode

## 3. Visualization Requirements by Group

### Group A (Tabular)
| Plot | Columns | Library |
|------|---------|---------|
| Histogram | All quantitative | matplotlib/seaborn |
| Boxplot | All quantitative | seaborn |
| Scatter Plot | All quantitative pairs | seaborn pairplot |
| Bar Chart | All qualitative | matplotlib |
| Pie Chart | Qualitative with ≤10 unique values | matplotlib |

### Group B (Time Series)
| Plot | Note |
|------|------|
| Line Chart (time on x-axis) | **MANDATORY** — shows trend & seasonality |
| Histogram | Distribution of values |

### Group C (Multimedia)
Extract numeric features first:
- **Images**: width, height, mean_brightness, std_brightness, aspect_ratio (per image)
- **Audio**: duration_sec, sample_rate, mean_amplitude, rms_energy, zero_crossing_rate (librosa)
- **Video**: duration_sec, fps, frame_count, width, height
- **Text**: char_count, word_count, sentence_count, avg_word_length, unique_word_ratio

Then apply Group A statistics & visualizations on the extracted feature DataFrame.

## 4. Preprocessing Pipelines

### Pipeline 1 — Raw Data Report
```
1. Count nulls per column → save to outputs/{name}/eda/raw/null_report.csv
2. Plot Boxplots → save to outputs/{name}/eda/raw/boxplot_{col}.png
3. Do NOT modify data
```

### Pipeline 2 — Transformation

#### Group A
```
1. Fill nulls:
   - Quantitative Continuous → Mean
   - Quantitative Discrete → Median
   - Qualitative → Mode
2. Outlier handling (IQR-capping):
   lower = Q1 - 1.5 * IQR
   upper = Q3 + 1.5 * IQR
   Clip values to [lower, upper]
3. Normalization (configurable via params.yaml):
   - min-max: (x - min) / (max - min)
   - z-score: (x - mean) / std
```

#### Group B
```
1. Fill nulls: Forward Fill → then Backward Fill for leading nulls
2. Linear Interpolation for remaining gaps
3. Optional transforms (from params.yaml):
   - Differencing: df.diff(periods=1)
   - Log-transform: np.log1p(series)
```

### Output Format
Save BOTH pipelines:
```python
df_raw.to_excel(f"outputs/{name}/eda/raw/data_raw.xlsx", index=False)
df_transformed.to_excel(f"outputs/{name}/eda/transformed/data_transformed.xlsx", index=False)
df_transformed.to_csv(f"outputs/{name}/eda/transformed/data_transformed.csv", index=False)
# Also save as parquet for fast ML loading:
df_transformed.to_parquet(f"data/interim/{name}_transformed.parquet", index=False)
```

### Bilingual Chart Requirement (R11 — BẮT BUỘC)

Mọi biểu đồ, sơ đồ, hình ảnh đều phải lưu **2 phiên bản**: tiếng Anh (`_en.png`) và tiếng Việt (`_vi.png`). Dùng helper sau trong `eda.py`:

```python
def _savefig_bilingual(
    fig,
    ax_or_axes,
    base_path: str,
    titles: dict,
    xlabels: dict = None,
    ylabels: dict = None,
    legend_maps: dict = None,
) -> None:
    """
    Lưu figure thành 2 phiên bản ngôn ngữ.

    Args:
        fig:          matplotlib Figure
        ax_or_axes:   Axes hoặc list[Axes] cần đổi text
        base_path:    Đường dẫn KHÔNG có hậu tố, ví dụ:
                      "outputs/ds/eda/transformed/hist_age"
                      → sinh ra hist_age_en.png và hist_age_vi.png
        titles:       {"en": "Age Distribution", "vi": "Phân Phối Tuổi"}
        xlabels:      {"en": "Age", "vi": "Tuổi"}  (tuỳ chọn)
        ylabels:      {"en": "Count", "vi": "Số lượng"}  (tuỳ chọn)
        legend_maps:  {"en": ["Male","Female"], "vi": ["Nam","Nữ"]}  (tuỳ chọn)
    """
    from pathlib import Path
    import copy

    axes = ax_or_axes if isinstance(ax_or_axes, (list, tuple)) else [ax_or_axes]
    Path(base_path).parent.mkdir(parents=True, exist_ok=True)

    for lang in ("en", "vi"):
        for ax in axes:
            if titles:
                ax.set_title(titles[lang])
            if xlabels:
                ax.set_xlabel(xlabels[lang])
            if ylabels:
                ax.set_ylabel(ylabels[lang])
            if legend_maps and ax.get_legend():
                for text, label in zip(ax.get_legend().get_texts(), legend_maps[lang]):
                    text.set_text(label)
        fig.savefig(f"{base_path}_{lang}.png", bbox_inches="tight", dpi=150)
        logger.info(f"[EDA] Đã lưu biểu đồ → {base_path}_{lang}.png")
    plt.close(fig)
```

**Ví dụ sử dụng:**
```python
# Histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["age"], bins=20, color="#4C72B0")
_savefig_bilingual(
    fig, ax,
    base_path=f"outputs/{name}/eda/transformed/hist_age",
    titles={"en": "Age Distribution", "vi": "Phân Phối Tuổi"},
    xlabels={"en": "Age", "vi": "Tuổi"},
    ylabels={"en": "Count", "vi": "Số lượng"},
)
# → sinh ra: hist_age_en.png, hist_age_vi.png

# Boxplot nhiều cột
fig, ax = plt.subplots(figsize=(10, 5))
df[["age", "score"]].boxplot(ax=ax)
_savefig_bilingual(
    fig, ax,
    base_path=f"outputs/{name}/eda/raw/boxplot_overview",
    titles={"en": "Outlier Detection — Boxplot", "vi": "Phát Hiện Ngoại Lệ — Boxplot"},
    ylabels={"en": "Value", "vi": "Giá trị"},
)
# → sinh ra: boxplot_overview_en.png, boxplot_overview_vi.png
```

**Quy tắc đặt tên file:**
```
outputs/{dataset}/eda/raw/
├── boxplot_{col}_en.png
└── boxplot_{col}_vi.png

outputs/{dataset}/eda/transformed/
├── hist_{col}_en.png
├── hist_{col}_vi.png
├── scatter_matrix_en.png
├── scatter_matrix_vi.png
├── bar_{col}_en.png
└── bar_{col}_vi.png

outputs/{dataset}/ml/clustering/
├── kmeans_elbow_silhouette_en.png
├── kmeans_elbow_silhouette_vi.png
├── dendrogram_en.png
└── dendrogram_vi.png
```

> **Không lưu file không có hậu tố ngôn ngữ** (ví dụ `hist_age.png` là SAI — phải là `hist_age_en.png` và `hist_age_vi.png`).
