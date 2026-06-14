# Hướng dẫn chạy Pipeline CTH621

## Mục lục

1. [Yêu cầu môi trường](#1-yêu-cầu-môi-trường)
2. [Cú pháp lệnh và tham số](#2-cú-pháp-lệnh-và-tham-số)
3. [Kết quả sau khi chạy](#3-kết-quả-sau-khi-chạy)
4. [Tổ chức lưu trữ kết quả](#4-tổ-chức-lưu-trữ-kết-quả)
5. [Cơ chế Checkpoint](#5-cơ-chế-checkpoint)
6. [Ví dụ thực tế](#6-ví-dụ-thực-tế)
7. [Xử lý lỗi thường gặp](#7-xử-lý-lỗi-thường-gặp)

---

## 1. Yêu cầu môi trường

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Hoặc tạo môi trường conda
conda env create -f environment.yml
conda activate cth621
```

Khi khởi động, pipeline **tự động kiểm tra** phiên bản thư viện so với `requirements.txt` và in cảnh báo nếu có lệch phiên bản (không dừng pipeline).

---

## 2. Cú pháp lệnh và tham số

### Cú pháp cơ bản

```bash
python src/presentation/run_pipeline.py \
    --task <TASK> \
    --dataset <DATASET> \
    [--algo <ALGO>] \
    [--config <CONFIG>] \
    [--reset]
```

### Mô tả tham số

| Tham số | Bắt buộc | Mặc định | Mô tả |
|---|---|---|---|
| `--task` | ✅ | — | Task cần chạy. Xem bảng task bên dưới. |
| `--dataset` | ✅ | — | Tên dataset trong `params.yaml`. Dùng `all` để chạy mọi dataset. |
| `--algo` | ❌ | tất cả | Chỉ chạy một thuật toán cụ thể. |
| `--config` | ❌ | `configs/params.yaml` | Đường dẫn tới file cấu hình YAML. |
| `--reset` | ❌ | `false` | Xóa checkpoint của dataset, buộc chạy lại từ đầu. |

### Các giá trị `--task`

| Giá trị | Mô tả |
|---|---|
| `eda` | Phân tích khám phá dữ liệu — thống kê, visualize, preprocessing |
| `classification` | Huấn luyện & đánh giá các mô hình phân loại (Group A) |
| `regression` | Huấn luyện & đánh giá các mô hình hồi quy chuỗi thời gian (Group B) |
| `clustering` | Chạy K-Means, Hierarchical, DBSCAN, Gaussian Mixture (mọi group) |
| `all` | Chạy tuần tự: EDA → Classification/Regression → Clustering |

### Các giá trị `--dataset`

Dataset phải được khai báo trong phần `datasets:` của `configs/params.yaml`.

| Tên dataset | Nhóm | Loại bài toán |
|---|---|---|
| `healthcare_dataset_stroke` | A | Classification + Clustering |
| `neurofibromatosis` | A | Classification + Clustering |
| `stock_prices` | B | Regression + Clustering |
| `image_dataset` | C | Clustering |

> Dùng `--dataset all` để chạy lần lượt toàn bộ dataset có trong config.

### Các giá trị `--algo`

**Classification:**
`logistic`, `decision_tree`, `svm`, `random_forest`, `gradient_boosting`, `knn`, `naive_bayes`

**Regression:**
`linear_regression`, `arima`, `xgboost`, `ridge`, `lasso`, `random_forest_regressor`, `svr`

---

## 3. Kết quả sau khi chạy

### 3.1 Output ra console / log

Mỗi bước in log theo định dạng:

```
2026-06-13 11:36:00 | INFO | src.domain.classification | [CLASSIFY] gradient_boosting | target=stroke | Acc=0.9511 | Prec=0.9298 | Rec=0.9511 | F1=0.9291
2026-06-13 11:36:05 | INFO | src.domain.classification | [CLASSIFY] 5-Fold StratifiedKFold | Acc=0.9495±0.0013 | F1=0.9266±0.0006
```

Log cũng được ghi vào thư mục `logs/`.

### 3.2 Metrics Classification (Hold-out + 5-Fold CV)

| Metric | Ý nghĩa |
|---|---|
| `accuracy` | Độ chính xác trên tập test (Hold-out) |
| `precision` | Precision (weighted avg) |
| `recall` | Recall (weighted avg) |
| `f1` | F1-score (weighted avg) |
| `cv_accuracy_mean` | Accuracy trung bình qua 5-Fold StratifiedKFold |
| `cv_accuracy_std` | Độ lệch chuẩn accuracy qua 5-Fold |
| `cv_f1_mean` | F1 trung bình qua 5-Fold |
| `cv_f1_std` | Độ lệch chuẩn F1 qua 5-Fold |

> **Lưu ý đánh giá:** Ưu tiên `cv_accuracy_mean` và `cv_f1_mean` để đánh giá khả năng tổng quát hóa. `cv_*_std` nhỏ nghĩa là mô hình ổn định. Nếu Hold-out cao nhưng CV thấp hơn nhiều → **overfit**.

### 3.3 Metrics Regression (Chronological Split + TimeSeriesSplit CV)

| Metric | Ý nghĩa |
|---|---|
| `mae` | Mean Absolute Error (Hold-out) |
| `rmse` | Root Mean Squared Error (Hold-out) |
| `r2` | Hệ số xác định R² (Hold-out) |
| `cv_mae_mean/std` | MAE trung bình ± std qua TimeSeriesSplit |
| `cv_rmse_mean/std` | RMSE trung bình ± std qua TimeSeriesSplit |
| `cv_r2_mean/std` | R² trung bình ± std qua TimeSeriesSplit |

### 3.4 Metrics Clustering

| Metric | Ý nghĩa |
|---|---|
| `n_clusters` | Số cluster tìm được (best_k với KMeans) |
| `silhouette_score` | Silhouette score toàn bộ dataset (−1 đến 1, cao hơn tốt hơn) |
| `cv_silhouette_mean/std` | Silhouette trung bình ± std qua Subsampling CV (5 lần × 80%) |
| `notes` | Thông tin thêm, ví dụ `outliers=351` với DBSCAN |

### 3.5 File tổng hợp `summary_results.csv`

Mọi kết quả đều được **append** (không overwrite) vào file `summary_results.csv` ở thư mục gốc.

```
timestamp, dataset, task, algorithm, target_col, split, n_train, n_test,
accuracy, precision, recall, f1,
cv_accuracy_mean, cv_accuracy_std, cv_f1_mean, cv_f1_std,
mae, rmse, r2, cv_mae_mean, cv_mae_std, cv_rmse_mean, cv_rmse_std, cv_r2_mean, cv_r2_std,
silhouette_score, cv_silhouette_mean, cv_silhouette_std,
n_clusters, notes
```

---

## 4. Tổ chức lưu trữ kết quả

```
outputs/
└── <dataset_name>/
    ├── eda/
    │   ├── raw/                          # Phân tích DỮ LIỆU GỐC (trước preprocessing)
    │   │   ├── boxplot_<col>.png         # Boxplot từng biến số
    │   │   ├── null_report.csv           # Thống kê giá trị null theo cột
    │   │   ├── statistics.csv            # describe(): mean/std/min/max/percentile
    │   │   ├── data_raw.csv              # Bản sao dữ liệu gốc (CSV)
    │   │   └── data_raw.xlsx             # Bản sao dữ liệu gốc (Excel)
    │   │
    │   └── transformed/                  # Phân tích DỮ LIỆU ĐÃ XỬ LÝ (sau preprocessing)
    │       ├── hist_<col>.png            # Histogram phân phối từng biến
    │       ├── boxplot_<col>.png         # Boxplot sau chuẩn hóa
    │       ├── scatter_matrix.png        # Ma trận scatter các biến số
    │       ├── data_transformed.csv      # Dữ liệu đã xử lý (CSV)
    │       └── data_transformed.xlsx     # Dữ liệu đã xử lý (Excel)
    │
    ├── ml/
    │   ├── classification/               # (placeholder — confusion matrix plots)
    │   └── clustering/
    │       ├── kmeans_elbow_silhouette.png   # Elbow curve + Silhouette score vs k
    │       ├── dendrogram.png               # Cây phân cấp Hierarchical Clustering
    │       ├── kmeans_labels.csv            # Nhãn cluster KMeans cho từng sample
    │       ├── hierarchical_labels.csv      # Nhãn cluster Hierarchical
    │       ├── dbscan_labels.csv            # Nhãn cluster DBSCAN (−1 = outlier)
    │       └── gmm_labels.csv              # Nhãn cluster Gaussian Mixture
    │
    └── models/
        ├── clf_<algo>_<target>.joblib    # Model classification đã train
        └── cluster_kmeans.joblib         # Model KMeans tốt nhất (best_k)

data/
└── interim/
    └── <dataset_name>_transformed.parquet  # Dữ liệu sau EDA — dùng lại cho ML

data/
└── progress.json                           # Trạng thái checkpoint từng bước
```

### Quy tắc đặt tên model

| Pattern | Ví dụ |
|---|---|
| `clf_<algo>_<target>.joblib` | `clf_gradient_boosting_stroke.joblib` |
| `reg_<algo>.joblib` | `reg_xgboost.joblib` |
| `cluster_<algo>.joblib` | `cluster_kmeans.joblib` |

---

## 5. Cơ chế Checkpoint

Pipeline dùng `data/progress.json` để theo dõi tiến độ, tránh chạy lại các bước đã hoàn thành.

### Ví dụ `progress.json`

```json
{
  "healthcare_dataset_stroke": {
    "eda": "DONE",
    "eda_completed_at": "2026-06-13T11:35:52.112715",
    "classification": "DONE",
    "classification_completed_at": "2026-06-13T11:49:16.547554",
    "clustering": "DONE",
    "clustering_completed_at": "2026-06-13T11:54:11.915097"
  }
}
```

### Hành vi khi chạy lại

| Trường hợp | Hành vi |
|---|---|
| Task đã `DONE`, **không có** `--reset` | `[SKIP]` — bỏ qua, không chạy lại |
| Task đã `DONE`, **có** `--reset` | Xóa checkpoint → chạy lại toàn bộ |
| Task đang `FAILED` | Chạy lại tự động (không cần `--reset`) |
| Model `.joblib` đã tồn tại | Load lại từ file, **không retrain** |

### Lưu ý khi dùng `--reset`

`--reset` chỉ xóa trạng thái checkpoint, **không xóa** model `.joblib` và `summary_results.csv`. Nếu muốn chạy lại hoàn toàn sạch:

```bash
# Xóa CSV cũ trước (tránh duplicate rows)
del summary_results.csv

# Xóa model cũ nếu muốn retrain (tùy chọn)
del outputs\<dataset>\models\clf_*.joblib

# Chạy lại
python src/presentation/run_pipeline.py --task all --dataset <dataset> --config configs/params.yaml --reset
```

---

## 6. Ví dụ thực tế

### Chạy toàn bộ pipeline cho một dataset

```bash
python src/presentation/run_pipeline.py \
    --task all \
    --dataset healthcare_dataset_stroke \
    --config configs/params.yaml
```

### Chỉ chạy EDA

```bash
python src/presentation/run_pipeline.py \
    --task eda \
    --dataset healthcare_dataset_stroke \
    --config configs/params.yaml
```

### Chỉ chạy Classification với một thuật toán cụ thể

```bash
python src/presentation/run_pipeline.py \
    --task classification \
    --dataset healthcare_dataset_stroke \
    --algo gradient_boosting \
    --config configs/params.yaml
```

### Chạy lại từ đầu (reset checkpoint)

```bash
python src/presentation/run_pipeline.py \
    --task all \
    --dataset healthcare_dataset_stroke \
    --config configs/params.yaml \
    --reset
```

### Chạy tất cả dataset cùng lúc

```bash
python src/presentation/run_pipeline.py \
    --task all \
    --dataset all \
    --config configs/params.yaml
```

### Xem kết quả tổng hợp (Python)

```python
import pandas as pd

df = pd.read_csv("summary_results.csv")

# Kết quả classification
clf = df[df["task"] == "classification"][
    ["dataset", "algorithm", "target_col", "accuracy", "f1", "cv_accuracy_mean", "cv_f1_mean"]
]
print(clf.to_string(index=False))

# Kết quả clustering
clu = df[df["task"] == "clustering"][
    ["dataset", "algorithm", "n_clusters", "silhouette_score", "cv_silhouette_mean"]
]
print(clu.to_string(index=False))
```

---

## 7. Xử lý lỗi thường gặp

### Dataset không tìm thấy trong config

```
[ERROR] Dataset 'my_dataset' không tìm thấy trong params.yaml.
```

**Giải pháp:** Thêm dataset vào phần `datasets:` trong `configs/params.yaml`.

### `summary_results.csv` bị lệch schema (duplicate hoặc sai cột)

Xảy ra khi chạy pipeline với schema cũ rồi thêm cột mới.

```bash
del summary_results.csv
python src/presentation/run_pipeline.py --task all --dataset <dataset> --config configs/params.yaml --reset
```

### Lỗi `y contains 1 class` khi train

Target column bị capping về 0 do IQR = 0 (dữ liệu mất cân bằng cực nặng).

**Giải pháp:** Thêm target column vào `protect_columns` trong config dataset:

```yaml
healthcare_dataset_stroke:
  protect_columns: ["stroke", "hypertension", "heart_disease"]
```

### Model `.joblib` cũ cho kết quả sai (Confusion Matrix toàn 0)

Pipeline đang load model từ checkpoint cũ (train trước khi có `protect_columns`).

**Giải pháp:** Xóa model cũ và chạy lại:

```bash
del outputs\healthcare_dataset_stroke\models\clf_decision_tree_stroke.joblib
python src/presentation/run_pipeline.py --task classification --dataset healthcare_dataset_stroke --config configs/params.yaml
```

### `statsmodels` không tìm thấy (ARIMA)

```
[WARNING] statsmodels not installed — ARIMA bị bỏ qua.
```

**Giải pháp:** Cài đặt thủ công:

```bash
pip install statsmodels
```
