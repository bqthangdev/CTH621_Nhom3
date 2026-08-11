# CTH621 Pipeline

Pipeline phân tích dữ liệu và machine learning cho đồ án CTH621. Dự án hỗ trợ 3 nhóm dữ liệu:

- Nhóm A: dữ liệu dạng bảng, chạy `EDA -> Classification -> Clustering`
- Nhóm B: dữ liệu chuỗi thời gian, chạy `EDA -> Regression -> Clustering`
- Nhóm C: dữ liệu đa phương tiện, chạy `EDA -> Clustering`

Toàn bộ pipeline được điều khiển qua `src/presentation/run_pipeline.py` và cấu hình tập trung trong `configs/params.yaml`.

## Yêu cầu

- Python 3.11+
- `pip` hoặc `conda`
- Khuyến nghị dùng virtual environment riêng cho dự án

## Cài đặt

### Cách 1: `venv`

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux hoặc macOS:

```bash
source .venv/bin/activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

### Cách 2: `conda`

```bash
conda env create -f environment.yml
conda activate cth621
```

## Cấu trúc chính

```text
CTH621_Nhom3/
├── configs/
│   └── params.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── progress.json
├── docs/
├── logs/
│   └── pipeline.log
├── outputs/
├── src/
│   ├── data/
│   ├── domain/
│   ├── infrastructure/
│   └── presentation/
│       └── run_pipeline.py
├── requirements.txt
├── environment.yml
└── summary_results.csv
```

## Dataset hiện có

| Dataset | Nhóm | Tác vụ chính |
|---|---|---|
| `healthcare_dataset_stroke` | A | Classification, Clustering |
| `neurofibromatosis` | A | Classification, Clustering |
| `higher_education_predictors_of_student_retention` | A | Classification, Clustering |
| `pgcb_hourly_generation` | B | Regression, Clustering |
| `har70plus` | B | Regression, Clustering |
| `dow_jones_index` | B | Regression, Clustering |
| `realwaste` | C | Clustering |
| `heartbeat_sounds` | C | Clustering |

## Cấu hình

Mọi cấu hình nằm trong `configs/params.yaml`, bao gồm:

- đường dẫn dataset
- loại dataset A, B, C
- target columns hoặc `target_col`
- hyperparameter cho classification, regression, clustering
- tùy chọn visualization, logging, tracking

Nguyên tắc của repo là không hardcode tham số trong Python nếu có thể đưa vào YAML.

## Cú pháp chạy

```bash
python src/presentation/run_pipeline.py --task <TASK> --dataset <DATASET> [--algo <ALGO>] [--config <PATH>] [--reset]
```

| Tham số | Ý nghĩa |
|---|---|
| `--task` | `eda`, `classification`, `regression`, `clustering`, `all` |
| `--dataset` | tên dataset trong `configs/params.yaml` hoặc `all` |
| `--algo` | chỉ chạy 1 thuật toán cụ thể nếu task hỗ trợ |
| `--config` | file config, mặc định là `configs/params.yaml` |
| `--reset` | xóa checkpoint của dataset trước khi chạy lại |

## Ví dụ chạy

### 1. Chạy EDA cho dataset bảng

```bash
python src/presentation/run_pipeline.py --task eda --dataset healthcare_dataset_stroke --config configs/params.yaml
```

### 2. Chạy classification cho dataset bảng

```bash
python src/presentation/run_pipeline.py --task classification --dataset healthcare_dataset_stroke --config configs/params.yaml
```

Chỉ chạy một thuật toán:

```bash
python src/presentation/run_pipeline.py --task classification --dataset healthcare_dataset_stroke --algo logistic --config configs/params.yaml
```

### 3. Chạy regression cho dataset chuỗi thời gian

```bash
python src/presentation/run_pipeline.py --task regression --dataset pgcb_hourly_generation --config configs/params.yaml
```

Ví dụ chạy riêng `xgboost`:

```bash
python src/presentation/run_pipeline.py --task regression --dataset pgcb_hourly_generation --algo xgboost --config configs/params.yaml
```

### 4. Chạy clustering

```bash
python src/presentation/run_pipeline.py --task clustering --dataset higher_education_predictors_of_student_retention --config configs/params.yaml
```

### 5. Chạy toàn bộ pipeline cho một dataset

```bash
python src/presentation/run_pipeline.py --task all --dataset healthcare_dataset_stroke --config configs/params.yaml
```

### 6. Chạy toàn bộ pipeline cho tất cả dataset

```bash
python src/presentation/run_pipeline.py --task all --dataset all --config configs/params.yaml
```

### 7. Chạy lại từ đầu với checkpoint mới

```bash
python src/presentation/run_pipeline.py --task all --dataset healthcare_dataset_stroke --config configs/params.yaml --reset
```

## Kết quả đầu ra

Mỗi dataset có thư mục riêng trong `outputs/<dataset_name>/` với ba nhóm artifact chính:

- `eda/`
- `ml/`
- `models/`

Ví dụ:

```text
outputs/healthcare_dataset_stroke/
├── eda/
│   ├── raw/
│   └── transformed/
├── ml/
│   ├── classification/
│   └── clustering/
└── models/
```

Một số file thường gặp:

- `statistics.csv`
- `null_report.csv`
- `data_transformed.csv`
- `correlation_heatmap.png`
- `scatter_matrix.png`
- `cm_<algo>_<target>.png`
- `model_comparison_<target>.png`
- `kmeans_elbow_silhouette.png`
- `dendrogram.png`
- `cluster_profile_*.csv`
- `clf_<algo>_<target>.joblib`
- `reg_<algo>_<target>.joblib`
- `cluster_kmeans.joblib`

Ngoài ra:

- `summary_results.csv` lưu metric tổng hợp của các lần chạy
- `logs/pipeline.log` lưu log chi tiết
- `data/interim/` lưu dữ liệu đã biến đổi để tái sử dụng cho các bước ML

## Checkpoint

Checkpoint được lưu tại `data/progress.json`.

Pipeline dùng file này để:

- bỏ qua các bước đã hoàn thành
- tiếp tục chạy lại sau khi bị gián đoạn
- hỗ trợ `--reset` để chạy lại có kiểm soát

Xem checkpoint:

Windows PowerShell:

```powershell
Get-Content data\progress.json
```

Linux hoặc macOS:

```bash
cat data/progress.json
```

Lưu ý:

- `--reset` xóa trạng thái checkpoint của dataset hoặc task tương ứng
- `--reset` không chủ động xóa toàn bộ output cũ
- nếu cần chạy sạch hoàn toàn, hãy tự xóa artifact cũ trong `outputs/` và cân nhắc làm mới `summary_results.csv`

## Theo dõi kết quả

Có thể xem nhanh `summary_results.csv` bằng Python:

```python
import pandas as pd

df = pd.read_csv("summary_results.csv")
print(df.head())
```

## Troubleshooting

| Lỗi | Cách xử lý |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | chạy lệnh từ thư mục gốc của repo |
| `FileNotFoundError` cho dataset | kiểm tra lại đường dẫn trong `configs/params.yaml` |
| dataset không tìm thấy | kiểm tra tên dataset trong phần `datasets:` của YAML |
| kết quả khác nhau giữa các máy | cài lại đúng phiên bản từ `requirements.txt` |
| task bị bỏ qua ngoài ý muốn | kiểm tra `data/progress.json` hoặc thêm `--reset` |

## Làm việc nhóm

Khuyến nghị commit:

- `src/`
- `configs/`
- `requirements.txt`
- `environment.yml`
- `docs/`

Chỉ commit `summary_results.csv` hoặc `data/progress.json` khi cả nhóm thống nhất dùng chung chúng như artifact theo dõi.

## Quy ước commit

- `feat(eda): ...`
- `feat(ml): ...`
- `fix(loader): ...`
- `config: ...`
- `docs: ...`
- `chore: ...`

## Tracking

Repo hỗ trợ cấu hình tracking trong `configs/params.yaml`:

- `none`
- `wandb`
- `mlflow`

Ví dụ bật Weights & Biases:

```bash
wandb login
```

Sau đó cập nhật trong YAML:

```yaml
tracking:
  provider: "wandb"
  wandb_project: "CTH621"
```

## Ghi chú

- `README.md` là tài liệu hướng dẫn chạy chính thức của repo.
- `RUN.md` đã được loại bỏ để tránh trùng lặp và lệch tài liệu theo thời gian.
