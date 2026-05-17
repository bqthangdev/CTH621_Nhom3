# CTH621 — Hướng Dẫn Cài Đặt & Vận Hành Pipeline

> **Dự án:** Phân tích dữ liệu bậc Thạc sĩ — Nhóm 3  
> **Môn học:** CTH621 – Hệ thống thông tin  
> **Học kỳ:** 3 – Năm học 2025–2026  
> **Yêu cầu:** Python 3.11+

---

## Mục Lục

1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Cài đặt môi trường](#2-cài-đặt-môi-trường)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Cấu hình dataset](#4-cấu-hình-dataset)
5. [Chạy từng task](#5-chạy-từng-task)
6. [Chạy toàn bộ pipeline](#6-chạy-toàn-bộ-pipeline)
7. [Checkpoint & Resume](#7-checkpoint--resume)
8. [Xem kết quả](#8-xem-kết-quả)
9. [Làm việc nhóm với Git & GitHub](#9-làm-việc-nhóm-với-git--github)
10. [Đồng bộ nhóm (W&B / MLflow)](#10-đồng-bộ-nhóm-wb--mlflow)
11. [Xử lý lỗi thường gặp](#11-xử-lý-lỗi-thường-gặp)
12. [Sử dụng Skill GitHub Copilot](#12-sử-dụng-skill-github-copilot)

---

## 1. Yêu Cầu Hệ Thống

| Thành phần | Phiên bản tối thiểu |
|------------|-------------------|
| Python | 3.11 |
| pip | 23.0+ |
| RAM | 8 GB (khuyến nghị 16 GB cho dataset lớn) |
| Disk | 10 GB trống |
| OS | Windows 10+ / Ubuntu 20.04+ / macOS 12+ |

---

## 2. Cài Đặt Môi Trường

### Cách A — Dùng `pip` + `venv` (khuyến nghị Windows)

```bash
# 1. Clone repo
git clone <repo_url>
cd CTH621_Nhom3

# 2. Tạo virtual environment
python -m venv .venv

# 3. Kích hoạt (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Kích hoạt (Linux / macOS)
source .venv/bin/activate

# 4. Cài đặt dependencies
pip install -r requirements.txt
```

### Cách B — Dùng Conda

```bash
conda env create -f environment.yml
conda activate cth621
```

### Kiểm Tra Cài Đặt

```bash
python -c "import pandas, sklearn, xgboost, matplotlib; print('OK')"
```

---

## 3. Cấu Trúc Thư Mục

```
CTH621_Nhom3/
├── src/
│   ├── data/
│   │   ├── loader.py          # Nạp dữ liệu CSV, ảnh, audio, video, text
│   │   └── validator.py       # Kiểm tra ràng buộc nhóm A / B / C
│   ├── domain/
│   │   ├── eda.py             # EDA, thống kê, tiền xử lý, biểu đồ
│   │   ├── classification.py  # Phân loại (Logistic, Decision Tree, SVM)
│   │   ├── regression.py      # Hồi quy chuỗi thời gian (Linear, ARIMA, XGBoost)
│   │   └── clustering.py      # Phân cụm (K-Means, Hierarchical, DBSCAN)
│   ├── infrastructure/
│   │   ├── logger.py          # Logging ra file + console
│   │   └── checkpoint.py      # Theo dõi tiến độ qua progress.json
│   └── presentation/
│       └── run_pipeline.py    # CLI entry point
├── configs/
│   └── params.yaml            # Toàn bộ hyperparameter và cấu hình
├── data/
│   ├── raw/                   # Dataset gốc (không chỉnh sửa)
│   └── interim/               # Dữ liệu trung gian .parquet
├── outputs/                   # Kết quả EDA, ML, model checkpoint
├── logs/
│   └── pipeline.log           # Log toàn bộ quá trình chạy
├── progress.json              # Trạng thái checkpoint
├── summary_results.csv        # Tổng hợp metrics tất cả run
├── requirements.txt
├── environment.yml
└── ecosystem.config.js        # PM2 config cho Linux server
```

---

## 4. Cấu Hình Dataset

Mọi cấu hình nằm trong [`configs/params.yaml`](configs/params.yaml).

### 4.1 Thêm Dataset Mới

**Nhóm A (Tabular):**
```yaml
datasets:
  ten_dataset_cua_ban:
    type: "A"
    file: "data/raw/ten_dataset_cua_ban.csv"
    classification:
      target_columns: ["CotNhan1", "CotNhan2", "CotNhan3"]   # >= 3 cột
    clustering:
      label_columns: ["CotNhan1", "CotNhan2", "CotNhan3"]
```

**Nhóm B (Time Series):**
```yaml
datasets:
  chuan_thoi_gian:
    type: "B"
    file: "data/raw/chuan_thoi_gian.csv"
    datetime_col: "Date"        # Tên cột chứa thời gian
    target_col: "Value"         # Cột cần dự đoán
    clustering:
      label_columns: []
```

**Nhóm C (Multimedia — ảnh):**
```yaml
datasets:
  bo_anh:
    type: "C"
    subtype: "image"            # "image" | "audio" | "video" | "text"
    dir: "data/raw/images/"
    clustering:
      label_columns: ["label"]
```

---

### 4.2 Chọn Thuật Toán Per-Dataset

Mỗi dataset có thể khai báo **riêng** các thuật toán sẽ chạy và **tinh chỉnh siêu tham số** ngay trong `params.yaml`. Không cần sửa code Python.

**Logic ưu tiên (cao → thấp):**

| Mức | Cơ chế | Kết quả |
|-----|--------|---------|
| 1 | CLI `--algo logistic` | Chỉ chạy 1 algo đó, bỏ qua tất cả config |
| 2 | `datasets[name][task].algorithms:` | Chỉ chạy các algo được liệt kê |
| 3 | Global `classification.algorithms:` | Fallback — chạy tất cả global algo |

**Hyperparameter merge:** dataset params **override** global params (shallow merge 1 cấp).

#### Ví dụ — Classification:

```yaml
# --- Global (mặc định) ---
classification:
  test_size: 0.2
  algorithms:
    logistic:
      C: 1.0
      max_iter: 1000
    decision_tree:
      max_depth: 5
    svm:
      C: 1.0
      kernel: "rbf"

# --- Per-dataset override ---
datasets:
  student_performance:
    classification:
      target_columns: ["Grade", "Pass_Fail", "GPA_Category"]
      # svm KHÔNG có trong danh sách → sẽ không chạy
      algorithms:
        logistic:
          C: 0.5          # ← override global C: 1.0
          max_iter: 2000  # ← override global max_iter: 1000
        decision_tree:
          max_depth: 8    # ← override global max_depth: 5
```

#### Ví dụ — Regression:

```yaml
# --- Per-dataset override ---
datasets:
  stock_prices:
    regression:
      train_ratio: 0.85           # override global 0.8
      lags: [1, 5, 10, 20]        # override global lags
      # linear_regression KHÔNG có trong danh sách → sẽ không chạy
      algorithms:
        arima:
          order: [2, 1, 2]        # override global [1, 1, 1]
        xgboost:
          n_estimators: 300       # override global 200
          max_depth: 4            # override global 6
```

#### Ví dụ — Clustering:

Clustering luôn chạy đủ 3 giải thuật (K-Means + Hierarchical + DBSCAN). Có thể override tham số:

```yaml
datasets:
  student_performance:
    clustering:
      label_columns: ["Grade", "Pass_Fail", "GPA_Category"]
      max_k: 8          # override global max_k: 10
      n_clusters: 4     # override global n_clusters: 3
      dbscan_eps: 0.3   # override global dbscan_eps: 0.5
```

---

### 4.3 Đặt File Dữ Liệu

```
data/raw/
├── ten_dataset_cua_ban.csv
├── chuan_thoi_gian.csv
└── images/
    ├── class_1/img001.jpg
    └── class_2/img001.jpg
```

---

## 5. Chạy Từng Task

Cú pháp chung:

```
python src/presentation/run_pipeline.py --task <TASK> --dataset <TÊN_DATASET> [--algo <ALGO>] [--config <PATH>]
```

| Tham số | Bắt buộc | Mô tả |
|---------|----------|-------|
| `--task` | ✅ | `eda` \| `classification` \| `regression` \| `clustering` \| `all` |
| `--dataset` | ✅ | Tên dataset trong `params.yaml`, hoặc `all` để chạy tất cả |
| `--algo` | ❌ | Chỉ định 1 thuật toán (override per-dataset config, bỏ qua các algo khác) |
| `--config` | ❌ | Đường dẫn file config (mặc định: `configs/params.yaml`) |
| `--reset` | ❌ | Xóa checkpoint để buộc chạy lại từ đầu |

---

### 5.1 EDA

```bash
python src/presentation/run_pipeline.py \
  --task eda \
  --dataset student_performance \
  --config configs/params.yaml
```

**Output sinh ra:**
```
outputs/student_performance/eda/
├── raw/
│   ├── null_report.csv        # Báo cáo null/NaN trước xử lý
│   ├── data_raw.xlsx          # Dữ liệu gốc
│   ├── statistics.csv         # Mean, Median, Mode, IQR, CV,...
│   └── boxplot_*.png          # Boxplot phát hiện outlier
└── transformed/
    ├── data_transformed.xlsx  # So sánh trực tiếp với Excel
    ├── data_transformed.csv
    ├── hist_*.png
    ├── boxplot_*.png
    ├── scatter_matrix.png
    └── bar_*.png / pie_*.png
```

---

### 5.2 Classification (Nhóm A)

```bash
# Chạy tất cả algorithms
python src/presentation/run_pipeline.py \
  --task classification \
  --dataset student_performance \
  --config configs/params.yaml

# Chạy chỉ 1 thuật toán
python src/presentation/run_pipeline.py \
  --task classification \
  --dataset student_performance \
  --algo logistic \
  --config configs/params.yaml
```

**Thuật toán:** `logistic` | `decision_tree` | `svm`

**Output:**
```
outputs/student_performance/models/
├── clf_logistic_Grade.joblib
├── clf_decision_tree_Pass_Fail.joblib
└── clf_svm_GPA_Category.joblib
```

---

### 5.3 Regression (Nhóm B — Time Series)

```bash
python src/presentation/run_pipeline.py \
  --task regression \
  --dataset stock_prices \
  --config configs/params.yaml
```

> Pipeline dùng **Chronological Split** (80/20 theo thứ tự thời gian) — không có random split.

**Thuật toán:** `linear_regression` | `arima` | `xgboost`

**Output:**
```
outputs/stock_prices/
├── ml/regression/
│   ├── linear_Close.png       # Thực tế vs Dự đoán
│   ├── xgb_Close.png
│   └── arima_Close.png
└── models/
    ├── reg_linear_Close.joblib
    ├── reg_xgb_Close.joblib
    └── reg_arima_Close.pkl
```

---

### 5.4 Clustering (Nhóm A, B, C)

```bash
python src/presentation/run_pipeline.py \
  --task clustering \
  --dataset student_performance \
  --config configs/params.yaml
```

**Output:**
```
outputs/<dataset>/ml/clustering/
├── kmeans_elbow_silhouette.png   # Elbow Method + Silhouette Score
├── dendrogram.png                # Hierarchical Dendrogram
├── kmeans_labels.csv
├── hierarchical_labels.csv
└── dbscan_labels.csv             # cluster = -1 là outlier
```

---

## 6. Chạy Toàn Bộ Pipeline

| Nhóm | Task được chạy tự động |
|------|------------------------|
| A | EDA → Classification → Clustering |
| B | EDA → Regression → Clustering |
| C | Clustering |

```bash
# Chạy toàn bộ pipeline cho 1 dataset
python src/presentation/run_pipeline.py \
  --task all \
  --dataset student_performance \
  --config configs/params.yaml
```

### Chạy Tất Cả Datasets Cùng Lúc (`--dataset all`)

Truyền `--dataset all` để **tự động lặp qua tất cả** datasets khai báo trong `params.yaml`:

```bash
python src/presentation/run_pipeline.py \
  --task all \
  --dataset all \
  --config configs/params.yaml
```

> Pipeline đọc `config["datasets"]`, lần lượt chạy từng dataset theo `type`. Checkpoint bảo vệ: dataset nào đã `DONE` sẽ bị bỏ qua.

Cũng có thể kết hợp với `--task` cụ thể để chạy 1 task trên tất cả datasets:

```bash
# EDA cho tất cả datasets
python src/presentation/run_pipeline.py --task eda --dataset all

# Classification trên tất cả datasets (mỗi dataset dùng config riêng)
python src/presentation/run_pipeline.py --task classification --dataset all
```

### Chạy Nhiều Dataset Thủ Công

**Windows PowerShell:**
```powershell
$datasets = @("dataset_A1", "dataset_A2", "dataset_A3", "dataset_B1", "dataset_B2", "dataset_B3")
foreach ($ds in $datasets) {
    Write-Host "=== Chạy: $ds ===" -ForegroundColor Cyan
    python src/presentation/run_pipeline.py --task all --dataset $ds --config configs/params.yaml
}
```

**Linux / macOS:**
```bash
for ds in dataset_A1 dataset_A2 dataset_A3 dataset_B1 dataset_B2 dataset_B3; do
    echo "=== Chạy: $ds ==="
    python src/presentation/run_pipeline.py --task all --dataset "$ds" --config configs/params.yaml
done
```

---

## 7. Checkpoint & Resume

Pipeline tự động lưu tiến độ vào [`progress.json`](progress.json). Nếu bị ngắt giữa chừng, chạy lại lệnh cũ — các bước đã hoàn thành sẽ **bỏ qua tự động**.

### Xem Trạng Thái

```powershell
# Windows
Get-Content progress.json
```
```bash
# Linux / macOS
cat progress.json
```

Ví dụ:
```json
{
  "student_performance": {
    "eda": "DONE",
    "eda_completed_at": "2026-05-17T10:30:00",
    "classification": "FAILED",
    "classification_reason": "FileNotFoundError: ..."
  }
}
```

### Reset Checkpoint

```bash
# Reset toàn bộ dataset → chạy lại từ đầu
python src/presentation/run_pipeline.py --task all --dataset student_performance --reset --config configs/params.yaml

# Reset chỉ 1 bước cụ thể
python -c "from src.infrastructure.checkpoint import reset_step; reset_step('student_performance', 'clustering')"
```

---

## 8. Xem Kết Quả

### Tổng Hợp Metrics

```bash
python -c "
import pandas as pd
df = pd.read_csv('summary_results.csv')
print(df.to_string(index=False))
"
```

| Cột | Ý nghĩa |
|-----|---------|
| `timestamp` | Thời điểm chạy |
| `dataset` | Tên dataset |
| `task` | `classification` / `regression` / `clustering` |
| `algorithm` | Thuật toán đã dùng |
| `accuracy`, `precision`, `recall`, `f1` | Metrics phân loại |
| `mae`, `rmse`, `r2` | Metrics hồi quy |
| `silhouette_score`, `n_clusters` | Metrics phân cụm |

### Xem Log Thời Gian Thực

```powershell
# Windows
Get-Content logs/pipeline.log -Wait
```
```bash
# Linux / macOS
tail -f logs/pipeline.log
```

---

## 9. Làm Việc Nhóm Với Git & GitHub

### Thiết Lập Lần Đầu

```bash
# Clone repo về máy
git clone <repo_url>
cd CTH621_Nhom3

# Cài đặt môi trường (xem mục 2)
pip install -r requirements.txt

# Tạo nhánh làm việc cá nhân
git checkout develop
git checkout -b feature/<ten_thanh_vien>-<mo_ta>
# Ví dụ: git checkout -b feature/an-eda-group-a
```

### File Cần Commit và Không Nên Commit

Thêm vào `.gitignore`:

```gitignore
# Môi trường
.venv/
__pycache__/
*.pyc
*.egg-info/

# Dữ liệu thô (thường quá lớn cho git)
data/raw/
data/interim/

# Output sinh ra — không cần commit
outputs/
logs/

# File nhạy cảm
.env
wandb/
mlflow.db
```

**Nên commit:**
```
src/          ← Code nguồn
configs/      ← params.yaml (hyperparameter)
requirements.txt
environment.yml
progress.json ← Nếu muốn chia sẻ checkpoint với nhóm
summary_results.csv
```

### Chia Sẻ Dữ Liệu & Kết Quả

Vì `data/raw/` và `outputs/` không commit vào git, nhóm dùng một trong hai cách:

**Cách 1 — Google Drive / OneDrive (đơn giản):**
```
1. Tạo thư mục chung: CTH621_Nhom3_Data/
2. Upload data/raw/ và outputs/ lên đó
3. Các thành viên tải về và đặt vào đúng thư mục trước khi chạy
```

**Cách 2 — Git LFS (cho file lớn):**
```bash
# Cài Git LFS
git lfs install

# Track file lớn
git lfs track "data/raw/*.csv"
git lfs track "outputs/**/*.png"
git lfs track "outputs/**/*.joblib"

git add .gitattributes
git commit -m "chore: setup git lfs"
```

### Quy Ước Commit Message

```
feat(eda):        Thêm tính năng EDA mới
feat(ml):         Thêm thuật toán mới
fix(loader):      Sửa lỗi trong loader.py
config:           Cập nhật params.yaml
docs:             Cập nhật README
chore:            Cập nhật requirements, gitignore,...
```

---

## 10. Đồng Bộ Nhóm (W&B / MLflow)

### Weights & Biases

```bash
# Đăng nhập 1 lần
wandb login

# Bật trong params.yaml:
# tracking:
#   provider: "wandb"
#   wandb_project: "CTH621"

# Chạy bình thường — kết quả tự lên dashboard
python src/presentation/run_pipeline.py --task all --dataset student_performance --config configs/params.yaml
```

### MLflow

```bash
# Khởi động server
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# Bật trong params.yaml:
# tracking:
#   provider: "mlflow"
#   mlflow_tracking_uri: "http://localhost:5000"

# Mở UI: http://localhost:5000
```

---

## 11. Xử Lý Lỗi Thường Gặp

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|----------|
| `ModuleNotFoundError: No module named 'src'` | Chạy sai thư mục | `cd CTH621_Nhom3` rồi chạy lại |
| `FileNotFoundError: data/raw/...` | Sai đường dẫn trong `params.yaml` | Kiểm tra `file:` hoặc `dir:` |
| `ValueError: Không tìm thấy target_col` | Thiếu cấu hình dataset | Thêm `target_col` / `target_columns` vào `params.yaml` |
| `ValidationError: Số samples < 100` | Dataset chưa đủ ràng buộc | Kiểm tra lại file CSV |
| Kết quả khác nhau giữa các máy | Khác phiên bản thư viện | Chạy `pip install -r requirements.txt` đúng version |

### Thay Đổi Hyperparameter

Chỉnh sửa `configs/params.yaml` — **không bao giờ sửa file Python**:
```yaml
classification:
  algorithms:
    decision_tree:
      max_depth: 8      # ← Thay đổi ở đây, rồi chạy với --reset
```

---

## 12. Sử Dụng Skill GitHub Copilot

Dự án tích hợp **Copilot Skill `data-pipeline`** tại [`.github/skills/data-pipeline/SKILL.md`](.github/skills/data-pipeline/SKILL.md). Skill này hướng dẫn GitHub Copilot sinh code Python đúng chuẩn CTH621 (Clean Architecture, YAML config, bilingual charts, checkpoint, …).

### Cách Kích Hoạt

Trong VS Code, mở **Copilot Chat** (`Ctrl+Alt+I`) và gõ lệnh theo format:

```
#data-pipeline <mô tả task>
```

Hoặc Copilot tự động chọn skill phù hợp khi bạn mô tả task liên quan đến dự án.

### Các Task Được Hỗ Trợ

| Task | Prompt mẫu |
|------|------------|
| Scaffold project | `#data-pipeline scaffold toàn bộ project CTH621` |
| EDA cho dataset | `#data-pipeline eda cho dataset student_performance nhóm A` |
| Classification | `#data-pipeline classification với logistic và decision_tree cho student_performance` |
| Regression | `#data-pipeline regression time series cho stock_prices dùng XGBoost và ARIMA` |
| Clustering | `#data-pipeline clustering K-Means + DBSCAN cho image_dataset nhóm C` |
| Thêm dataset mới | `#data-pipeline thêm dataset mới tên heart_disease nhóm A vào params.yaml` |
| Sửa hyperparameter | `#data-pipeline chỉnh max_depth=10 và C=0.1 cho dataset student_performance` |

### Quy Tắc Code Copilot Sẽ Tuân Thủ

Mọi code sinh ra từ skill đều tự động đảm bảo:

- `random_state` đọc từ `params.yaml`, không hardcode
- Regression dùng Chronological Split, **không** `train_test_split` random
- Clustering **drop label columns** trước khi fit
- Biểu đồ lưu **2 ngôn ngữ**: `<tên>_en.png` và `<tên>_vi.png`
- Log qua `logging`, không dùng `print()`
- Hyperparameter đọc từ YAML qua `**kwargs`
- Output lưu theo cấu trúc `outputs/{dataset}/eda|ml|models/`
- Metrics append vào `summary_results.csv`

### Cập Nhật Skill

Khi cần thêm quy tắc mới, chỉnh sửa các file trong `.github/skills/data-pipeline/`:

```
.github/skills/data-pipeline/
├── SKILL.md                  ← Điểm vào chính, ràng buộc R1–R11
└── references/
    ├── eda-guide.md          ← Hướng dẫn EDA + bilingual charts
    ├── ml-guide.md           ← Hướng dẫn Classification/Regression/Clustering
    ├── infra-guide.md        ← Checkpoint, logging, CLI
    ├── project-structure.md  ← Cấu trúc thư mục chuẩn
    └── output-checklist.md   ← Checklist trước khi hoàn thành
```

---

## Tham Khảo Nhanh

| Lệnh | Mục đích |
|------|----------|
| `--task eda` | Phân tích thống kê + biểu đồ + tiền xử lý |
| `--task classification` | Phân loại (nhóm A) |
| `--task regression` | Hồi quy chuỗi thời gian (nhóm B) |
| `--task clustering` | Phân cụm (nhóm A, B, C) |
| `--task all` | Toàn bộ pipeline phù hợp với nhóm dataset |
| `--reset` | Xóa checkpoint, buộc chạy lại từ đầu |
| `--algo <tên>` | Chỉ chạy 1 thuật toán |
| `--config <path>` | Dùng file config khác |
- [Thành phần khác](#thành-phần-khác)

---

## Dataset

Tập hợp các bộ dữ liệu thực hành, phân loại theo dạng dữ liệu.

### Dữ liệu dạng bảng (Tabular)

| Tên dataset | Nguồn |
|---|---|
| Stroke Prediction Dataset | [Kaggle](https://www.kaggle.com/datasets/kukuroo3/stoke-prediction-dataset) |
| AI Impact on Job Sector | [Kaggle](https://www.kaggle.com/datasets/sumeakash/ai-impact-on-job-sector) |
| Predict Students' Dropout and Academic Success | [Kaggle](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention) |

### Dữ liệu chuỗi thời gian (Time Series)

| Tên dataset | Nguồn |
|---|---|
| Superstore Sales Dataset | [Kaggle](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) |
| Wind & Solar Energy Production Dataset | [Kaggle](https://www.kaggle.com/datasets/ahmeduzaki/wind-and-solar-energy-production-dataset) |
| Cinema Tickets | [Kaggle](https://www.kaggle.com/datasets/arashnic/cinema-ticket) |

### Dữ liệu hình ảnh (Image)

| Tên dataset | Nguồn |
|---|---|
| GRAPE Leaf Diseases | [Kaggle](https://www.kaggle.com/datasets/yusufmurtaza01/grape-leaf-diseases) |

### Dữ liệu âm thanh (Audio)

| Tên dataset | Nguồn |
|---|---|
| Heartbeat Sounds | [Kaggle](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds) |

---

## Thành phần khác

> Sẽ được bổ sung sau.