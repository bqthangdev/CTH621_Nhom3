---
name: data-pipeline
description: "Generate production-grade Python code for the CTH621 Master's data analysis project. Use when: generating EDA code, preprocessing pipelines, ML classification/regression/clustering, CLI batch scripts, checkpoint/fault-tolerance setup, logging, team sync with W&B or MLflow, or scaffolding the full project structure per CTH621 guidelines."
argument-hint: "Mô tả task cần generate: eda|classification|regression|clustering|infra|scaffold. Ví dụ: 'eda cho dataset_A1' hoặc 'scaffold toàn bộ project'"
---

# CTH621 Data Pipeline — Code Generation Skill

## Mục Đích
Skill này hướng dẫn AI sinh code Python chuẩn cho dự án phân tích dữ liệu CTH621 (Thạc sĩ Hệ thống thông tin). Mọi code sinh ra phải tuân thủ **toàn bộ** ràng buộc trong [guidelines.md](../../guidelines.md).

---

## ⚠️ Quy Tắc Đặc Biệt Quan Trọng

> Hai quy tắc dưới đây có **độ ưu tiên cao nhất**, áp dụng trong mọi tình huống, không có ngoại lệ.

1. **Báo cáo sau khi hoàn thành (R12):** Khi thực hiện xong bất kỳ task nào, AI phải báo cáo lại toàn bộ các công việc đã thực hiện — bao gồm file đã tạo/chỉnh sửa, logic chính và các quyết định thiết kế quan trọng.
2. **Hỏi lại khi thiếu thông tin (R13):** Luôn hỏi lại người dùng đối với những thông tin còn thiếu. Không tự ý bịa đặt hoặc sử dụng thông tin không chính thống, trừ khi người dùng yêu cầu rõ ràng.

---

## Khi Nào Dùng Skill Này
- Tạo code nạp/validate dữ liệu nhóm A (Tabular), B (Time Series), C (Multimedia)
- Sinh pipeline EDA: thống kê mô tả, trực quan hóa, tiền xử lý
- Sinh code ML: Classification, Regression (Time Series), Clustering
- Scaffold cấu trúc thư mục `src/` và `outputs/` chuẩn
- Tạo file `configs/params.yaml`, `requirements.txt`, `ecosystem.config.js` (PM2)
- Thiết lập checkpoint, logging, CLI (argparse/click)
- Tích hợp W&B / MLflow experiment tracking

---

## Quy Trình Sinh Code (Bắt Buộc Theo Thứ Tự)

### Bước 0 — Xác Nhận Input
Trước khi sinh code, AI phải xác nhận:
1. **Loại dataset:** A / B / C (và tên cụ thể nếu có)
2. **Task:** `eda` | `classification` | `regression` | `clustering` | `infra` | `scaffold`
3. **Target column(s)** (nếu là classification/regression)
4. Nếu thông tin chưa đủ → hỏi người dùng, không đoán.

### Bước 1 — Scaffold Cấu Trúc (nếu chưa có)
Tạo cấu trúc thư mục theo [project-structure.md](./references/project-structure.md):
```
src/ configs/ outputs/ logs/ data/
```

### Bước 2 — Data Layer (`src/data/`)
Sinh code theo [data-layer.md](./references/data-layer.md):
- `loader.py`: Đọc CSV/Parquet/ảnh/audio/video/text. Trả về `pd.DataFrame` hoặc `dict`.
- `validator.py`: Kiểm tra đầu vào từng nhóm (ràng buộc samples, features, kiểu dữ liệu).

### Bước 3 — EDA & Preprocessing (`src/domain/eda.py`)
Sinh code theo [eda-guide.md](./references/eda-guide.md):
- Phân loại biến tự động (Qualitative / Quantitative / Continuous / Discrete)
- Tính đầy đủ: Mean, Median, Mode, Percentiles, Range, Variance, SD, CV, IQR
- Biểu đồ đúng theo nhóm: Histogram, Boxplot, Scatter, Bar/Pie, Line Chart
- Pipeline 1 (raw) và Pipeline 2 (transformed) xuất ra `.xlsx`/`.csv`
- **Mỗi biểu đồ/sơ đồ sinh ra BẮT BUỘC có 2 phiên bản:** `<tên>_en.png` (nhãn tiếng Anh) và `<tên>_vi.png` (nhãn tiếng Việt) — xem R11

### Bước 4 — ML Modules (`src/domain/`)
Sinh code theo [ml-guide.md](./references/ml-guide.md):
- `classification.py`: ≥2 thuật toán, ≥3 target columns, Confusion Matrix + metrics
- `regression.py`: Time Series Split (KHÔNG random split), MAE/RMSE/R²
- `clustering.py`: K-Means + Hierarchical (Dendrogram) + DBSCAN, Elbow/Silhouette

### Bước 5 — Infrastructure (`src/infrastructure/`)
Sinh code theo [infra-guide.md](./references/infra-guide.md):
- `checkpoint.py`: `progress.json` tracking, skip-if-done logic
- `logger.py`: Python `logging` module, KHÔNG dùng `print`

### Bước 6 — CLI Entry Point (`src/presentation/run_pipeline.py`)
- Dùng `argparse` hoặc `click`
- Tham số: `--task`, `--dataset`, `--algo`, `--config`
- Load hyperparameters từ `configs/params.yaml`

### Bước 7 — Kiểm Tra Checklist Trước Khi Trả Code
Xem [output-checklist.md](./references/output-checklist.md). Tất cả ô phải ✅ trước khi hoàn thành.

---

## Ràng Buộc Tuyệt Đối (Không Được Vi Phạm)

| # | Ràng Buộc |
|---|-----------|
| R1 | `random_state` / `seed` đọc từ `params.yaml`, KHÔNG hardcode |
| R2 | Regression dùng Time Series Split, KHÔNG dùng `random_split` |
| R3 | Clustering phải drop cột label trước khi fit |
| R4 | KHÔNG dùng `print()` trong batch pipeline — chỉ dùng `logging` |
| R5 | Mỗi model class nhận `**kwargs` để map hyperparameter từ config |
| R6 | Output lưu theo cấu trúc `outputs/{dataset_name}/eda|ml/...` |
| R7 | Mọi kết quả metrics append vào `summary_results.csv` |
| R8 | Dataset nhóm A: >5 features với kiểu dữ liệu khác nhau, >100 samples |
| R9 | Dataset nhóm B: >500 samples time series |
| R10 | Dataset nhóm C: ≥2 trong 4 loại multimedia (ảnh/audio/video/text) |
| R11 | Mọi biểu đồ/sơ đồ/hình ảnh sinh ra phải có **2 phiên bản ngôn ngữ**: `<tên>_en.png` (tiếng Anh) và `<tên>_vi.png` (tiếng Việt). Dùng helper `_savefig_bilingual()` — KHÔNG lưu file không có hậu tố ngôn ngữ |
| **R12** ⚠️ | **Báo cáo sau khi hoàn thành:** Khi thực hiện xong task, phải báo cáo lại toàn bộ công việc đã thực hiện (file tạo/sửa, logic chính, quyết định thiết kế) |
| **R13** ⚠️ | **Hỏi lại khi thiếu thông tin:** Luôn hỏi lại người dùng khi thiếu thông tin. Không tự ý bịa đặt hoặc dùng thông tin không chính thống, trừ khi người dùng yêu cầu |

---

## Tham Khảo Nhanh

- Cấu trúc thư mục đầy đủ: [project-structure.md](./references/project-structure.md)
- Hướng dẫn EDA chi tiết: [eda-guide.md](./references/eda-guide.md)
- Hướng dẫn ML chi tiết: [ml-guide.md](./references/ml-guide.md)
- Hướng dẫn Infrastructure: [infra-guide.md](./references/infra-guide.md)
- Checklist output: [output-checklist.md](./references/output-checklist.md)
- Template params.yaml: [assets/params.yaml](./assets/params.yaml)
