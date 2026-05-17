# Output Checklist — CTH621

AI phải kiểm tra TẤT CẢ các mục dưới đây trước khi trả code cho người dùng.

## Architecture & Config
- [ ] `random_state` đọc từ `params.yaml`, KHÔNG hardcode trong Python
- [ ] Mọi hyperparameter đọc từ `params.yaml` / `.json`, không hardcode
- [ ] Mỗi model class nhận `**kwargs` khi khởi tạo
- [ ] Clean Architecture: tách biệt `data/`, `domain/`, `infrastructure/`, `presentation/`
- [ ] CLI sử dụng `argparse` hoặc `click` với các tham số `--task`, `--dataset`, `--algo`, `--config`

## Data Validation
- [ ] Validator Group A: check >5 features, >100 samples, kiểu dữ liệu đa dạng
- [ ] Validator Group B: check >500 samples, datetime index hoặc datetime column
- [ ] Validator Group C: check loại multimedia và số lượng samples

## EDA
- [ ] Phân loại biến tự động: Qualitative / Quantitative Continuous / Quantitative Discrete
- [ ] Tính đủ: Mean, Median, Mode, P5/P25/P50/P75/P95, Range, Variance, SD, CV, IQR
- [ ] Group A: Histogram, Boxplot, Scatter, Bar/Pie
- [ ] Group B: Line Chart (bắt buộc), Histogram
- [ ] Group C: Trích xuất numeric features, rồi thống kê như Group A
- [ ] Pipeline 1 (raw): null report + boxplot → `outputs/{name}/eda/raw/`
- [ ] Pipeline 2 (transformed): fill null + IQR-cap + normalize → `outputs/{name}/eda/transformed/`
- [ ] Xuất `.xlsx` và `.csv` để đối sánh Excel
- [ ] Lưu `.parquet` trung gian vào `data/interim/`

## Machine Learning
- [ ] Classification: ≥3 target columns, ≥2 algorithms, Confusion Matrix + 4 metrics, log imbalance
- [ ] Regression: Time Series Split (KHÔNG random), LAG features, MAE + RMSE + R²
- [ ] Clustering: K-Means + Hierarchical + DBSCAN, Elbow/Silhouette, drop label columns
- [ ] Tất cả model được lưu `.joblib` vào `outputs/{name}/models/`

## Infrastructure
- [ ] KHÔNG dùng `print()` — chỉ dùng `logging` với timestamp và level
- [ ] `progress.json` checkpoint: skip-if-DONE trước mỗi bước
- [ ] Metrics append vào `summary_results.csv`
- [ ] `ecosystem.config.js` có cho PM2 (server deployment)

## Team Sync
- [ ] `requirements.txt` với version cụ thể
- [ ] Environment validation check khi startup
- [ ] W&B hoặc MLflow integration cho experiment tracking

## Output Structure
- [ ] Tất cả file output trong `outputs/{dataset_name}/eda/` hoặc `outputs/{dataset_name}/ml/`
- [ ] Log ghi vào `logs/pipeline.log`
