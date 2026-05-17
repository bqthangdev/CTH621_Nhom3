# HƯỚNG DẪN DÀNH CHO AI AGENT: DỰ ÁN PHÂN TÍCH DỮ LIỆU BẬC THẠC SĨ

## 1. Ngữ Cảnh Dự Án & Phạm Vi Hỗ Trợ
* **Mục tiêu:** Hỗ trợ lập trình viên hoàn thành các bài tập lập trình Python cho môn Phân tích dữ liệu thuộc chương trình Thạc sĩ ngành Hệ thống thông tin.
* **Phạm vi làm việc của AI:** Trọng tâm của AI agent là hỗ trợ viết code chuẩn xác cho các bước tiền xử lý, trực quan hóa, huấn luyện mô hình và đánh giá kết quả trên các bộ dữ liệu **đã được người dùng chuẩn bị sẵn**. AI tuyệt đối không tự ý đề xuất, tìm kiếm hay thay đổi chủ đề của bộ dữ liệu trừ khi được người dùng yêu cầu trực tiếp.

## 2. Yêu Cầu Nạp & Kiểm Tra Dữ Liệu (Data Validation)
Người dùng đã có sẵn dữ liệu. Khâu đầu tiên AI cần hỗ trợ là viết code để nạp (load) và tự động kiểm tra (validate) 3 nhóm dữ liệu này, đảm bảo chúng thỏa mãn các ràng buộc đầu vào sau:
* **Nhóm A (Tabular):** Dữ liệu dạng bảng với trên 5 features và trên 100 samples. Mỗi feature phải có kiểu dữ liệu khác nhau (ví dụ: giới tính, tuổi...).
* **Nhóm B (Time Series):** Dữ liệu chuỗi thời gian với trên 500 samples.
* **Nhóm C (Multimedia):** Hỗ trợ code đọc và số hóa ít nhất 2 trong 4 loại dữ liệu đa phương tiện:
    * Dữ liệu ảnh: Khung hình tối thiểu 256x256, ít nhất 500 tấm.
    * Dữ liệu audio: Tối thiểu 200 samples.
    * Dữ liệu video: Tối thiểu 50 samples.
    * Dữ liệu văn bản (text): Tối thiểu 1000 samples.
* **Tổng số lượng:** Đảm bảo pipeline mã nguồn được thiết kế linh hoạt, có thể duyệt qua và xử lý mượt mà toàn bộ ít nhất 8 bộ dữ liệu (Ít nhất 3 bộ loại A, 3 bộ loại B, và 2 bộ loại C).

## 3. Quy Trình Thống Kê & Trực Quan Hóa (Exploratory Data Analysis - EDA)
Khi sinh code thực hiện EDA, AI phải tuân thủ các bước phân tích sau:

### 3.1. Phân Tích Thống Kê & Phân Loại
* Lập trình để tự động phân loại bản chất biến: Qualitative (Định tính) hoặc Quantitative (Định lượng).
* Đối với biến Quantitative, code cần phân định rõ là số liên tục (Continuous) hay số rời rạc (Discrete).
* **Với biến Quantitative (Nhóm A & B):** Tính Mean, Median, Mode, Percentiles, Range, Variance, Standard Deviation, CV và IQR.
* **Với biến Qualitative (Nhóm A):** Lập bảng phân phối tần suất, tính tần suất tuyệt đối, tương đối và Mode.

### 3.2. Trực Quan Hóa Dữ Liệu
* **Nhóm A (Tabular):**
    * Histogram: Xem hình dáng phân phối của các biến định lượng.
    * Boxplot: Phát hiện Outliers.
    * Scatter Plot: Xem mức độ tương quan giữa các cặp biến định lượng.
    * Bar Chart / Pie Chart: Biểu diễn phân phối của các biến định tính.
* **Nhóm B (Time Series):**
    * Line Chart (trục hoành là thời gian): Xem xu hướng và tính mùa vụ — **bắt buộc**.
    * Histogram: Xem mức độ biến động của chuỗi.
* **Nhóm C (Multimedia):** Trích xuất đặc trưng thô thành các features định lượng (ví dụ: kích thước ảnh, độ sáng trung bình, cường độ âm thanh), sau đó thực hiện thống kê tổng thể và thống kê theo phân lớp trên các features này.

### 3.3. Tiền Xử Lý & So Sánh
* **Pipeline 1 — Dữ liệu Thô:** Nhận diện và báo cáo Null/NaN theo từng cột, xuất Boxplot chứa Outliers trước khi xử lý.
* **Pipeline 2 — Dữ liệu Đã Biến Đổi:**
    * Nhóm A: Điền khuyết bằng Mean/Median/Mode (tùy kiểu biến), xử lý Outliers bằng IQR-capping hoặc loại bỏ, chuẩn hóa Min-Max hoặc Z-score.
    * Nhóm B: Áp dụng Forward/Backward Fill, Linear Interpolation, Differencing hoặc Log-Transform.
* Xuất kết quả cả hai pipeline ra định dạng `.xlsx` / `.csv` để đối sánh trực tiếp với kết quả xử lý trên Excel/Spreadsheet.

## 4. Triển Khai Máy Học (Machine Learning)
Mã nguồn ML phải được chia rõ thành 3 module bài toán:

> **Ràng buộc tái lập (Reproducibility):** Mọi thuật toán có yếu tố ngẫu nhiên đều phải nhận tham số `random_state` (hoặc `seed`) từ file cấu hình `.yaml`. Tuyệt đối không hardcode giá trị này trong Python. Đây là điều kiện tiên quyết để kết quả nhất quán giữa các thành viên.

### 4.1. Bài toán Classification (Dữ liệu A)
* Thực nghiệm dự đoán với ít nhất 3 cột dữ liệu khác nhau làm Target/Y.
* Chia tập dữ liệu theo tỷ lệ chuẩn (ví dụ: Train 80% / Test 20%).
* Sử dụng tối thiểu 2 thuật toán (Logistic Regression, Decision Tree, SVM, v.v.).
* Đánh giá mô hình bằng Confusion Matrix, Accuracy, Precision, Recall và F1-Score. Code cần in ra log để xem có bị Imbalanced Data hay không.

### 4.2. Bài toán Regression (Dữ liệu B)
* Thiết lập mô hình dự đoán một giá trị số liên tục trong tương lai.
* **Ràng buộc ngặt:** Tuyệt đối không dùng Random Split. Phải triển khai Time Series Split / Train-Test Split theo mốc thời gian.
* Áp dụng các thuật toán như Linear Regression, ARIMA/SARIMA, XGBoost.
* Đánh giá độ sai số qua MAE, RMSE, và $R^2$ score.

### 4.3. Bài toán Clustering (Áp dụng cho cả 3 nhóm: A, B và C)
> Phải chạy Clustering độc lập trên từng bộ dữ liệu của mỗi nhóm — không gộp chung dữ liệu các nhóm với nhau.
* Drop/Loại bỏ hoàn toàn các cột nhãn Target trước khi nạp vào mô hình.
* Triển khai đồng thời 3 giải thuật: K-Means, Hierarchical Clustering (vẽ Dendrogram), và DBSCAN (để cô lập Outliers).
* Code cần xuất ra biểu đồ Elbow Method hoặc tính Silhouette Score để tìm K tối ưu.

## 5. Quy Chuẩn Đầu Ra Mạch Lạc
* Toàn bộ code, data, biểu đồ, kết quả và log terminal phải được lưu lại đầy đủ.
* Mọi block code sinh ra đều phải tuân thủ Clean Architecture, có document string rõ ràng để dễ dàng sao chép đưa vào báo cáo khoa học.
* **Cấu trúc thư mục output chuẩn** (AI phải tuân thủ khi lưu file):

```
outputs/
├── {dataset_name}/
│   ├── eda/
│   │   ├── raw/          # Boxplot, báo cáo Null/NaN trước xử lý
│   │   └── transformed/  # Biểu đồ sau tiền xử lý, file .xlsx/.csv đối sánh
│   ├── ml/
│   │   ├── classification/
│   │   ├── regression/
│   │   └── clustering/
│   └── models/           # File .pkl / .joblib checkpoint
logs/
│   └── pipeline.log
summary_results.csv
progress.json
```

## 6. Cơ Chế Thực Thi Batch Script & CLI (Command Line Interface)
* Bắt buộc sử dụng thư viện `argparse` hoặc `click` để thiết lập giao diện dòng lệnh linh hoạt. 
* Cấu trúc lệnh chạy chuẩn cần hỗ trợ truyền tham số đầy đủ, ví dụ: `python run_pipeline.py --task classification --dataset data_A1.csv --algo xgboost --config params.yaml`
* Tách biệt toàn bộ cấu hình siêu tham số (Hyperparameters) ra khỏi mã nguồn bằng cách sử dụng file `.yaml` hoặc `.json`. Tuyệt đối không hardcode siêu tham số trong file Python.
* Đối với mỗi thuật toán, AI phải thiết lập nhận `**kwargs` khi khởi tạo mô hình nhằm đảm bảo có thể map tự động tất cả các siêu tham số từ file cấu hình vào thuật toán mà không bị giới hạn.

## 7. Cơ Chế Checkpoint & Phục Hồi (Fault Tolerance)
* Triển khai hệ thống theo dõi tiến độ bằng file `progress.json` hoặc database nhẹ (như SQLite). Trước khi chạy một khối lượng công việc (ví dụ: EDA, Train, Evaluate), script phải kiểm tra xem trạng thái của bước đó đã được đánh dấu là "DONE" chưa. Nếu rồi thì tự động bỏ qua (Skip) và chuyển sang bước tiếp theo.
* Lưu trữ toàn bộ dữ liệu trung gian (dữ liệu sau EDA, dữ liệu sau điền khuyết/chuẩn hóa) dưới định dạng `.parquet` (ưu tiên tốc độ) hoặc `.csv` để script có thể nạp thẳng vào bước Train mà không cần chạy lại tiền xử lý.
* Trong quá trình huấn luyện, phải lưu lại trọng số (model weights) hoặc object mô hình thông qua `joblib` hoặc `pickle`. Nếu script bị dừng đột ngột, chương trình phải có khả năng tự động load lại model từ file checkpoint gần nhất.

## 8. Cấu Trúc Mã Nguồn & Quản Lý Tiến Trình
* Tổ chức mã nguồn chặt chẽ theo các nguyên lý của Clean Architecture. Tách biệt rõ ràng tầng đọc/ghi dữ liệu (Data), tầng xử lý thuật toán cốt lõi (Domain), và tầng tương tác dòng lệnh (Presentation) để dễ dàng bảo trì khi phải thay đổi cấu trúc dataset.

  **Sơ đồ thư mục mã nguồn chuẩn:**

  ```
  src/
  ├── presentation/       # CLI entry points (argparse / click)
  │   └── run_pipeline.py
  ├── domain/             # Thuật toán & business logic
  │   ├── eda.py
  │   ├── classification.py
  │   ├── regression.py
  │   └── clustering.py
  ├── data/               # Đọc, ghi, validate dữ liệu
  │   ├── loader.py
  │   └── validator.py
  ├── infrastructure/     # Logging, checkpoint, I/O phụ trợ
  │   ├── checkpoint.py
  │   └── logger.py
  configs/
  └── params.yaml         # Toàn bộ hyperparameter & random_state
  ```
* Thiết kế script để chạy ổn định trên các môi trường server Linux. Để đảm bảo an toàn khi treo máy chạy batch script lâu dài, AI nên sinh sẵn các file lệnh cấu hình để tương thích tốt với các trình quản lý tiến trình (như PM2). Điều này giúp script tiếp tục chạy ngầm ngay cả khi ngắt kết nối SSH và có khả năng tự khởi động lại khi crash.

## 9. Tiêu Chuẩn Ghi Log (Logging & Tracking)
* Cấm sử dụng hàm `print` thông thường cho quá trình chạy batch. Bắt buộc sử dụng module `logging` của Python để xuất log ra file (ví dụ: `pipeline.log`), ghi nhận rõ ràng mức độ (INFO, WARNING, ERROR) kèm Timestamp.
* Tất cả các kết quả độ đo cuối cùng (Accuracy, RMSE, Silhouette Score...) ứng với từng bộ siêu tham số phải được tự động ghi nối (append) vào một file tổng kết `summary_results.csv` để phục vụ trực tiếp cho việc trích xuất số liệu viết báo cáo nghiên cứu.

## 10. Cơ Chế Đồng Bộ Quá Trình Làm Việc Nhóm (Team Collaboration & State Sync)
Để đảm bảo quá trình huấn luyện và phân tích dữ liệu không bị lặp lại khi chuyển giao giữa các thành viên, AI cần thiết lập cơ chế quản lý trạng thái phân tán:

* **Quản Lý Checkpoint Qua Cloud Storage / DVC:** Cấu trúc mã nguồn phải cho phép thiết lập đường dẫn gốc (Base Path) cho tất cả các file trạng thái (`progress.json`), dữ liệu trung gian (`.parquet`) và trọng số mô hình (`.pkl`, `.joblib`).
  * Đường dẫn này phải linh hoạt để các thành viên có thể trỏ vào thư mục đồng bộ chung (ví dụ: Google Drive, OneDrive) hoặc tích hợp sẵn công cụ quản lý phiên bản dữ liệu như DVC (Data Version Control). Khi thành viên A chạy xong một bước, file trạng thái trên thư mục chung sẽ cập nhật thành "DONE", giúp máy của thành viên B tự động bỏ qua bước đó khi khởi chạy.

* **Tích Hợp Công Cụ Theo Dõi Thí Nghiệm (Experiment Tracking):**
  * Mã nguồn máy học cần tích hợp các thư viện MLOps chuyên nghiệp (ưu tiên sử dụng **Weights & Biases (W&B)** hoặc **MLflow**). 
  * Mỗi khi một kịch bản (run) được kích hoạt, hệ thống phải tự động log cấu hình siêu tham số, tiến độ chạy (epoch/step) và kết quả lên dashboard chung của nhóm. Thành viên B có thể theo dõi trực tiếp tiến trình của A trên trình duyệt và dễ dàng tải xuống (download) các artifact (mô hình, checkpoint) để chạy tiếp phần dự đoán hoặc gom cụm mà không cần bắt đầu lại.

* **Đồng Bộ Môi Trường Khép Kín:**
  * Mã nguồn sinh ra phải đi kèm file `requirements.txt` hoặc `environment.yml` (nếu dùng Conda) kèm theo version cụ thể của từng thư viện.
  * Cơ chế load checkpoint phải đi kèm một bước kiểm tra (validation) để đảm bảo môi trường trên máy của thành viên B giống hệt máy của thành viên A trước khi tiếp tục thực thi, tránh các lỗi xung đột do khác biệt phiên bản thư viện (đặc biệt là Pandas hoặc Scikit-Learn).