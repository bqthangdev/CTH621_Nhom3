TÀI LIỆU 1: ĐẶC TẢ YÊU CẦU DỰ ÁN (PROJECT SPECIFICATION)
Tên dự án: Bài tập nhóm học phần Phân Tích Dữ Liệu (Bậc Thạc sĩ - Hệ thống Thông tin)
Mục tiêu: Thu thập, tiền xử lý, phân tích thống kê, trực quan hóa và ứng dụng học máy trên 8 bộ dữ liệu đa dạng. Báo cáo kết quả theo chuẩn format khoa học.
Hiện trạng Workspace: Đã có mã nguồn, tài liệu báo cáo thô và 8 bộ dữ liệu [Chưa xác minh].

YÊU CẦU 1: THU THẬP VÀ PHÂN LOẠI DỮ LIỆU (TỔNG CỘNG >= 8 DATASETS)

Nhóm A (Tabular Data): Tối thiểu 3 bộ. Yêu cầu: >5 features (đa dạng kiểu dữ liệu như giới tính, tuổi, nghề nghiệp...), >100 samples.

Nhóm B (Time-Series Data): Tối thiểu 3 bộ. Yêu cầu: Dữ liệu chuỗi thời gian (doanh thu, chứng khoán, lượng mưa...), >500 samples.

Nhóm C (Multimedia Data): Tối thiểu 2 bộ thuộc 2 loại khác nhau trong 4 loại sau: Ảnh (gốc từ 256x256, >500 tấm), Audio (>200 samples), Video (>50 samples), Text (>1000 samples).

Nhiệm vụ chung Yêu cầu 1: Mô tả kỹ dữ liệu, phân tích rõ kết quả/kỹ thuật của các bài báo đã chạy trên dữ liệu này.

YÊU CẦU 2: PHÂN TÍCH THỐNG KÊ VÀ TRỰC QUAN HÓA (EDA)

Nền tảng lý thuyết: Nêu ý nghĩa các phép toán thống kê (Mean, Median, Mode, Percentiles, Range, Variance, Standard deviation, CV, IQR) và các loại biểu đồ (Histogram, Boxplot, Scatter, Line, Bar, Pie). Lấy tối thiểu 1 bộ từ Nhóm A, B, C để minh họa lý thuyết.

Thực hành Nhóm A & B (Chọn tối thiểu 5 biến/dataset): Phân loại bản chất biến (Định tính/Định lượng, Rời rạc/Liên tục). Tính toán thông số thống kê, lập bảng tần suất. Vẽ biểu đồ hệ thống (Histogram, Boxplot, Scatter, Bar/Pie cho A; Line Chart cho B). Nhận xét đối chiếu số liệu và hình học.

So sánh Tiền xử lý (Nhóm A & B): Thực hiện song song trên Dữ liệu thô (nhận diện Null/NaN, Outliers) và Dữ liệu sạch (Điền khuyết, loại bỏ Outliers, chuẩn hóa Min-Max/Z-score, Forward/Backward Fill, Log-Transform). Đánh giá sự thay đổi.

So sánh Công cụ: Chọn tối thiểu 1 bộ Nhóm A và 1 bộ Nhóm B để so sánh quá trình EDA bằng Excel/Spreadsheet và lập trình Python.

Thực hành Nhóm C (Multimedia): Số hóa thô thành các thuộc tính định lượng (Features). Thống kê và trực quan hóa theo 2 cấp độ: Tổng thể (Dataset-level) và Phân lớp/Phân vùng (Class-level). Nhận xét đặc sắc để tìm ra dấu hiệu vàng cho phân lớp.

YÊU CẦU 3: PHÂN TÍCH DỮ LIỆU VỚI MÁY HỌC (MACHINE LEARNING)

Bài toán Phân lớp (Classification) - Dữ liệu Nhóm A: Thực nghiệm 3 lần với 3 cột Target khác nhau. Chia tách Train/Test. Dùng tối thiểu 2 thuật toán (Logistic Regression, Decision Tree, Random Forest, SVM). Đánh giá: Confusion Matrix, Accuracy, Precision, Recall, F1-Score. Nhận xét mất cân bằng lớp.

Bài toán Hồi quy (Regression) - Dữ liệu Nhóm B: Dự đoán giá trị tương lai. Bắt buộc dùng Time Series Split (Không chia ngẫu nhiên). Thuật toán: Linear Regression, ARIMA/SARIMA, XGBoost, Random Forest Regressor. Đánh giá: MAE, RMSE, R^2. Nhận xét khả năng bắt xu hướng và tính mùa vụ.

Bài toán Gom cụm (Clustering) - Dữ liệu Nhóm A, B, C: Học không giám sát, ẩn nhãn Target. Thuật toán: K-Means, Hierarchical, DBSCAN. Đánh giá: Silhouette Score hoặc Elbow Method để chọn K. Lập bảng thống kê mô tả đặc trưng các cụm (Mean, Median, Mode) để "đọc vị" bản chất thực tế.

YÊU CẦU 4: BÁO CÁO KHOA HỌC VÀ MINH CHỨNG

Cấu trúc: Bìa, Lời cảm ơn, Mục lục (Nội dung, Hình, Bảng, Chữ viết tắt).

Chương 1: Cơ sở lý thuyết.

Chương 2: Thu thập, mô tả và tổng quan các bộ dữ liệu.

Chương 3: Phân tích đặc điểm dữ liệu qua thống kê và trực quan hóa.

Chương 4: Phân tích dữ liệu với học máy và đối sánh kết quả.

Phần cuối: Kết luận, Kiến nghị, Tài liệu tham khảo (IEEE/APA), Phụ lục (Minh chứng, Log thực nghiệm, Đường dẫn mã nguồn).

Quy tắc: Tất cả hình/bảng phải có phân tích/nhận xét. Lưu trữ toàn bộ code/log để minh chứng. Báo cáo định dạng chuẩn khoa học.