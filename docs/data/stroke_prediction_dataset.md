# Healthcare Dataset Stroke Data

## Nguồn và phạm vi sử dụng

Bộ dữ liệu được sử dụng trong pipeline là **Healthcare Dataset Stroke Data** do Aouatif Cherdid phân phối trên [Kaggle](https://www.kaggle.com/datasets/aouatifcherdid/healthcare-dataset-stroke-data), theo giấy phép CC0: Public Domain. Tệp đầu vào duy nhất là `healthcare-dataset-stroke-data.csv`; nguồn không cung cấp sẵn các tập huấn luyện, kiểm thử hoặc xác nhận.

Tệp cục bộ đã được đối chiếu với tệp tải trực tiếp từ nguồn Kaggle nêu trên. Nội dung dữ liệu khớp nhau; khác biệt byte chỉ xuất phát từ ký tự xuống dòng CRLF/LF. Vì vậy, mọi thống kê và kết quả mô hình trong dự án được tính trên toàn bộ 5.110 quan sát, sau đó pipeline tự tạo phép chia huấn luyện–kiểm thử.

| Thành phần | Giá trị |
|:---|---:|
| Số quan sát | 5.110 |
| Số cột trong tệp nguồn | 12 |
| Số biến dự báo sau khi loại `id` | 10 |
| Biến mục tiêu gốc | `stroke` |
| Phép chia do nguồn cung cấp | Không có |

## Cấu trúc dữ liệu

| Cột | Kiểu dữ liệu | Ý nghĩa và cách sử dụng trong dự án |
|:---|:---|:---|
| `id` | Số nguyên, định danh | Mã bệnh nhân; bị loại trước khi phân tích và huấn luyện |
| `gender` | Phân loại | `Male`, `Female` hoặc `Other` |
| `age` | Số thực | Tuổi của bệnh nhân |
| `hypertension` | Nhị phân | 1 nếu có tiền sử tăng huyết áp, ngược lại là 0 |
| `heart_disease` | Nhị phân | 1 nếu có tiền sử bệnh tim, ngược lại là 0 |
| `ever_married` | Phân loại nhị phân | Tình trạng đã từng kết hôn |
| `work_type` | Phân loại | Loại hình công việc |
| `Residence_type` | Phân loại nhị phân | Khu vực cư trú thành thị hoặc nông thôn |
| `avg_glucose_level` | Số thực | Mức đường huyết trung bình |
| `bmi` | Số thực | Chỉ số khối cơ thể |
| `smoking_status` | Phân loại | Tình trạng hút thuốc; `Unknown` biểu thị chưa xác định |
| `stroke` | Nhị phân, mục tiêu gốc | 1 nếu có đột quỵ, ngược lại là 0 |

`stroke` là nhãn gốc của bộ dữ liệu. Để đáp ứng thiết kế ba mục tiêu của bài tập, cấu hình dự án còn sử dụng `hypertension` và `heart_disease` làm hai mục tiêu phân loại phụ. Cách mở rộng này thuộc thiết kế thực nghiệm của nhóm, không phải mô tả nhiệm vụ gốc trên Kaggle.

## Chất lượng dữ liệu và phân phối mục tiêu

Kiểm tra trực tiếp tệp huấn luyện cho thấy `bmi` có 201 giá trị `N/A`, tương ứng 3,93% số quan sát. Cột `age` được đọc hoàn toàn dưới dạng số và không tồn tại giá trị lỗi `"*82"`. Giá trị `Unknown` trong `smoking_status` được giữ như một mức phân loại riêng vì không có căn cứ để suy ra trạng thái hút thuốc thực tế.

| Biến nhị phân | Lớp 0 | Lớp 1 | Tỷ lệ lớp 1 |
|:---|---:|---:|---:|
| `stroke` | 4.861 | 249 | 4,87% |
| `hypertension` | 4.612 | 498 | 9,75% |
| `heart_disease` | 4.834 | 276 | 5,40% |

Phân phối 4.861/249 của `stroke` cho thấy mức mất cân bằng lớp nghiêm trọng. Do đó, Accuracy không đủ để đánh giá khả năng phát hiện ca đột quỵ; Recall và F1-score của lớp dương, độ đặc hiệu và ma trận nhầm lẫn phải được báo cáo đồng thời.

## Tiền xử lý và thiết kế thực nghiệm hiện hành

Theo `configs/params.yaml` và mã nguồn EDA, pipeline loại `id`; điền giá trị thiếu của biến liên tục bằng trung bình; giới hạn ngoại lai theo quy tắc IQR; chuẩn hóa min–max các biến định lượng không được bảo vệ; và one-hot encode năm biến phân loại. Ba cột `stroke`, `hypertension` và `heart_disease` được bảo vệ khỏi bước capping và chuẩn hóa.

Sau tiền xử lý, dữ liệu có 22 cột và không còn giá trị thiếu. Với mỗi mục tiêu phân loại, pipeline dùng phép chia 80/20 có phân tầng và Stratified K-Fold 5 phần. Logistic Regression, Decision Tree và Random Forest dùng `class_weight="balanced"`; cấu hình Gradient Boosting hiện không có cơ chế cân bằng lớp tương đương.

Hai giới hạn phương pháp cần được lưu ý khi đọc kết quả hiện hành. Thứ nhất, transformer EDA được fit trên toàn bộ dữ liệu trước phép chia, nên thống kê điền khuyết và chuẩn hóa có thể rò sang tập kiểm thử. Thứ hai, khi dự báo một trong ba mục tiêu, mã huấn luyện chỉ loại mục tiêu đang xét; hai nhãn còn lại vẫn có thể xuất hiện trong tập đầu vào. Một đánh giá nghiêm ngặt hơn cần đặt toàn bộ bước tiền xử lý trong pipeline chỉ fit trên tập huấn luyện và loại đồng thời các nhãn không có sẵn tại thời điểm dự báo.

## Mốc nghiên cứu để diễn giải

Akinwumi và cộng sự đánh giá nhiều mô hình trên 5.110 hồ sơ có cùng hệ biến lâm sàng và chỉ ra một nghịch lý quan trọng của dữ liệu mất cân bằng: các mô hình có thể đạt Accuracy xấp xỉ 95% nhưng hầu như không phát hiện được ca dương tính. Gradient Boosting trong nghiên cứu báo cáo TP=1, TN=968, FP=4 và FN=49; Logistic Regression báo cáo TP=0, TN=972, FP=0 và FN=50. Mốc này được dùng để giải thích hạn chế của Accuracy, không được xem là phép so sánh tuyệt đối vì giao thức chia mẫu và tiền xử lý có thể khác dự án.

Nguồn nghiên cứu: P. O. Akinwumi *và cộng sự*, “Evaluating machine learning models for stroke prediction based on clinical variables,” *Frontiers in Neurology*, tập 16, bài 1668420, 2025, doi: [10.3389/fneur.2025.1668420](https://doi.org/10.3389/fneur.2025.1668420).
