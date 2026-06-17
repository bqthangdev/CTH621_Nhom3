### 2.1.3 Predict Students' Dropout and Academic Success Dataset

#### 2.1.3.1 Giới thiệu Tập dữ liệu

**Tên tập dữ liệu:** Predict Students' Dropout and Academic Success  
**Nguồn tải về:** Kaggle — [https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)  
**Nguồn gốc:** Zenodo — [https://doi.org/10.5281/zenodo.5777340](https://doi.org/10.5281/zenodo.5777340)  
**Nhóm nghiên cứu tạo lập:** Valentim Realinho, Jorge Machado, Luís Baptista, Mónica V. Martins — Instituto Politécnico de Portalegre, Bồ Đào Nha  
**Năm công bố dữ liệu:** 2021  
**Giấy phép:** CC0 1.0 Universal (CC0 1.0) — Public Domain Dedication  
**Bản chất dữ liệu:** Dữ liệu thực (Real-world Data) — thu thập từ nhiều cơ sở dữ liệu nội bộ tại một cơ sở giáo dục đại học ở Bồ Đào Nha  
**Nhiệm vụ học máy:** Phân loại đa lớp (Multi-class Classification) — 3 lớp  
**Biến mục tiêu:** `Target` — `"Dropout"`, `"Enrolled"`, `"Graduate"`

Tập dữ liệu này được xây dựng từ dữ liệu thực của một cơ sở giáo dục đại học tại Bồ Đào Nha, thu thập từ nhiều hệ thống cơ sở dữ liệu rời rạc, bao gồm thông tin từ nhiều chương trình đại học như nông nghiệp, thiết kế, giáo dục, điều dưỡng, báo chí, quản lý, công tác xã hội và công nghệ. Bộ dữ liệu tích hợp ba nhóm thông tin chính: **(1)** thông tin hành chính và nhân khẩu học được ghi nhận tại thời điểm nhập học, **(2)** thông tin kinh tế-xã hội của sinh viên, và **(3)** kết quả học tập tại cuối học kỳ 1 và học kỳ 2.

Bài toán được hình thức hóa như một nhiệm vụ **phân loại 3 lớp**: tại thời điểm kết thúc thời gian học danh nghĩa của chương trình học (normal course duration), mỗi sinh viên được gán nhãn là *Dropout* (bỏ học), *Enrolled* (vẫn đang theo học), hoặc *Graduate* (tốt nghiệp). Điểm đặc biệt là nhãn "Enrolled" đại diện cho nhóm sinh viên vẫn đang theo học tại thời điểm khảo sát — đây là nhóm có kết quả chưa xác định cuối cùng, làm tăng độ khó của bài toán phân loại.

Mục tiêu ứng dụng của tập dữ liệu là xây dựng hệ thống cảnh báo sớm (*early warning system*), hỗ trợ nhóm cố vấn học thuật xác định sinh viên có nguy cơ bỏ học ngay từ đầu quá trình học, từ đó triển khai các biện pháp hỗ trợ kịp thời. Dự án được tài trợ bởi chương trình "SATDAP — Capacitação da Administração Pública" theo hợp đồng POCI-05-5762-FSE-000191, Bồ Đào Nha [1].

---

#### 2.1.3.2 Thống kê Số lượng Mẫu

Tập dữ liệu được cung cấp dưới dạng một tệp CSV duy nhất, không có phân tách tập huấn luyện/kiểm tra sẵn. Theo khuyến nghị của nhóm tác giả gốc [1], thực nghiệm nên sử dụng phân chia **80% huấn luyện / 20% kiểm tra** (80/20 split). Bảng 2.12 trình bày thống kê tổng quát về tập dữ liệu.

**Bảng 2.12. Thống kê số lượng mẫu**

| Thông tin | Giá trị |
|:----------|--------:|
| Tổng số mẫu | 4.424 |
| Tổng số cột | 35 |
| Số đặc trưng đầu vào | 34 |
| Biến mục tiêu | `Target` (3 lớp) |
| Giá trị thiếu (trên toàn bộ 35 cột) | 0 |
| Số đặc trưng kiểu số nguyên (`int64`) | 29 |
| Số đặc trưng kiểu số thực (`float64`) | 5 |
| Kiểu tệp | CSV (phân cách bằng dấu phẩy) |
| Phân chia khuyến nghị | 80% huấn luyện / 20% kiểm tra |

> **Lưu ý:** Tất cả 34 đặc trưng đầu vào đều là kiểu số. Không có cột chuỗi ký tự (string) nào trong tập đặc trưng — các biến phân loại danh nghĩa (nominal categorical) được mã hóa sẵn dưới dạng số nguyên. Việc giải mã nhãn phân loại cần tham chiếu từ tài liệu gốc của tập dữ liệu [1].

##### 2.1.3.2.1 Phân phối Lớp Mục tiêu (Target)

Phân phối lớp mục tiêu trong tập dữ liệu cho thấy **sự mất cân bằng lớp ở mức độ vừa phải** (mild class imbalance), với lớp "Graduate" chiếm gần một nửa tổng số mẫu và lớp "Enrolled" là nhóm thiểu số.

**Bảng 2.13. Phân phối lớp mục tiêu (`Target`)**

| Nhãn | Mô tả | Số lượng | Tỷ lệ |
|:-----|:------|--------:|------:|
| `Graduate` | Sinh viên đã tốt nghiệp tại thời điểm đánh giá | 2.209 | 49,93% |
| `Dropout` | Sinh viên đã bỏ học trước khi hoàn thành chương trình | 1.421 | 32,12% |
| `Enrolled` | Sinh viên vẫn đang theo học tại thời điểm đánh giá | 794 | 17,95% |
| **Tổng** | | **4.424** | **100%** |

**Nhận xét về phân phối lớp:**
- **Đường cơ sở ngẫu nhiên (Random Baseline):** Mô hình dự đoán toàn bộ là lớp đa số ("Graduate") đạt accuracy ~49,93%, không phải ~33,3% như trong trường hợp cân bằng hoàn toàn.
- **Lớp thiểu số "Enrolled"** (~18%) có thể bị phân loại nhầm nhiều hơn, đặc biệt là nhầm với "Graduate". Cần sử dụng kỹ thuật xử lý mất cân bằng như SMOTE, class-weight adjustment, hoặc đánh giá bằng macro-averaged F1 thay vì accuracy đơn thuần.
- **Ý nghĩa của lớp "Enrolled":** Đây là nhóm chưa có kết quả cuối cùng — về mặt học thuật, họ có thể tốt nghiệp hoặc bỏ học trong tương lai. Sự tồn tại của lớp này làm cho bài toán phức tạp hơn so với phân loại nhị phân Graduate/Dropout.

---

#### 2.1.3.3 Mô tả Đặc trưng

Tập dữ liệu gồm 34 đặc trưng đầu vào được chia thành bốn nhóm chức năng: thông tin nhân khẩu học và đăng ký nhập học, kết quả học kỳ 1, kết quả học kỳ 2, và các chỉ số kinh tế vĩ mô. Bảng 2.14 mô tả toàn bộ các đặc trưng theo thứ tự xuất hiện trong tệp CSV.

**Bảng 2.14. Mô tả đặc trưng của tập dữ liệu (34 đặc trưng + 1 biến mục tiêu)**

| STT | Tên cột | Kiểu | Nhóm | Mô tả | Phạm vi / Mã giá trị |
|:---:|:--------|:-----|:-----|:------|:----------------------|
| 1 | `Marital status` | int64 | Nhân khẩu học | Tình trạng hôn nhân của sinh viên | 1=Độc thân, 2=Đã kết hôn, 3=Góa, 4=Ly hôn, 5=Chung sống không kết hôn, 6=Ly thân hợp pháp |
| 2 | `Application mode` | int64 | Đăng ký | Phương thức nộp đơn vào trường | 17 mã (1=Đợt 1 tuyển sinh chung, 7=Chủ nhân bằng đại học khác, 39=Trên 23 tuổi, 42=Chuyển tiếp, …) |
| 3 | `Application order` | int64 | Đăng ký | Thứ tự ưu tiên của đơn đăng ký | 0=Lựa chọn thứ nhất → 9=Lựa chọn cuối cùng |
| 4 | `Course` | int64 | Đăng ký | Mã chương trình đào tạo đại học | 17 chương trình (Nông nghiệp, Thiết kế, Điều dưỡng, Báo chí, Quản lý, …) |
| 5 | `Daytime/evening attendance` | int64 | Đăng ký | Hình thức học | 1=Ban ngày, 0=Ban tối |
| 6 | `Previous qualification` | int64 | Nhân khẩu học | Mã loại bằng cấp/trình độ trước khi nhập học | 14 mã (1=THPT, 2=Cử nhân, 3=Bằng đại học, 4=Thạc sĩ, 5=Tiến sĩ, …) |
| 7 | `Nacionality` | int64 | Nhân khẩu học | Quốc tịch của sinh viên | 21 mã quốc gia (1=Bồ Đào Nha, 41=Brazil, 62=Romania, …) |
| 8 | `Mother's qualification` | int64 | Gia đình | Trình độ học vấn của mẹ | Thang mã hóa trình độ học vấn |
| 9 | `Father's qualification` | int64 | Gia đình | Trình độ học vấn của cha | Thang mã hóa trình độ học vấn |
| 10 | `Mother's occupation` | int64 | Gia đình | Ngành nghề của mẹ | Mã hóa danh mục nghề nghiệp |
| 11 | `Father's occupation` | int64 | Gia đình | Ngành nghề của cha | Mã hóa danh mục nghề nghiệp |
| 12 | `Displaced` | int64 | Nhân khẩu học | Sinh viên bị dời chỗ ở (displaced person) | 0=Không, 1=Có |
| 13 | `Educational special needs` | int64 | Nhân khẩu học | Có nhu cầu giáo dục đặc biệt | 0=Không, 1=Có |
| 14 | `Debtor` | int64 | Tài chính | Sinh viên đang nợ tiền học phí | 0=Không, 1=Có |
| 15 | `Tuition fees up to date` | int64 | Tài chính | Học phí đã được thanh toán đúng hạn | 0=Không, 1=Có |
| 16 | `Gender` | int64 | Nhân khẩu học | Giới tính | 1=Nam, 0=Nữ |
| 17 | `Scholarship holder` | int64 | Tài chính | Đang nhận học bổng | 0=Không, 1=Có |
| 18 | `Age at enrollment` | int64 | Nhân khẩu học | Tuổi tại thời điểm nhập học | 17–70 tuổi; TB ≈ 23,27 ± 7,59 |
| 19 | `International` | int64 | Nhân khẩu học | Sinh viên quốc tế | 0=Không, 1=Có |
| 20 | `Curricular units 1st sem (credited)` | int64 | Học kỳ 1 | Số môn học được công nhận (credited) trong học kỳ 1 | 0–20 |
| 21 | `Curricular units 1st sem (enrolled)` | int64 | Học kỳ 1 | Số môn học đã đăng ký trong học kỳ 1 | 0–26 |
| 22 | `Curricular units 1st sem (evaluations)` | int64 | Học kỳ 1 | Số lần thi/đánh giá trong học kỳ 1 | 0–45 |
| 23 | `Curricular units 1st sem (approved)` | int64 | Học kỳ 1 | Số môn đã qua (passed) trong học kỳ 1 | 0–26 |
| 24 | `Curricular units 1st sem (grade)` | float64 | Học kỳ 1 | Điểm trung bình các môn qua trong học kỳ 1 | 0,0–18,875; TB ≈ 10,64 ± 4,84 |
| 25 | `Curricular units 1st sem (without evaluations)` | int64 | Học kỳ 1 | Số môn không có đánh giá trong học kỳ 1 | 0–12 |
| 26 | `Curricular units 2nd sem (credited)` | int64 | Học kỳ 2 | Số môn học được công nhận (credited) trong học kỳ 2 | 0–19 |
| 27 | `Curricular units 2nd sem (enrolled)` | int64 | Học kỳ 2 | Số môn học đã đăng ký trong học kỳ 2 | 0–23 |
| 28 | `Curricular units 2nd sem (evaluations)` | int64 | Học kỳ 2 | Số lần thi/đánh giá trong học kỳ 2 | 0–33 |
| 29 | `Curricular units 2nd sem (approved)` | int64 | Học kỳ 2 | Số môn đã qua (passed) trong học kỳ 2 | 0–20 |
| 30 | `Curricular units 2nd sem (grade)` | float64 | Học kỳ 2 | Điểm trung bình các môn qua trong học kỳ 2 | 0,0–18,571 |
| 31 | `Curricular units 2nd sem (without evaluations)` | int64 | Học kỳ 2 | Số môn không có đánh giá trong học kỳ 2 | 0–12 |
| 32 | `Unemployment rate` | float64 | Kinh tế vĩ mô | Tỷ lệ thất nghiệp tại thời điểm đăng ký (%) | 7,6–16,2% |
| 33 | `Inflation rate` | float64 | Kinh tế vĩ mô | Tỷ lệ lạm phát tại thời điểm đăng ký (%) | −0,8% – 3,7% |
| 34 | `GDP` | float64 | Kinh tế vĩ mô | Tốc độ tăng trưởng GDP tại thời điểm đăng ký | −4,06 – 3,51 |
| 35 | **`Target`** | string | **Mục tiêu** | **Kết quả học tập của sinh viên** | `"Graduate"`, `"Dropout"`, `"Enrolled"` |

> **Lưu ý đặc trưng kinh tế vĩ mô:** Ba đặc trưng `Unemployment rate`, `Inflation rate`, và `GDP` phản ánh điều kiện kinh tế vĩ mô tại thời điểm sinh viên nhập học. Đây là đặc trưng cấp độ hệ thống (system-level features) — mọi sinh viên nhập học cùng kỳ chia sẻ cùng giá trị, dẫn đến hiện tượng đa cộng tuyến tiềm ẩn giữa ba đặc trưng này.

##### 2.1.3.3.1 Đặc điểm và Vấn đề Chất lượng Dữ liệu

Tập dữ liệu đã qua xử lý tiền xử lý nghiêm ngặt trước khi công bố, bao gồm xử lý dị thường, ngoại lệ không giải thích được và giá trị thiếu [1]. Tuy nhiên, một số đặc điểm cần lưu ý:

1. **Mã hóa số nguyên cho biến phân loại:** Phần lớn các đặc trưng phân loại danh nghĩa (Application mode, Course, Marital status, Nacionality, v.v.) được mã hóa dưới dạng số nguyên không liên tục. Nếu sử dụng các thuật toán nhạy cảm với thứ tự số (như hồi quy tuyến tính, SVM với nhân tuyến tính), cần áp dụng one-hot encoding hoặc ordinal encoding có chủ đích.

2. **Biến nhị phân dạng 0/1:** Các đặc trưng `Displaced`, `Educational special needs`, `Debtor`, `Tuition fees up to date`, `Gender`, `Scholarship holder`, `International`, `Daytime/evening attendance` là biến nhị phân đã mã hóa sẵn — không cần xử lý thêm.

3. **Điểm trung bình bằng 0:** Sinh viên bỏ học thường có giá trị 0,0 cho cả `Curricular units 1st sem (grade)` và `Curricular units 2nd sem (grade)` do không tham dự thi. Điều này có thể gây rò rỉ dữ liệu (data leakage) nếu các đặc trưng học kỳ 2 được sử dụng để dự đoán kết quả — cần cân nhắc phương án dự đoán dựa trên dữ liệu chỉ từ học kỳ 1 cho các mô hình cảnh báo sớm.

4. **Đặc trưng kinh tế vĩ mô đồng nhất theo thời điểm nhập học:** Ba đặc trưng macro là hằng số trong phạm vi từng đợt nhập học, làm giảm tính đa dạng thông tin thực sự của chúng.

---

#### 2.1.3.4 Tổng quan Nghiên cứu Liên quan

##### 2.1.3.4.1 Nghiên cứu Tiêu biểu

Kể từ khi được công bố năm 2021, tập dữ liệu này nhanh chóng trở thành **benchmark chuẩn** trong lĩnh vực nghiên cứu dự đoán bỏ học đại học. Tính đến năm 2025, bài báo gốc Realinho et al. (2022) đã được trích dẫn hơn 157 lần [1], trong khi bài báo hội nghị tiền thân Martins et al. (2021) nhận hơn 115 trích dẫn [2]. Bảng 2.15 tổng hợp các nghiên cứu tiêu biểu sử dụng hoặc liên quan đến tập dữ liệu này.

**Bảng 2.15. Tổng quan nghiên cứu liên quan đến tập dữ liệu Predict Students' Dropout**

| # | Tác giả | Năm | Công bố | Phương pháp chính | Đặc điểm nổi bật |
|:-:|:--------|:---:|:--------|:-----------------|:----------------|
| [1] | Realinho, V.; Machado, J.; Baptista, L.; Martins, M. V. | 2022 | *Data* (MDPI), 7(11), 146. DOI: 10.3390/data7110146 | LR, DT, RF, XGBoost, LightGBM, Gradient Boosting + SMOTE | **Bài báo dữ liệu gốc** tạo lập và mô tả bộ dữ liệu; cung cấp các mô hình phân loại tích hợp vào hệ thống tư vấn học thuật; phân tích tầm quan trọng đặc trưng |
| [2] | Martins, M. V.; Tolledo, D.; Machado, J.; Baptista, L.; Realinho, V. | 2021 | Springer AIST 2021. DOI: 10.1007/978-3-030-72657-7_16 | RF, GB, Gradient Boosting, phân tích đặc trưng quan trọng | **Nghiên cứu tiên phong** sử dụng tập dữ liệu này; RF dẫn đầu kết quả tổng thể; GB vượt trội trên hầu hết các lớp |
| [3] | Martins, M. V.; Baptista, L.; Machado, J.; Realinho, V. | 2023 | *Applied Sciences* (MDPI), 13(8), 4702. DOI: 10.3390/app13084702 | RF (multi-phase), tiếp cận dự đoán theo giai đoạn | **Phân loại đa giai đoạn** (phased prediction): RF đạt kết quả tốt nhất; thời điểm dự đoán tối ưu được phân tích kỹ; cùng nhóm tác giả gốc; 64 trích dẫn |
| [4] | Villar, A.; Andrade, C. R. V. | 2024 | *Discover Artificial Intelligence* (Springer). DOI: 10.1007/s44163-023-00079-z | LR, SVM, DT, RF, Gradient Boosting, XGBoost — nghiên cứu so sánh có hệ thống | Gradient Boosting vượt trội các phương pháp khác; so sánh toàn diện 6 thuật toán trên cùng tập dữ liệu; 133 trích dẫn |
| [5] | Goran, R.; Jovanovic, L.; Bacanin, N.; Stanković, M. S. et al. | 2024 | *IEEE Access*. DOI: 10.1109/ACCESS.2024.3440075 | Metaheuristic-optimized hybrid (RF + XGBoost + GB + FFNN) + SHAP + LIME | Kết hợp tối ưu hóa tham số bằng metaheuristic và giải thích mô hình bằng XAI; 49 trích dẫn |
| [6] | Bettahi, A.; Belouadha, F. Z.; Harroud, H. | 2025 | *Algorithms* (MDPI), 18(10), 662. DOI: 10.3390/a18100662 | XGBoost + pipeline mô-đun + SHAP | **Sử dụng trực tiếp tập dữ liệu này**; pipeline mô-đun gồm tiền xử lý, lựa chọn đặc trưng, XGBoost, giải thích SHAP; vượt qua các kết quả đã công bố; các mô hình chuẩn đạt accuracy 75–80% |
| [7] | Duro, B.; Gomes, A.; Borges, A. R.; Correia, F. B. | 2026 | *Procedia Computer Science* (Elsevier). DOI: 10.1016/j.procs.2026.xxx | Tổng quan hệ thống (Systematic Literature Review) — 71 nghiên cứu | **Tổng quan toàn diện** 71 nghiên cứu ML/DL về dự đoán bỏ học; các nghiên cứu hàng đầu báo cáo AUC lên tới 0,95; F1 ở mức cao trên nhiều tập dữ liệu |

##### 2.1.3.4.2 Phân tích Tổng hợp

**Xu hướng phương pháp học máy:** Qua các nghiên cứu từ 2021 đến 2026, một số nhận định có thể rút ra:

- **Thuật toán ensemble (RF, Gradient Boosting, XGBoost)** nhất quán dẫn đầu so với các phương pháp đơn lẻ như LR hay SVM [2][3][4][5]. RF được Martins et al. (2023) [3] xác nhận là tốt nhất trong tiếp cận phân loại đa giai đoạn.

- **Phạm vi accuracy điển hình** của các mô hình chuẩn (RF, GB, XGBoost) trên tập dữ liệu này là **75–80%** [6]; các mô hình tối ưu hóa nâng cao vượt ngưỡng này [5][6]. Đây là mức tham chiếu hữu ích để so sánh kết quả thực nghiệm.

- **Xử lý mất cân bằng lớp:** SMOTE và class-weight adjustment được sử dụng phổ biến [1][6] do lớp "Enrolled" chỉ chiếm ~18% mẫu.

- **Khả năng giải thích (Explainability):** Xu hướng mới từ 2024–2025 tập trung vào kết hợp XAI (SHAP, LIME) với các mô hình ensemble để cung cấp giải thích có thể hành động được cho đội ngũ tư vấn học thuật [5][6].

- **Thời điểm dự đoán:** Martins et al. (2023) [3] chỉ ra rằng dữ liệu học kỳ 2 cải thiện đáng kể độ chính xác so với chỉ dùng dữ liệu nhập học, nhưng dự đoán sớm (chỉ dựa trên dữ liệu nhập học hoặc học kỳ 1) quan trọng hơn về mặt can thiệp thực tiễn.

- **Đặc trưng quan trọng nhất:** Kết quả học kỳ 1 và 2 (số môn qua, điểm trung bình) nhất quán đứng đầu về tầm quan trọng trong hầu hết các nghiên cứu [1][2][3][4]. Các đặc trưng tài chính (`Tuition fees up to date`, `Debtor`) và nhân khẩu học (`Age at enrollment`) cũng có đóng góp đáng kể.

---

#### 2.1.3.5 Các Chỉ số Đánh giá Đề xuất

Do đây là bài toán **phân loại 3 lớp với mất cân bằng lớp**, việc chỉ dùng accuracy không phản ánh đầy đủ hiệu suất mô hình. Bảng 2.16 tổng hợp các chỉ số đánh giá phù hợp và cơ sở lựa chọn.

**Bảng 2.16. Các chỉ số đánh giá phù hợp cho bài toán dự đoán bỏ học (3 lớp, mất cân bằng)**

| Chỉ số | Ký hiệu | Phạm vi | Ý nghĩa trong ngữ cảnh | Ghi chú |
|:-------|:--------|:-------:|:----------------------|:--------|
| Accuracy | Acc | [0, 1] | Tỷ lệ dự đoán đúng trên toàn bộ mẫu | Không đủ với lớp mất cân bằng; cần dùng kèm F1 |
| Macro-averaged F1 | $F1_{\text{macro}}$ | [0, 1] | Trung bình F1 không trọng số của 3 lớp | **Chỉ số khuyến nghị chính** — đánh giá công bằng cả 3 lớp, kể cả lớp thiểu số "Enrolled" |
| Weighted-averaged F1 | $F1_{\text{weighted}}$ | [0, 1] | Trung bình F1 có trọng số theo số lượng mẫu từng lớp | Phù hợp khi quan tâm tới phân phối thực tế của lớp |
| Per-class F1 | $F1_c,\ c \in \{G, D, E\}$ | [0, 1] | F1 riêng cho từng lớp Graduate / Dropout / Enrolled | Chẩn đoán điểm yếu của mô hình trên lớp thiểu số |
| Precision (per-class) | $P_c$ | [0, 1] | Độ chính xác dự đoán dương tính của mỗi lớp | Quan trọng khi cần giảm báo động sai (false positives) |
| Recall (per-class) | $R_c$ | [0, 1] | Tỷ lệ phát hiện đúng trong lớp thực tế | Quan trọng khi cần phát hiện tối đa sinh viên có nguy cơ |
| Cohen's Kappa | $\kappa$ | [−1, 1] | Mức độ đồng thuận giữa dự đoán và thực tế, hiệu chỉnh theo cơ hội ngẫu nhiên | Phù hợp cho đánh giá trên bộ dữ liệu mất cân bằng đa lớp |
| Confusion Matrix | — | — | Ma trận nhầm lẫn 3×3 cho thấy mẫu nhầm lẫn giữa các cặp lớp | Phân tích định tính: Graduate↔Enrolled thường bị nhầm nhiều nhất |
| AUC-ROC (One-vs-Rest) | AUC | [0.5, 1] | Diện tích dưới đường cong ROC theo chiến lược One-vs-Rest | Các nghiên cứu hàng đầu báo cáo AUC lên tới 0,95 [7] |

> **Khuyến nghị thực nghiệm:** Trong ngữ cảnh hệ thống cảnh báo sớm, **Recall của lớp "Dropout"** ($R_{\text{Dropout}}$) có ý nghĩa đặc biệt quan trọng — bỏ sót một sinh viên có nguy cơ bỏ học (false negative) có hậu quả thực tiễn nghiêm trọng hơn việc phát cảnh báo nhầm (false positive). Do đó, tùy theo mục tiêu triển khai, có thể ưu tiên tối ưu hóa Recall lớp "Dropout" thay vì chỉ tối ưu macro-F1.

---

### Tài liệu Tham khảo

[1] V. Realinho, J. Machado, L. Baptista, và M. V. Martins, "Predicting Student Dropout and Academic Success," *Data*, tập 7, số 11, tr. 146, 2022. DOI: [10.3390/data7110146](https://doi.org/10.3390/data7110146)

[2] M. V. Martins, D. Tolledo, J. Machado, L. M. T. Baptista, và V. Realinho, "Early Prediction of Student's Performance in Higher Education: A Case Study," trong *Trends and Applications in Information Systems and Technologies*, tập 1, Springer AIST 2021, tr. 166–175. DOI: [10.1007/978-3-030-72657-7_16](https://doi.org/10.1007/978-3-030-72657-7_16)

[3] M. V. Martins, L. Baptista, J. Machado, và V. Realinho, "Multi-Class Phased Prediction of Academic Performance and Dropout in Higher Education," *Applied Sciences*, tập 13, số 8, tr. 4702, 2023. DOI: [10.3390/app13084702](https://doi.org/10.3390/app13084702)

[4] A. Villar và C. R. V. de Andrade, "Supervised Machine Learning Algorithms for Predicting Student Dropout and Academic Success: A Comparative Study," *Discover Artificial Intelligence*, Springer, 2024. DOI: [10.1007/s44163-023-00079-z](https://doi.org/10.1007/s44163-023-00079-z)

[5] R. Goran, L. Jovanovic, N. Bacanin, và M. S. Stanković et al., "Identifying and Understanding Student Dropouts Using Metaheuristic Optimized Classifiers and Explainable Artificial Intelligence Techniques," *IEEE Access*, 2024. DOI: [10.1109/ACCESS.2024.3440075](https://doi.org/10.1109/ACCESS.2024.3440075)

[6] A. Bettahi, F. Z. Belouadha, và H. Harroud, "A Modular and Explainable Machine Learning Pipeline for Student Dropout Prediction in Higher Education," *Algorithms*, tập 18, số 10, tr. 662, 2025. DOI: [10.3390/a18100662](https://doi.org/10.3390/a18100662)

[7] B. Duro, A. Gomes, A. R. Borges, và F. B. Correia, "Understanding and Preventing Student Dropout in Higher Education: A Literature Review," *Procedia Computer Science* (Elsevier), 2026. URL: [https://www.sciencedirect.com/science/article/pii/S187705092600565X](https://www.sciencedirect.com/science/article/pii/S187705092600565X)
