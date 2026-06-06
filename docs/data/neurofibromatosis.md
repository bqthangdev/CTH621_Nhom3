### 2.1.2 Neurofibromatosis Type 1; Clinical Symptoms of Familial and Sporadic Cases

---

#### 2.1.2.1 Giới thiệu Tập dữ liệu

**Tên tập dữ liệu:** Neurofibromatosis Type 1; Clinical Symptoms of Familial and Sporadic Cases  
**Nguồn:** UCI Machine Learning Repository — [https://archive.ics.uci.edu/dataset/1162/neurofibromatosis+type+1+clinical+symptoms+of+familial+and+sporadic+cases](https://archive.ics.uci.edu/dataset/1162/neurofibromatosis+type+1+clinical+symptoms+of+familial+and+sporadic+cases)  
**Nhóm nghiên cứu tạo lập:** Parisa Sharafi, Hilal Arslan, Sibel Ersoy Evans, Ali Varan, Şükriye Ayter  
**Cơ quan thực hiện:** TOBB Üniversitesi Ekonomi ve Teknoloji, Ankara Yıldırım Beyazıt Üniversitesi, Hacettepe Üniversitesi — Thổ Nhĩ Kỳ  
**Năm công bố dữ liệu:** 2025 (được UCI nhận vào ngày 11/05/2025)  
**Giấy phép:** CC BY 4.0 (Creative Commons Attribution 4.0 International)  
**Bản chất dữ liệu:** Dữ liệu lâm sàng thực (Real-world clinical data) — thu thập hồi cứu (*retrospective*) từ hồ sơ y tế bệnh nhân NF1 tại Thổ Nhĩ Kỳ  
**Nhiệm vụ học máy:** Phân loại nhị phân (Binary Classification)  
**Biến mục tiêu chính:** `Case Type` — `0`: Trường hợp đơn lẻ (Sporadic), `1`: Trường hợp di truyền gia đình (Familial)  
**Biến mục tiêu phụ:** `Tumour Case` — `0`: Không có khối u, `1`: Có khối u

Neurofibromatosis type 1 (NF1) là một rối loạn thần kinh di truyền phổ biến do đột biến trên gen *NF1* (mã hóa protein neurofibromin) ở nhiễm sắc thể 17q11.2, với tần suất xuất hiện ước tính 1/3.000 đến 1/4.000 cá thể trên toàn thế giới. Biểu hiện lâm sàng của NF1 rất đa dạng, từ đốm cà phê sữa (*café-au-lait macules*) và nốt Lisch (*Lisch nodules*) đến các khối u thần kinh da, u thần kinh đám rối, u tuyến thần kinh thị giác và các bất thường xương. Điểm đặc thù của bệnh là tính **biến thiên kiểu hình rất cao** (*high phenotypic variability*), kể cả giữa các cá nhân trong cùng một gia đình mang cùng đột biến, khiến việc dự đoán diễn tiến lâm sàng gặp nhiều thách thức.

NF1 xuất hiện theo hai dạng: **đơn lẻ (sporadic)** — do đột biến mới phát sinh (*de novo*), không có tiền sử gia đình — và **gia đình (familial)** — di truyền theo kiểu trội nhiễm sắc thể thường với khoảng 50% nguy cơ truyền bệnh sang thế hệ sau. Phân biệt chính xác hai dạng này có ý nghĩa quan trọng trong tư vấn di truyền, lập kế hoạch can thiệp lâm sàng và nghiên cứu mối liên hệ kiểu gen–kiểu hình (*genotype-phenotype*).

Tập dữ liệu này được công bố cùng với bài báo giới thiệu của Sharafi và cộng sự (2025) [1], là **tập dữ liệu đầu tiên thuộc loại này** dành riêng cho bài toán phân biệt NF1 đơn lẻ và gia đình bằng các đặc trưng lâm sàng. Dữ liệu được thu thập hồi cứu từ hồ sơ y tế của bệnh nhân NF1 tại các cơ sở y tế ở Thổ Nhĩ Kỳ. Ngoài mục tiêu phân biệt sporadic/familial, tập dữ liệu cũng hỗ trợ bài toán thứ hai là phân loại sự hiện diện hay vắng mặt của khối u (`Tumour Case`).

**Phạm vi ứng dụng trong dự án:** Trong khuôn khổ dự án CTH621, tập dữ liệu được sử dụng để thực nghiệm các bài toán **phân loại nhị phân** (Classification — Nhóm A) với biến mục tiêu `Case Type`, đồng thời thực nghiệm **phân cụm không giám sát** (Clustering) nhằm khám phá cấu trúc dữ liệu tiềm ẩn trong không gian đặc trưng lâm sàng.

**Bảng 2.7 — Thông tin tổng quan tập dữ liệu Neurofibromatosis Type 1**

| Thuộc tính | Thông tin |
|:-----------|:----------|
| **Tên dataset** | Neurofibromatosis Type 1; Clinical Symptoms of Familial and Sporadic Cases |
| **Nguồn (UCI)** | [https://archive.ics.uci.edu/dataset/1162/...](https://archive.ics.uci.edu/dataset/1162/neurofibromatosis+type+1+clinical+symptoms+of+familial+and+sporadic+cases) |
| **Bài báo giới thiệu** | Sharafi et al. (2025), *Turkish Bulletin of Hygiene and Experimental Biology* |
| **DOI bài báo** | [10.5505/TurkHijyen.2025.06337](https://doi.org/10.5505/TurkHijyen.2025.06337) |
| **Năm tạo lập** | 2025 |
| **Giấy phép** | CC BY 4.0 |
| **Loại dữ liệu** | Dữ liệu dạng bảng (Tabular) — dữ liệu lâm sàng |
| **Số mẫu (CSV trong dự án)** | 295 bản ghi |
| **Số đặc trưng (CSV trong dự án)** | 19 cột (2 biến mục tiêu + 17 đặc trưng đầu vào) |
| **Kiểu đặc trưng** | Nhị phân (Binary) và Số nguyên (Integer) |
| **Nhiệm vụ học máy** | Phân loại nhị phân (Binary Classification) |
| **Bài toán phân loại** | Phân biệt NF1 đơn lẻ (sporadic) và NF1 gia đình (familial) |

---

#### 2.1.2.2 Thống kê Số lượng Mẫu

Tập dữ liệu không có sự phân tách tập con (train/test) được định nghĩa sẵn, toàn bộ dữ liệu được cung cấp trong một tệp duy nhất. Bảng 2.8 trình bày thống kê tổng quát của tập dữ liệu trong workspace dự án.

**Bảng 2.8 — Thống kê số lượng mẫu**

| Thông tin | Giá trị |
|:----------|--------:|
| Tổng số mẫu | 295 |
| Tổng số cột | 19 |
| Số đặc trưng đầu vào | 17 |
| Số biến mục tiêu | 2 (`Case Type`, `Tumour Case`) |
| Mẫu có giá trị thiếu (ít nhất 1 cột) | 73 |
| Kiểu tệp | CSV |

> **Lưu ý về số lượng mẫu:** Trang UCI mô tả bộ dữ liệu gốc bao gồm **331 trường hợp** (probands), trong khi bài báo giới thiệu báo cáo phân tích trên **241 bệnh nhân** (121 sporadic + 120 familial). Tệp CSV trong workspace dự án chứa **295 bản ghi**. Sự chênh lệch này có thể do quá trình lọc, xử lý dữ liệu đầu ra từ cơ sở dữ liệu gốc. Nhóm sử dụng tệp CSV hiện có (295 mẫu) cho mọi thực nghiệm.

##### 2.1.2.2.1 Phân phối Lớp — Biến mục tiêu chính (`Case Type`)

Phân phối nhãn trong biến mục tiêu chính phản ánh sự **mất cân bằng lớp ở mức độ nhẹ** (*mild class imbalance*):

**Bảng 2.9 — Phân phối lớp mục tiêu `Case Type`**

| Nhãn | Mô tả | Số lượng | Tỷ lệ |
|:----:|:------|--------:|------:|
| 0 | Trường hợp đơn lẻ (Sporadic) | 160 | 54,24% |
| 1 | Trường hợp gia đình (Familial) | 135 | 45,76% |
| **Tổng** | | **295** | **100%** |

**Nhận xét:** Tỷ lệ 54,24% / 45,76% thể hiện sự mất cân bằng lớp tương đối nhẹ, ít nghiêm trọng hơn so với nhiều bộ dữ liệu y tế thường gặp. Đường cơ sở ngẫu nhiên (*random baseline*) đạt khoảng ~54%. Tuy nhiên, cần lưu ý rằng trong bài toán phân loại lâm sàng, chi phí của phân loại sai (đặc biệt là False Negative trong tư vấn di truyền) vẫn cần được xem xét khi chọn chỉ số đánh giá.

##### 2.1.2.2.2 Phân phối Lớp — Biến mục tiêu phụ (`Tumour Case`)

**Bảng 2.10 — Phân phối lớp mục tiêu `Tumour Case`**

| Nhãn | Mô tả | Số lượng | Tỷ lệ |
|:----:|:------|--------:|------:|
| 0 | Không có khối u | 214 | 72,54% |
| 1 | Có khối u | 81 | 27,46% |
| **Tổng** | | **295** | **100%** |

**Nhận xét:** Phân phối biến mục tiêu phụ `Tumour Case` cho thấy mức độ mất cân bằng đáng kể hơn (~72,5%/~27,5%), cần áp dụng các kỹ thuật xử lý mất cân bằng phù hợp khi sử dụng làm nhãn phân loại.

---

#### 2.1.2.3 Mô tả Đặc trưng

Tập dữ liệu gồm 17 đặc trưng đầu vào và 2 biến mục tiêu, phân thành hai nhóm chính: **(1)** ba đặc trưng liên tục liên quan đến độ tuổi, và **(2)** mười bốn đặc trưng nhị phân biểu diễn sự hiện diện hay vắng mặt của các triệu chứng lâm sàng đặc trưng của NF1. Bảng 2.11 mô tả chi tiết từng cột.

**Bảng 2.11 — Mô tả đặc trưng của tập dữ liệu Neurofibromatosis Type 1**

| STT | Tên đặc trưng | Kiểu dữ liệu | Mô tả lâm sàng | Giá trị / Khoảng giá trị |
|:---:|:--------------|:-------------|:---------------|:--------------------------|
| — | `Case Type` | Nhị phân (0/1) — **Biến mục tiêu chính** | Phân loại dạng bệnh NF1 của bệnh nhân | `0` = Đơn lẻ (Sporadic), `1` = Gia đình (Familial) |
| — | `Tumour Case` | Nhị phân (0/1) — **Biến mục tiêu phụ** | Sự hiện diện của khối u | `0` = Không có khối u, `1` = Có khối u |
| 1 | `Age of Mother` | Số thực (float) | Tuổi người mẹ tại thời điểm bệnh nhân được sinh ra hoặc chẩn đoán | Liên tục; 16–45 tuổi; **35 giá trị thiếu** |
| 2 | `Age of Father` | Số thực (float) | Tuổi người cha tại thời điểm bệnh nhân được sinh ra hoặc chẩn đoán | Liên tục; 19–62 tuổi; **34 giá trị thiếu** |
| 3 | `Age at First Diagnosis` | Số thực (float) | Tuổi bệnh nhân tại thời điểm được chẩn đoán NF1 lần đầu | Liên tục; 0,5–61 tuổi; trung vị ~9 tuổi; **4 giá trị thiếu** |
| 4 | `Café au lait (CLS)` | Nhị phân (0/1) | Đốm cà phê sữa (*café-au-lait macules*) — dát sắc tố nâu trên da, một trong các tiêu chuẩn chẩn đoán NF1 chính | `0` = Vắng mặt, `1` = Hiện diện |
| 5 | `Axillary Freckles` | Nhị phân (0/1) | Tàn nhang nách (*Crowe's sign*) — chấm tàn nhang vùng nách, dấu hiệu đặc trưng của NF1 | `0` = Vắng mặt, `1` = Hiện diện |
| 6 | `Inguinal Freckles` | Nhị phân (0/1) | Tàn nhang bẹn — chấm tàn nhang vùng bẹn, tương tự Crowe's sign | `0` = Vắng mặt, `1` = Hiện diện |
| 7 | `Lisch Nodules` | Nhị phân (0/1) | Nốt Lisch (*Lisch nodules*) — hamartoma sắc tố mống mắt, là dấu hiệu bệnh lý đặc trưng và không gây giảm thị lực | `0` = Vắng mặt, `1` = Hiện diện |
| 8 | `Dermal Neurofibromins` | Nhị phân (0/1) | U xơ thần kinh da (*dermal neurofibromas*) — khối u lành tính dưới da, phát triển dọc theo dây thần kinh | `0` = Vắng mặt, `1` = Hiện diện |
| 9 | `Plexiform Neurofibromins` | Nhị phân (0/1) | U xơ thần kinh đám rối (*plexiform neurofibromas*) — khối u xâm lấn nhiều nhánh thần kinh, có nguy cơ ác tính hóa | `0` = Vắng mặt, `1` = Hiện diện |
| 10 | `Optic Glioma` | Nhị phân (0/1) | U thần kinh đệm thị giác (*optic pathway glioma*) — khối u tại đường dẫn truyền thị giác, ảnh hưởng thị lực | `0` = Vắng mặt, `1` = Hiện diện |
| 11 | `Skeletal Dysplasia` | Nhị phân (0/1) | Loạn sản xương (*skeletal dysplasia*) — bao gồm loãng xương, giả khớp, xương bướm | `0` = Vắng mặt, `1` = Hiện diện |
| 12 | `Learning Disability` | Nhị phân (0/1) | Khuyết tật học tập (*learning disability*) — khó khăn về nhận thức, học tập và trí tuệ | `0` = Vắng mặt, `1` = Hiện diện |
| 13 | `Hypertension` | Nhị phân (0/1) | Tăng huyết áp (*hypertension*) — thường liên quan đến hẹp động mạch thận hoặc u tủy thượng thận trong NF1 | `0` = Vắng mặt, `1` = Hiện diện |
| 14 | `Astrocytoma` | Nhị phân (0/1) | U tế bào hình sao (*astrocytoma*) — khối u não nguyên phát | `0` = Vắng mặt, `1` = Hiện diện |
| 15 | `Hamartoma` | Nhị phân (0/1) | Hamartoma — tổ chức phát triển bất thường nhưng lành tính | `0` = Vắng mặt, `1` = Hiện diện |
| 16 | `Scoliosis` | Nhị phân (0/1) | Vẹo cột sống (*scoliosis*) — biến dạng cột sống, biến chứng xương phổ biến trong NF1 | `0` = Vắng mặt, `1` = Hiện diện |
| 17 | `Other Symptoms` | Nhị phân (0/1) | Các triệu chứng khác — mã hóa sự hiện diện của ít nhất một trong các biểu hiện: Epilepsy, Rhabdomyoma, Ganglioblastoma, MPNST, Leukaemia, Hội chứng Noonan/Watson/Myelodysplastic, Khối u não/thân não (*cranial/brain stem tumour*) | `0` = Vắng mặt, `1` = Có ít nhất một triệu chứng |

##### 2.1.2.3.1 Vấn đề Chất lượng Dữ liệu

Phân tích sơ bộ (EDA) xác định các vấn đề chất lượng dữ liệu sau cần xử lý trong giai đoạn tiền xử lý:

Thứ nhất, **giá trị thiếu trong các đặc trưng độ tuổi:** Ba cột liên quan đến độ tuổi đều có giá trị thiếu, trong đó `Age of Mother` thiếu 35 giá trị (~11,9%), `Age of Father` thiếu 34 giá trị (~11,5%) và `Age at First Diagnosis` thiếu 4 giá trị (~1,4%). Đây là tình trạng phổ biến trong dữ liệu hồ sơ y tế hồi cứu, khi thông tin cha/mẹ không luôn được ghi nhận đầy đủ. Chiến lược điền khuyết phù hợp là dùng trung vị (median) hoặc trung bình (mean) của từng cột, hoặc phân tầng theo nhãn `Case Type`.

Thứ hai, **tất cả 14 đặc trưng nhị phân lâm sàng không có giá trị thiếu**, đảm bảo chất lượng đầy đủ cho phần lớn không gian đặc trưng.

Thứ ba, cần lưu ý rằng **trang UCI mô tả bộ dữ liệu không có giá trị thiếu**, trong khi tệp CSV thực tế trong dự án lại có giá trị thiếu rõ ràng trong các cột tuổi. Có thể tệp CSV được tạo ra từ phiên bản dữ liệu trung gian của quá trình tiền xử lý.

**Bảng 2.12 — Tổng hợp giá trị thiếu theo cột**

| Cột | Số giá trị thiếu | Tỷ lệ thiếu (%) | Loại biến |
|:----|----------------:|----------------:|:----------|
| `Age of Mother` | 35 | 11,86% | Liên tục |
| `Age of Father` | 34 | 11,53% | Liên tục |
| `Age at First Diagnosis` | 4 | 1,36% | Liên tục |
| 16 đặc trưng còn lại | 0 | 0% | Nhị phân |

---

#### 2.1.2.4 Tổng quan Các Nghiên cứu Liên quan

Tập dữ liệu Neurofibromatosis Type 1 được công bố vào tháng 5 năm 2025 cùng bài báo giới thiệu của Sharafi và cộng sự [1], là **tập dữ liệu mới nhất và duy nhất thuộc loại này** trong kho lưu trữ UCI tính đến thời điểm thực hiện dự án. Do bộ dữ liệu vừa được công bố, chưa có công trình bên ngoài nào độc lập sử dụng tập dữ liệu này để thực nghiệm học máy. Vì vậy, phần tổng quan tập trung phân tích chi tiết kết quả của bài báo giới thiệu.

##### 2.1.2.4.1 Phương pháp và Kết quả của Sharafi et al. (2025)

Nghiên cứu được thực hiện theo quy trình hồi cứu trên hồ sơ y tế 241 bệnh nhân NF1 (121 trường hợp đơn lẻ và 120 trường hợp gia đình). Để xác định các đặc trưng lâm sàng phân biệt rõ nhất giữa hai nhóm, nhóm tác giả áp dụng **Phân tích Phương sai (ANOVA)** như một bước lựa chọn đặc trưng (*feature selection*) trước khi huấn luyện mô hình.

Năm thuật toán học máy được thực nghiệm bao gồm: K-Nearest Neighbors (KNN), Mạng nơ-ron nhân tạo (ANN), Máy vectơ hỗ trợ (SVM), Cây quyết định (Decision Tree) và XGBoost. Đây là bộ thuật toán đa dạng, bao phủ các phương pháp từ học dựa trên khoảng cách, học sâu nông (*shallow neural network*), siêu phẳng phân tách, cây phân loại, đến kỹ thuật tăng cường gradient.

**Bảng 2.13 — Kết quả phân loại của Sharafi et al. (2025)**

| Thuật toán | Độ chính xác (Accuracy) | Ghi chú |
|:-----------|:-----------------------:|:--------|
| XGBoost | **62,86%** | Cao nhất trong các mô hình được thử nghiệm |
| ANN | — | Kết quả cụ thể không được báo cáo trong phần tóm tắt công bố |
| SVM | — | Kết quả cụ thể không được báo cáo trong phần tóm tắt công bố |
| Decision Tree | — | Kết quả cụ thể không được báo cáo trong phần tóm tắt công bố |
| KNN | — | Kết quả cụ thể không được báo cáo trong phần tóm tắt công bố |

> **Lưu ý:** Các chỉ số chi tiết (F1-Score, Precision, Recall, ROC-AUC) và kết quả đầy đủ của các thuật toán ngoài XGBoost không được trình bày trong phần tóm tắt công khai của bài báo. Nhóm cần tham khảo nội dung toàn văn bài báo để bổ sung bảng này.

##### 2.1.2.4.2 Phân tích Tổng hợp

Kết quả nghiên cứu của Sharafi et al. (2025) cho thấy một số nhận định quan trọng liên quan đến bài toán phân loại NF1 sporadic/familial bằng học máy.

Về hiệu năng tổng thể, mức độ chính xác tốt nhất đạt 62,86% (XGBoost) được nhóm tác giả đánh giá là **"độ tin cậy trung bình"** (*moderate reliability*) trong việc xác định trường hợp đơn lẻ. Con số này, mặc dù hạn chế theo tiêu chuẩn của học máy nói chung, phản ánh thực tế rằng **các đặc trưng lâm sàng quan sát được có tính chồng lấn cao** giữa nhóm sporadic và familial — phù hợp với sự hiểu biết y khoa hiện tại về tính biến thiên kiểu hình của NF1.

Về giá trị khoa học, dù hiệu năng phân loại ở mức trung bình, nghiên cứu đã xác định được sự khác biệt có ý nghĩa thống kê về phân phối triệu chứng lâm sàng giữa hai nhóm. Điều này gợi ý rằng **các yếu tố di truyền điều phối dùng chung (*shared genetic modifiers*)** có thể đóng vai trò quan trọng trong việc định hình mối quan hệ kiểu gen–kiểu hình trong NF1. Đây là cơ sở cho các hướng nghiên cứu tiếp theo về di truyền học chức năng của bệnh.

Về hướng cải thiện, các tác giả nhấn mạnh sự cần thiết của **bộ dữ liệu lớn hơn và đa dạng hơn** để nâng cao độ chính xác dự đoán. Với chỉ 241–295 mẫu và nhiều đặc trưng nhị phân thưa thớt (*sparse binary features*), mô hình học máy hiện tại bị giới hạn về khả năng tổng quát hóa. Các kỹ thuật tăng cường dữ liệu (*data augmentation*), học chuyển giao (*transfer learning*) hoặc tích hợp dữ liệu đa trung tâm có tiềm năng cải thiện đáng kể kết quả.

---

#### 2.1.2.5 Các Chỉ số Đánh giá

Với bài toán phân loại nhị phân trên tập dữ liệu lâm sàng này, nhóm sử dụng bộ chỉ số đánh giá như sau:

**Bảng 2.14 — Các chỉ số đánh giá mô hình phân loại**

| Chỉ số | Ký hiệu | Công thức | Ý nghĩa trong bài toán NF1 |
|:-------|:-------:|:----------|:---------------------------|
| Độ chính xác tổng thể | Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | Tỷ lệ phân loại đúng tổng thể trên toàn bộ mẫu |
| Độ nhạy / Recall | Sensitivity | $\frac{TP}{TP + FN}$ | Tỷ lệ phát hiện đúng trường hợp gia đình — quan trọng cho tư vấn di truyền |
| Độ đặc hiệu | Specificity | $\frac{TN}{TN + FP}$ | Tỷ lệ phân loại đúng trường hợp đơn lẻ |
| Độ chính xác dương tính | Precision | $\frac{TP}{TP + FP}$ | Tỷ lệ dự đoán "familial" thực sự chính xác |
| F1-Score | F1 | $\frac{2 \times Precision \times Recall}{Precision + Recall}$ | Trung bình điều hòa của Precision và Recall — phù hợp cho mất cân bằng nhẹ |
| Diện tích dưới đường ROC | ROC-AUC | — | Khả năng phân biệt hai lớp tại mọi ngưỡng quyết định |

---

### Tài liệu Tham khảo

[1] Sharafi, P., Arslan, H., Ersoy Evans, S., Varan, A., & Ayter, Ş. (2025). A machine learning approach for predicting familial and sporadic disease cases based on clinical symptoms: introduction of a new dataset. *Turkish Bulletin of Hygiene and Experimental Biology*, 82(1). https://doi.org/10.5505/TurkHijyen.2025.06337

---

### Trích dẫn LaTeX

```latex
% BibTeX entry cho bài báo giới thiệu dataset
@article{Sharafi2025,
  author    = {Sharafi, Parisa and Arslan, Hilal and {Ersoy Evans}, Sibel
               and Varan, Ali and Ayter, {\c{S}}{\"{u}}kriye},
  title     = {A machine learning approach for predicting familial and sporadic
               disease cases based on clinical symptoms: introduction of a new dataset},
  journal   = {Turkish Bulletin of Hygiene and Experimental Biology},
  volume    = {82},
  number    = {1},
  year      = {2025},
  doi       = {10.5505/TurkHijyen.2025.06337},
  url       = {https://doi.org/10.5505/TurkHijyen.2025.06337}
}

% BibTeX entry cho dataset trên UCI ML Repository
@misc{Sharafi2025dataset,
  author    = {Sharafi, Parisa and Arslan, Hilal and {Ersoy Evans}, Sibel
               and Varan, Ali and Ayter, {\c{S}}{\"{u}}kriye},
  title     = {{Neurofibromatosis Type 1; Clinical Symptoms of Familial and Sporadic Cases}
               [Dataset]},
  year      = {2025},
  publisher = {UCI Machine Learning Repository},
  doi       = {10.5505/TurkHijyen.2025.06337},
  url       = {https://archive.ics.uci.edu/dataset/1162/}
}
```

**Trích dẫn trong văn bản (IEEE / APA style):**

- IEEE: `[1] P. Sharafi, H. Arslan, S. Ersoy Evans, A. Varan, and Ş. Ayter, "A machine learning approach for predicting familial and sporadic disease cases based on clinical symptoms: introduction of a new dataset," *Turkish Bulletin of Hygiene and Experimental Biology*, vol. 82, no. 1, 2025, doi: 10.5505/TurkHijyen.2025.06337.`

- APA: `Sharafi, P., Arslan, H., Ersoy Evans, S., Varan, A., & Ayter, Ş. (2025). A machine learning approach for predicting familial and sporadic disease cases based on clinical symptoms: introduction of a new dataset. *Turkish Bulletin of Hygiene and Experimental Biology*, *82*(1). https://doi.org/10.5505/TurkHijyen.2025.06337`

---

## Ghi chú cần bổ sung

### [GHI CHÚ 1] Kết quả đầy đủ của bài báo (Bảng 2.13 còn thiếu)

Tóm tắt công khai của Sharafi et al. (2025) chỉ công bố kết quả của mô hình tốt nhất (XGBoost — 62,86%). Kết quả chi tiết của bốn thuật toán còn lại (KNN, ANN, SVM, Decision Tree) cũng như các chỉ số F1-Score, Precision, Recall và ROC-AUC của toàn bộ mô hình **chưa được thu thập** do toàn văn bài báo yêu cầu xác thực truy cập.

**Cần bổ sung:** Đề nghị nhóm cung cấp kết quả từ toàn văn bài báo (Table kết quả trong phần Results) để hoàn thiện Bảng 2.13. Các thông tin cần điền:
- Accuracy, F1-Score, Precision, Recall (và nếu có: ROC-AUC) của tất cả 5 thuật toán: KNN, ANN, SVM, Decision Tree, XGBoost.
- Thông tin về phương pháp chia tập (train/test ratio, cross-validation nếu có).
- Thông tin về feature selection: danh sách các đặc trưng được ANOVA chọn lọc.

### [GHI CHÚ 2] Bất nhất số lượng mẫu giữa các nguồn

Có sự chênh lệch về số lượng mẫu giữa ba nguồn thông tin:

| Nguồn | Số mẫu | Phân bố |
|:------|-------:|:--------|
| UCI ML Repository (metadata) | 331 | 167 sporadic + 142 familial (theo mô tả trên trang UCI) |
| Bài báo Sharafi et al. (2025) | 241 | 121 sporadic + 120 familial |
| CSV trong workspace dự án | 295 | 160 sporadic (Case Type=0) + 135 familial (Case Type=1) |

**Cần xác nhận:** Nhóm có biết lý do chênh lệch không? Ví dụ:
- CSV hiện tại có phải là phiên bản đã được lọc/tiền xử lý không?
- Hay đây là bản cập nhật mới hơn so với dữ liệu dùng trong bài báo?

Trong khi chờ xác nhận, nhóm sử dụng CSV hiện có (295 mẫu) cho mọi thực nghiệm và ghi chú sự khác biệt này trong phần phân tích kết quả.
