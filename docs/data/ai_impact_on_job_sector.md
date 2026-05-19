### 2.1.2 AI Impact on Job Sector Dataset

#### 2.1.2.1 Giới thiệu Tập dữ liệu

**Tên tập dữ liệu:** AI Impact on Job Sector  
**Nguồn:** Kaggle — Akash Kumar Barnwal (sumeakash), [https://www.kaggle.com/datasets/sumeakash/ai-impact-on-job-sector](https://www.kaggle.com/datasets/sumeakash/ai-impact-on-job-sector)  
**Giấy phép:** CC0: Public Domain  
**Bản chất dữ liệu:** Tổng hợp nhân tạo (Synthetic) — được tạo ra cho mục đích giáo dục và phân tích; không đại diện cho cá nhân hay tổ chức thực tế  
**Nhiệm vụ học máy:** Phân loại đa lớp (Multi-class Classification) / Phân tích xu hướng  
**Nhãn mục tiêu:** `Job_Status` — `"Unchanged"`, `"Modified"`, `"Replaced"`

Tập dữ liệu này cung cấp một bức tranh tổng hợp về tác động của việc áp dụng Trí tuệ nhân tạo (AI) đến lực lượng lao động trên nhiều lĩnh vực công nghiệp. Mỗi bản ghi mô phỏng một nhân viên với các thông tin về nhân khẩu học, kinh nghiệm, ngành nghề, mức độ triển khai AI, rủi ro tự động hóa, biến động lương, giờ làm việc và các chỉ số hài lòng — tất cả được đặt trong bối cảnh chuyển đổi do AI gây ra.

Câu hỏi nghiên cứu trung tâm của tập dữ liệu là: **"Với các đặc trưng về nghề nghiệp và môi trường AI, liệu công việc của một nhân viên sẽ không đổi, bị điều chỉnh, hay bị thay thế hoàn toàn?"** — đây là bài toán phân loại 3 lớp có giá trị thực tiễn trong hoạch định chính sách nhân lực và định hướng tái đào tạo.

---

#### 2.1.2.2 Thống kê Số lượng Mẫu

Tập dữ liệu được cung cấp dưới dạng một tệp CSV duy nhất, không có tập huấn luyện/kiểm tra phân tách sẵn. Bảng 2.7 trình bày thống kê tổng quát.

**Bảng 2.7. Thống kê số lượng mẫu**

| Thông tin | Giá trị |
|:----------|--------:|
| Tổng số mẫu | 2.000 |
| Tổng số cột | 17 |
| Số đặc trưng đầu vào | 16 |
| Biến mục tiêu | `Job_Status` |
| Giá trị thiếu (trên toàn bộ 17 cột) | 0 |
| Kiểu tệp | CSV (một tệp duy nhất) |
| Kích thước tệp | ~197 kB |

> **Lưu ý:** Không có tập kiểm tra tách biệt. Việc chia tập huấn luyện/xác nhận/kiểm tra cần thực hiện thủ công trong quá trình thực nghiệm.

##### 2.1.2.2.1 Phân phối Lớp Mục tiêu (Job_Status)

Phân phối lớp mục tiêu trong tập dữ liệu đạt trạng thái **cân bằng hoàn toàn** (balanced classes), là điểm khác biệt đáng chú ý so với nhiều bộ dữ liệu thực tế thường có sự mất cân bằng lớp.

**Bảng 2.8. Phân phối lớp mục tiêu (`Job_Status`)**

| Nhãn | Mô tả | Số lượng | Tỷ lệ |
|:-----|:------|--------:|------:|
| `Unchanged` | Công việc không thay đổi sau khi AI được triển khai | ~667 | ~33,35% |
| `Modified` | Công việc bị điều chỉnh/tái định hình do AI | ~667 | ~33,35% |
| `Replaced` | Công việc bị thay thế hoàn toàn bởi AI/tự động hóa | ~666 | ~33,30% |
| **Tổng** | | **2.000** | **100%** |

**Nhận xét:** Tỷ lệ phân bổ ~33,3% đồng đều cho cả ba lớp là **đặc trưng của dữ liệu tổng hợp**, được thiết kế có chủ ý để tạo thuận lợi cho việc học máy mà không cần xử lý mất cân bằng. Trong thực tế, phân phối này nhiều khả năng không cân bằng — làm cho mô hình học được trên dữ liệu này cần được đánh giá cẩn thận trước khi áp dụng vào dữ liệu thực.

**Đường cơ sở ngẫu nhiên (Random Baseline):** Với 3 lớp cân bằng, mô hình dự đoán ngẫu nhiên đạt ~33,3% accuracy. Bất kỳ mô hình học máy nào cũng cần vượt đáng kể mức này.

---

#### 2.1.2.3 Mô tả Đặc trưng

Tập dữ liệu gồm 16 đặc trưng đầu vào và 1 biến mục tiêu, bao phủ bốn nhóm thông tin: nhân khẩu học nhân viên, bối cảnh ngành nghề, điều kiện làm việc, và chỉ số tác động AI. Bảng 2.9 mô tả chi tiết từng cột.

**Bảng 2.9. Mô tả đặc trưng của tập dữ liệu AI Impact on Job Sector**

| STT | Tên cột | Kiểu dữ liệu | Nhóm | Mô tả | Giá trị / Khoảng giá trị |
|:---:|:--------|:-------------|:-----|:------|:--------------------------|
| 1 | `Employee_ID` | Định danh (object) | — | Mã định danh duy nhất của nhân viên | `E0001` – `E2000`; không sử dụng trong mô hình |
| 2 | `Age` | Số nguyên (int64) | Nhân khẩu học | Tuổi của nhân viên (năm) | 22 – 59; TB = 40,56 ± 10,79 |
| 3 | `Gender` | Phân loại danh nghĩa | Nhân khẩu học | Giới tính | `"Male"`, `"Female"` |
| 4 | `Education_Level` | Phân loại thứ bậc | Nhân khẩu học | Trình độ học vấn cao nhất | `"High School"`, `"Bachelor"`, `"Master"`, `"PhD"` |
| 5 | `Industry` | Phân loại danh nghĩa | Ngành nghề | Lĩnh vực công nghiệp | `"IT"`, `"Healthcare"`, `"Finance"`, `"Manufacturing"`, `"Education"`, `"Retail"`, `"Marketing"`, `"Transportation"` |
| 6 | `Job_Role` | Phân loại danh nghĩa | Ngành nghề | Vị trí/chức danh cụ thể trong ngành | Nhiều giá trị (Content Creator, DevOps Engineer, Quality Inspector, …) |
| 7 | `Years_Experience` | Số nguyên (int64) | Nhân khẩu học | Số năm kinh nghiệm làm việc | 0 – 37; TB = 16,66 ± 10,75 |
| 8 | `AI_Adoption_Level` | Phân loại thứ bậc | Tác động AI | Mức độ ứng dụng AI trong vai trò công việc hiện tại | `"Low"`, `"Medium"`, `"High"` |
| 9 | `Automation_Risk` | Phân loại thứ bậc | Tác động AI | Mức độ rủi ro công việc bị tự động hóa | `"Low"`, `"Medium"`, `"High"` |
| 10 | `Upskilling_Required` | Nhị phân danh nghĩa | Tác động AI | Công việc có yêu cầu nâng cấp kỹ năng do AI không | `"Yes"`, `"No"` |
| 11 | `Salary_Before_AI` | Số nguyên (int64) | Tác động AI | Mức lương hàng năm trước khi AI được triển khai (USD) | 30.036 – 119.976; TB = 73.942 |
| 12 | `Salary_After_AI` | Số nguyên (int64) | Tác động AI | Mức lương hàng năm sau khi AI được triển khai (USD) | 24.447 – 161.745; TB = 78.429 |
| 13 | `Job_Status` | Phân loại danh nghĩa — **Biến mục tiêu** | — | Tình trạng công việc sau chuyển đổi AI | `"Unchanged"`, `"Modified"`, `"Replaced"` |
| 14 | `Work_Hours_Per_Week` | Số nguyên (int64) | Điều kiện làm việc | Số giờ làm việc trung bình mỗi tuần | 35 – 54; TB = 44,85 |
| 15 | `Remote_Work` | Nhị phân danh nghĩa | Điều kiện làm việc | Hình thức làm việc từ xa | `"Yes"`, `"No"` |
| 16 | `Job_Satisfaction` | Số nguyên (int64) | Điều kiện làm việc | Điểm mức độ hài lòng công việc | 3 – 9 (thang điểm nguyên); TB = 6,02 |
| 17 | `Productivity_Change_%` | Số thực (float64) | Tác động AI | Phần trăm thay đổi năng suất sau khi AI được áp dụng | −19,99% – +39,99%; TB = +9,79% |

##### 2.1.2.3.1 Đặc điểm Dữ liệu Tổng hợp và Lưu ý Phân tích

Tập dữ liệu này là dữ liệu **tổng hợp nhân tạo** (synthetically generated), do đó có một số đặc điểm phân tích khác biệt so với dữ liệu thực tế:

1. **Không có giá trị thiếu:** Tất cả 17 cột đều có 0 giá trị null — điều này hiếm gặp trong dữ liệu thực tế và là hệ quả của việc sinh dữ liệu tổng hợp. Mô hình học được trên dữ liệu này có thể không xử lý tốt các giá trị thiếu khi triển khai thực tế.

2. **Phân phối lớp cân bằng hoàn hảo:** Việc ba lớp `Job_Status` có tỷ lệ ~33,3% là kết quả thiết kế có chủ ý, không phản ánh phân phối thực tế trong thị trường lao động. Trong thực tế, tỷ lệ "Replaced" thường thấp hơn nhiều so với "Unchanged" hoặc "Modified" ở thời điểm hiện tại.

3. **Tương quan đặc trưng-nhãn có thể đơn giản hóa:** Do dữ liệu được sinh tổng hợp, mối quan hệ giữa các đặc trưng (đặc biệt `Automation_Risk`, `AI_Adoption_Level`) và nhãn `Job_Status` có thể đã được lập trình sẵn với quy tắc rõ ràng, dẫn đến độ chính xác mô hình cao hơn so với dữ liệu thực tế phức tạp hơn.

4. **`Salary_After_AI` và `Productivity_Change_%` là đặc trưng nhạy cảm:** Hai cột này về bản chất là **kết quả** (outcomes) của quá trình tác động AI, do đó có thể có tương quan cao với nhãn `Job_Status`. Khi xây dựng mô hình dự đoán, cần cân nhắc khả năng **data leakage** nếu sử dụng các đặc trưng này là thông tin chỉ có sau khi sự kiện xảy ra.

---

#### 2.1.2.4 Tổng quan Các Nghiên cứu Liên quan

Dự đoán và phân loại tác động của AI đối với trạng thái công việc là một lĩnh vực nghiên cứu liên ngành giữa kinh tế lao động, khoa học dữ liệu và chính sách công. Phần này tổng hợp các công trình tiêu biểu sử dụng phương pháp học máy để đánh giá rủi ro tự động hóa và tác động AI đến lực lượng lao động.

> **Lưu ý:** Tập dữ liệu AI Impact on Job Sector là dữ liệu tổng hợp, được công bố vào tháng 4 năm 2026. Tại thời điểm nghiên cứu, chưa có công trình học thuật đồng nghiệp bình duyệt nào trực tiếp sử dụng bộ dữ liệu này. Các nghiên cứu dưới đây được tổng hợp theo chủ đề liên quan để làm cơ sở phương pháp và đối sánh kết quả.

##### 2.1.2.4.1 Các nghiên cứu tiêu biểu

**Bảng 2.10. Tổng hợp các nghiên cứu liên quan về dự đoán rủi ro tự động hóa và tác động AI đến lao động**

| STT | Tác giả & Năm | Tạp chí / Hội nghị | Bối cảnh / Dữ liệu | Phương pháp chính | Kết quả chính | Ghi chú |
|:---:|:--------------|:-------------------|:-------------------|:------------------|:--------------|:--------|
| 1 | Frey & Osborne (2017) [1] | *Technological Forecasting and Social Change*, 114:254–280 | 702 nghề nghiệp tại Hoa Kỳ (O\*NET) | Gaussian Process Classifier; chuyên gia gán nhãn 70 nghề tham chiếu | ~47% số lao động Mỹ nằm trong vùng có rủi ro tự động hóa cao; 3 nhóm đặc trưng quan trọng: dexterity, creativity, social intelligence | Công trình nền tảng; phương pháp Gaussian Process + annotation được nhiều nghiên cứu sau kế thừa |
| 2 | Xu et al. (2022) [2] | *Proc. 18th Int. Conf. Advanced Data Mining and Applications (ADMA 2022)*, LNCS 13725. Springer | Dữ liệu O\*NET về đặc điểm nghề nghiệp (graph-based) | Graph Neural Network (GNN); so sánh với RF, LR, SVM | GNN vượt trội các mô hình ML truyền thống trong phân loại rủi ro tự động hóa; khai thác mối quan hệ giữa các nghề hiệu quả hơn | Tiếp cận graph-based cho dữ liệu nghề nghiệp; arXiv:2209.02182 |
| 3 | Balakumar et al. (2024) [3] | *2024 IEEE Silchar Subsection Conference (SILCON)* | Dữ liệu lao động; phân tích đa ngành | Gradient Boosting Machine (GBM); phân tích XAI (Explainable AI) | GBM đạt kết quả tốt nhất trong phân loại nguy cơ thay thế việc làm; XAI làm rõ đặc trưng quan trọng (ngành, trình độ học vấn, tự động hóa) | Kết hợp GBM + XAI cho bài toán AI-driven automation; 8 trích dẫn |
| 4 | Adesola et al. (2025) [4] | *2025 IEEE 5th International Conference* (IEEE Xplore: 11323513) | Dữ liệu rủi ro công việc tại London (đặc trưng lao động) | RF, Decision Tree, Logistic Regression, SVM (supervised ML) | **Random Forest: 97% accuracy** trong phân loại rủi ro công việc do AI | Nghiên cứu mới nhất (2025); RF vượt trội đáng kể so với các thuật toán so sánh; kết quả từ Google Scholar abstract |
| 5 | Park & Kim (2025) [5] | *Sustainable Cities and Society* (Elsevier), 2025 | Dữ liệu lao động khu vực tại Hoa Kỳ; Industry 4.0 | ML để tính xác suất tự động hóa; phân tích khu vực | Rủi ro tự động hóa phân phối không đồng đều theo khu vực địa lý; lao động thủ công dễ bị tổn thương nhất | Phân tích khu vực hóa rủi ro; có liên quan đến bài toán phân loại đa ngành |

##### 2.1.2.4.2 Phân tích Tổng hợp

Từ việc tổng hợp các công trình nghiên cứu, có thể rút ra một số nhận định chung phục vụ nghiên cứu của nhóm:

1. **Random Forest và Gradient Boosting là các thuật toán hiệu quả hàng đầu:** Cả hai nghiên cứu thực nghiệm (Balakumar et al., 2024; Adesola et al., 2025) đều xác nhận GBM và RF đạt hiệu năng cao nhất trong các bài toán phân loại tác động AI/tự động hóa đến lao động, phù hợp với xu hướng của bài toán phân loại dạng bảng nói chung.

2. **Đặc trưng quan trọng nhất tập trung vào nhóm "tác động AI":** Các nghiên cứu của Frey & Osborne (2017) và Balakumar et al. (2024) đều nhấn mạnh rằng các đặc trưng mô tả bản chất công việc (mức độ tự động hóa, yêu cầu nhận thức xã hội, dexterity) và môi trường triển khai AI là các yếu tố dự báo mạnh nhất. Trong bộ dữ liệu này, các cột `Automation_Risk` và `AI_Adoption_Level` nhiều khả năng đóng vai trò đặc trưng có tầm quan trọng cao nhất.

3. **Tiếp cận multi-class vs. binary:** Đa phần nghiên cứu hiện tại (Frey & Osborne, 2017; Xu et al., 2022) sử dụng phân loại nhị phân (nguy cơ cao/thấp) hoặc hồi quy xác suất. Phân loại 3 lớp (Unchanged/Modified/Replaced) như trong bộ dữ liệu này là một đặc điểm đáng chú ý, yêu cầu sử dụng các chỉ số đánh giá đa lớp (macro F1, weighted F1) thay vì binary metrics.

4. **Explainability ngày càng được coi trọng:** Xu hướng kết hợp ML + XAI (như Balakumar et al., 2024) cho thấy việc giải thích *tại sao* một công việc bị phân loại vào nhóm "Replaced" có giá trị thực tiễn cao, đặc biệt cho hoạch định chính sách đào tạo lại lao động.

5. **Hạn chế của dữ liệu tổng hợp:** Không có nghiên cứu nào trong danh sách sử dụng dữ liệu tổng hợp cân bằng hoàn hảo như bộ dữ liệu này. Kết quả mô hình trên dữ liệu tổng hợp có thể quá lạc quan so với thực tế — điều này cần được ghi nhận khi so sánh với các nghiên cứu sử dụng dữ liệu lao động thực tế.

---

#### 2.1.2.5 Các Chỉ số Đánh giá

Bài toán phân loại 3 lớp cân bằng với dữ liệu tổng hợp yêu cầu sử dụng các chỉ số đánh giá hỗ trợ đa lớp. Nhóm áp dụng các chỉ số sau:

**Bảng 2.11. Các chỉ số đánh giá mô hình phân loại đa lớp**

| Chỉ số | Ký hiệu | Công thức / Mô tả | Ý nghĩa |
|:-------|:-------:|:------------------|:--------|
| Độ chính xác tổng thể | Accuracy | $\frac{\text{Số dự đoán đúng}}{\text{Tổng số mẫu}}$ | Tỷ lệ phân loại đúng; đường cơ sở = 33,3% (ngẫu nhiên) |
| Precision đa lớp | Macro Precision | Trung bình đơn giản Precision của 3 lớp | Đánh giá độ chính xác dương tính trung bình, không phụ thuộc kích thước lớp |
| Recall đa lớp | Macro Recall | Trung bình đơn giản Recall của 3 lớp | Đánh giá khả năng phát hiện đúng mẫu của từng lớp, không phụ thuộc kích thước lớp |
| F1-Score đa lớp (Macro) | Macro F1 | $\text{Harmonic mean of Precision \& Recall per class, then averaged}$ | Chỉ số tổng hợp cân bằng; phù hợp khi kích thước các lớp bằng nhau |
| F1-Score có trọng số | Weighted F1 | Trung bình F1 từng lớp, trọng số = kích thước lớp | Ưu tiên lớp có nhiều mẫu hơn; tham chiếu khi lớp không cân bằng |
| Ma trận nhầm lẫn | Confusion Matrix | Ma trận $3 \times 3$ thể hiện TP, FP, FN từng lớp | Phân tích chi tiết lỗi phân loại giữa từng cặp lớp |

---

### Tài liệu Tham khảo

[1] Frey, C. B., & Osborne, M. A. (2017). The future of employment: How susceptible are jobs to computerisation? *Technological Forecasting and Social Change*, 114, 254–280. https://doi.org/10.1016/j.techfore.2016.08.019

[2] Xu, D., Yang, H., Rizoiu, M. A., & Xu, G. (2022). Being automated or not? Risk identification of occupations with graph neural networks. In *Proceedings of the 18th International Conference on Advanced Data Mining and Applications (ADMA 2022)*, Lecture Notes in Computer Science, vol. 13725 (pp. 489–504). Springer. https://doi.org/10.1007/978-3-031-22064-7_37  *(Preprint: https://arxiv.org/abs/2209.02182)*

[3] Balakumar, A., Sawant, P. D., Nimma, D., et al. (2024). Impact of AI-driven automation on job displacement and skill development: A societal perspective. In *2024 IEEE Silchar Subsection Conference (SILCON)*. IEEE. https://ieeexplore.ieee.org/document/10910660

[4] Adesola, A. E., Ojo, O. A., Aboutorabi, N., et al. (2025). Predicting job risk from artificial intelligence in London using supervised machine learning models. In *2025 IEEE 5th International Conference*. IEEE. https://ieeexplore.ieee.org/document/11323513

[5] Park, J., & Kim, D. (2025). The vulnerable future of manual jobs? Automation probabilities and regional patterns in the US under the Industry 4.0 transformation. *Sustainable Cities and Society* (Elsevier). https://www.sciencedirect.com/science/article/pii/S2210670725009084



# Ghi chú:
1. Dữ liệu tổng hợp → kết quả mô hình có thể lạc quan hơn thực tế
Khi dữ liệu được sinh nhân tạo bằng code, người tạo ra nó phải lập trình quy tắc để gán nhãn. Ví dụ: "Nếu Automation_Risk = High VÀ AI_Adoption_Level = High thì Job_Status = Replaced". Dù quy tắc này được làm mờ đi bằng randomness, mô hình ML vẫn có thể học được pattern này rất dễ dàng vì nó nhất quán và không có noise thực tế.

Trong dữ liệu thực tế, điều tương tự không xảy ra:

Một người làm kế toán (Automation_Risk = High) vẫn giữ được việc vì sếp quen biết
Một lập trình viên (AI_Adoption_Level = High) bị sa thải vì công ty phá sản — không liên quan AI
Dữ liệu thiếu, mâu thuẫn, lỗi nhập liệu, bias thu thập...
Hệ quả thực tế: Mô hình đạt 95% accuracy trên dataset tổng hợp này không có nghĩa nó sẽ đạt 95% khi triển khai trên dữ liệu nhân sự thật. Đây là lý do các benchmark paper thường phân biệt rõ "evaluated on synthetic data" vs "evaluated on real-world data".

2. Chưa có bài báo đồng nghiệp bình duyệt nào dùng trực tiếp dataset này
Đây đơn giản là vấn đề thời gian:

Dataset được upload lên Kaggle vào tháng 4/2026 — tức là chỉ khoảng 1 tháng trước thời điểm hiện tại (tháng 5/2026)
Quy trình công bố bài báo khoa học (submit → peer review → chỉnh sửa → accept → publish) thường mất 6 tháng đến 2 năm
Ngay cả khi có nhóm nào đó đang dùng dataset này để nghiên cứu ngay lúc nó ra mắt, bài báo của họ cũng chưa thể được đăng tạp chí
Hệ quả cho báo cáo: Phần "Nghiên cứu liên quan" của nhóm không thể tìm các paper chạy trực tiếp trên bộ này để đối sánh kết quả ở Chương 4. Thay vào đó, nhóm đối sánh theo phương pháp (RF của nhóm vs RF của Adesola et al. trên bài toán tương tự) và theo đặc trưng quan trọng (so sánh feature importance). Điều này hoàn toàn hợp lệ trong báo cáo khoa học — đây là cách xử lý chuẩn khi nghiên cứu trên dữ liệu mới.