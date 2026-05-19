## 2.1 Nhóm A: Dữ liệu dạng Bảng (Tabular Data)

### 2.1.1 Stroke Prediction Dataset

#### 2.1.1.1 Giới thiệu Tập dữ liệu

**Tên tập dữ liệu:** Stroke Prediction Dataset  
**Nguồn:** Kaggle — kukuroo3, [https://www.kaggle.com/datasets/kukuroo3/stoke-prediction-dataset](https://www.kaggle.com/datasets/kukuroo3/stoke-prediction-dataset)  
**Gốc dữ liệu:** Kế thừa từ tập dữ liệu gốc của fedesoriano — *"Stroke Prediction Dataset"*, Kaggle (2021)  
**Giấy phép:** CC0: Public Domain  
**Nhiệm vụ học máy:** Phân loại nhị phân (Binary Classification)  
**Nhãn mục tiêu:** `stroke` — 0: Không đột quỵ, 1: Có đột quỵ

Tập dữ liệu này được xây dựng nhằm hỗ trợ phát triển các mô hình học máy có khả năng dự đoán nguy cơ đột quỵ dựa trên các thông tin nhân khẩu học, tiền sử bệnh lý và lối sống của bệnh nhân. Đột quỵ là một trong những nguyên nhân hàng đầu gây tử vong và tàn tật lâu dài trên toàn cầu, do đó việc phát hiện sớm các cá nhân có nguy cơ cao là hết sức cần thiết cho công tác phòng ngừa và can thiệp kịp thời.

---

#### 2.1.1.2 Thống kê Số lượng Mẫu

Tập dữ liệu được cung cấp dưới dạng hai tập con tách biệt gồm tập huấn luyện và tập kiểm tra, không có tập xác nhận (validation) được định nghĩa sẵn. Bảng 2.1 trình bày thống kê số lượng mẫu chi tiết.

**Bảng 2.1. Thống kê số lượng mẫu**

| Tập con         | Số lượng mẫu | Số đặc trưng đầu vào | Biến mục tiêu (`stroke`) |
|:----------------|-------------:|---------------------:|:------------------------:|
| Tập huấn luyện (`train.csv`) | 1.137 | 11 | Có (0 / 1) |
| Tập kiểm tra (`test.csv`)    |   932 | 11 | Không có |
| **Tổng cộng**   | **2.069** | **11** | — |

> **Lưu ý:** Tập kiểm tra không chứa nhãn `stroke`, phù hợp với bài toán dự đoán cuối kỳ trên nền tảng Kaggle.

##### 2.1.1.2.1 Phân phối Lớp trong Tập Huấn luyện

Phân phối lớp mục tiêu trong tập huấn luyện cho thấy sự **mất cân bằng lớp nghiêm trọng** (class imbalance):

**Bảng 2.2. Phân phối lớp trong tập huấn luyện**

| Nhãn | Mô tả | Số lượng | Tỷ lệ |
|:----:|:------|--------:|------:|
| 0 | Không đột quỵ | 1.000 | 87,95% |
| 1 | Có đột quỵ | 137 | 12,05% |
| **Tổng** | | **1.137** | **100%** |

Tỷ lệ lớp thiểu số (stroke = 1) chỉ chiếm khoảng **12,05%**, tạo ra thách thức đáng kể khi huấn luyện mô hình phân loại. Nếu không áp dụng các kỹ thuật xử lý mất cân bằng, các mô hình có xu hướng thiên lệch về lớp đa số và bỏ sót các trường hợp đột quỵ thực sự. Đây là vấn đề đặc biệt nghiêm trọng trong bối cảnh y tế, vì bỏ sót trường hợp dương tính (False Negative cao) có thể dẫn đến hậu quả lâm sàng nghiêm trọng.

---

#### 2.1.1.3 Mô tả Đặc trưng

Tập dữ liệu gồm 11 đặc trưng đầu vào và 1 biến mục tiêu. Các đặc trưng bao gồm thông tin nhân khẩu học, tiền sử bệnh lý và đặc điểm lối sống của bệnh nhân. Bảng 2.3 mô tả chi tiết từng đặc trưng.

**Bảng 2.3. Mô tả đặc trưng của tập dữ liệu**

| STT | Tên đặc trưng | Kiểu dữ liệu | Mô tả | Giá trị / Khoảng giá trị |
|:---:|:--------------|:-------------|:------|:--------------------------|
| 1 | `id` | Định danh (integer) | Mã định danh duy nhất của bệnh nhân | Số nguyên dương, không sử dụng trong mô hình |
| 2 | `gender` | Phân loại danh nghĩa | Giới tính của bệnh nhân | `"Male"`, `"Female"`, `"Other"` |
| 3 | `age` | Số thực (float) | Tuổi của bệnh nhân (năm) | Liên tục; lưu ý: có 1 giá trị lỗi `"*82"` trong tập huấn luyện cần tiền xử lý |
| 4 | `hypertension` | Nhị phân (0/1) | Tiền sử tăng huyết áp | 0 = Không tăng huyết áp, 1 = Có tăng huyết áp |
| 5 | `heart_disease` | Nhị phân (0/1) | Tiền sử bệnh tim mạch | 0 = Không có bệnh tim, 1 = Có bệnh tim |
| 6 | `ever_married` | Nhị phân danh nghĩa | Tình trạng hôn nhân | `"Yes"` (đã từng kết hôn), `"No"` |
| 7 | `work_type` | Phân loại danh nghĩa | Loại hình công việc | `"Private"`, `"Self-employed"`, `"Govt_job"`, `"children"`, `"Never_worked"` |
| 8 | `Residence_type` | Nhị phân danh nghĩa | Khu vực cư trú | `"Urban"` (thành thị), `"Rural"` (nông thôn) |
| 9 | `avg_glucose_level` | Số thực (float) | Mức đường huyết trung bình trong máu (mg/dL) | Liên tục, dương |
| 10 | `bmi` | Số thực (float) | Chỉ số khối cơ thể — Body Mass Index (kg/m²) | Liên tục, dương; **52 giá trị bị thiếu** trong tập huấn luyện |
| 11 | `smoking_status` | Phân loại danh nghĩa | Tình trạng hút thuốc | `"never smoked"`, `"formerly smoked"`, `"smokes"`, `"Unknown"` |
| 12 | `stroke` | Nhị phân (0/1) — **Biến mục tiêu** | Bệnh nhân có bị đột quỵ hay không | 0 = Không đột quỵ, 1 = Có đột quỵ |

##### 2.1.1.3.1 Vấn đề Chất lượng Dữ liệu

Quá trình khảo sát sơ bộ (EDA) xác định các vấn đề chất lượng dữ liệu sau cần được xử lý trong giai đoạn tiền xử lý:

1. **Giá trị thiếu (`bmi`):** Cột `bmi` trong tập huấn luyện có **52 giá trị null** (chiếm ~4,6%). Cần áp dụng kỹ thuật điền khuyết (imputation), điển hình là điền bằng giá trị trung vị (median) hoặc trung bình (mean) theo nhóm.

2. **Giá trị lỗi (`age`):** Cột `age` trong tập huấn luyện có **1 giá trị bị lỗi dạng chuỗi** (`"*82"`), dẫn đến kiểu dữ liệu của cột này là `object` (chuỗi) thay vì số. Cần loại bỏ ký tự không hợp lệ và chuyển đổi kiểu.

3. **Giá trị "Unknown" trong `smoking_status`:** Nhãn `"Unknown"` không phải là một giá trị thực sự mà phản ánh sự thiếu thông tin. Có thể xử lý như một loại giá trị thiếu hoặc giữ nguyên như một danh mục riêng biệt.

4. **Không nhất quán kiểu dữ liệu:** Cột `age` có kiểu `object` trong tập huấn luyện nhưng `int64` trong tập kiểm tra — cần chuẩn hóa về `float64` sau khi làm sạch.

---

#### 2.1.1.4 Tổng quan Các Nghiên cứu Liên quan

Dự đoán đột quỵ sử dụng học máy là một lĩnh vực nghiên cứu đang phát triển mạnh, đặc biệt từ sau khi bộ dữ liệu công khai của fedesoriano (2021) được phát hành trên Kaggle và trở thành chuẩn mực benchmark cho nhiều công trình học thuật. Phần này tổng hợp các nghiên cứu tiêu biểu sử dụng cùng bộ dữ liệu hoặc bộ dữ liệu có cấu trúc tương đương, tập trung vào các thuật toán được áp dụng, phương pháp tiền xử lý và các chỉ số đánh giá hiệu năng.

##### 2.1.1.4.1 Các nghiên cứu tiêu biểu

**Bảng 2.4. Tổng hợp các nghiên cứu liên quan về dự đoán đột quỵ**

| STT | Tác giả & Năm | Tạp chí / Hội nghị | Bộ dữ liệu | Thuật toán chính | Tiền xử lý mất cân bằng | Độ chính xác tốt nhất | AUC / F1 tốt nhất | Ghi chú |
|:---:|:--------------|:-------------------|:-----------|:-----------------|:------------------------|:---------------------:|:-----------------:|:--------|
| 1 | Sailasya & Kumari (2021) [1] | *Int. J. Adv. Comput. Sci. Appl.* (IJACSA), 12(6):539–545 | Kaggle Stroke Dataset (fedesoriano) | RF (cải tiến), Decision Tree, Naive Bayes, SVM, LR | Không đề cập rõ | ~82% (RF cải tiến) | Không báo cáo | Bài báo được trích dẫn nhiều nhất (~331 lần); RF cải tiến vượt trội các mô hình cơ bản |
| 2 | Biswas et al. (2022) [2] | *Healthcare Analytics* (Elsevier) | Kaggle Stroke Dataset (fedesoriano) | SVM, RF, LR, KNN, Naive Bayes | Không đề cập rõ | SVM: tốt nhất | Không báo cáo | SVM đạt hiệu suất cao nhất; RF báo cáo 99,87% — kết quả có thể phụ thuộc vào cách chia dữ liệu |
| 3 | Guhdar, Melhum & Ibrahim (2023) [3] | *J. Technology & Informatics*, 4(2):41–47 | Kaggle Stroke Dataset | LR, SVM, RF | Không đề cập | LR: ~78% | Không báo cáo | Tập trung tối ưu hóa LR cho bài toán dự đoán đột quỵ |
| 4 | Chakraborty et al. (2024) [4] | *BMC Bioinformatics*, 25:329 | Kaggle Stroke Dataset | Stacking Ensemble (RF + DT + KNN), LR | Feature selection + tiền xử lý | **98,6%** | F-measure cao | Mô hình stacking ensemble đạt kết quả vượt trội; KNN làm meta-learner |
| 5 | Hassan et al. (2024) [5] | *Scientific Reports (Nature)*, 14:11498 | Kaggle Stroke Dataset | Naive Bayes, RF, SVM, LR, Ensemble | Không đề cập rõ | Naive Bayes: ~82% | Không báo cáo | Xác định các yếu tố nguy cơ quan trọng: tuổi, glucose, BMI, tăng huyết áp, hút thuốc |
| 6 | Moulaei et al. (2024) [6] | *Scientific Reports (Nature)*, 14:31392 | Kaggle Stroke Dataset | RF, LR, SVM, KNN, LSTM, FNN, CNN, GRU | SMOTE + tiền xử lý toàn diện | RF: tốt nhất (ML) | LSTM: Sensitivity 96,15%; FNN: F1 cao nhất | So sánh toàn diện 4 mô hình ML + 4 mô hình DL; phân tích giải thích bằng SHAP/XAI |
| 7 | Akinwumi et al. (2025) [7] | *Frontiers in Neurology*, 16:1668420 | Kaggle Stroke Dataset (fedesoriano, 5.110 bản ghi) | LR, RF, Gradient Boosting, SVM, KNN | Random Oversampling, 5-fold CV | LR: 95,11%; SVM: 95,12% | LR: AUC=0,8378; GB: AUC=0,8236 | Nghiên cứu đánh giá toàn diện nhất; mặc dù accuracy cao nhưng hầu hết mô hình có True Positive = 0 |

##### 2.1.1.4.2 Kết quả Chi tiết — Akinwumi et al. (2025)

Do bài báo Akinwumi et al. (2025) [7] sử dụng cùng bộ dữ liệu Kaggle và công bố kết quả chi tiết nhất với phương pháp minh bạch, nhóm trình bày bảng hiệu năng đầy đủ dưới đây để tham chiếu so sánh.

**Bảng 2.5. Hiệu năng các mô hình ML — Akinwumi et al. (2025)**
*(Đánh giá bằng 5-fold stratified cross-validation, sau khi áp dụng Random Oversampling)*

| Mô hình | Accuracy | ROC-AUC | TP | TN | FP | FN |
|:--------|:--------:|:-------:|:--:|:--:|:--:|:--:|
| Logistic Regression (LR) | **95,11%** | **0,8378** | 0 | 972 | 0 | 50 |
| Support Vector Machine (SVM) | 95,12% | 0,6152 | 0 | 972 | 0 | 50 |
| Random Forest (RF) | 94,95% | 0,7923 | 0 | 970 | 2 | 50 |
| Gradient Boosting (GB) | 94,90% | 0,8236 | 1 | 968 | 4 | 49 |
| K-Nearest Neighbors (KNN) | 94,18% | 0,6138 | 1 | 969 | 3 | 49 |

**Nhận xét:** Mặc dù tất cả các mô hình đạt độ chính xác cao (>94%), nhưng Confusion Matrix cho thấy hầu hết mô hình không nhận diện được bất kỳ trường hợp đột quỵ thực sự nào (TP = 0), ngoại trừ Gradient Boosting và KNN với TP = 1. Điều này chứng tỏ *accuracy không phải là chỉ số đánh giá phù hợp* cho bài toán dữ liệu mất cân bằng; các chỉ số Recall (Sensitivity), F1-Score và ROC-AUC cần được ưu tiên.

**Phân tích tầm quan trọng đặc trưng (Random Forest — Akinwumi et al., 2025):**

Thứ tự tầm quan trọng: **Tuổi** > **Mức đường huyết trung bình** > **BMI** > **Tình trạng hút thuốc** > **Loại hình công việc**

##### 2.1.1.4.3 Phân tích Tổng hợp

Từ việc tổng hợp các công trình nghiên cứu, có thể rút ra một số nhận định chung:

1. **Mất cân bằng lớp là thách thức trung tâm:** Tất cả các nghiên cứu đều ghi nhận sự mất cân bằng nghiêm trọng giữa lớp có đột quỵ (thiểu số ~4,9–12%) và không đột quỵ (đa số). Các kỹ thuật xử lý phổ biến bao gồm SMOTE, Random Oversampling và Cost-Sensitive Learning.

2. **Random Forest và Ensemble Methods vượt trội:** Theo tổng hợp của Asadi et al. (2024), RF là thuật toán hiệu quả nhất trong 25% các nghiên cứu về đột quỵ từ 2019–2023. Các mô hình stacking và boosting (XGBoost, Gradient Boosting) cũng cho kết quả cạnh tranh.

3. **Recall/Sensitivity cần được tối ưu ưu tiên hơn Accuracy:** Trong bài toán y tế, chi phí của False Negative (bỏ sót ca bệnh) cao hơn False Positive, do đó Recall và F1-Score là các chỉ số đánh giá quan trọng hơn Accuracy đơn thuần.

4. **Các yếu tố nguy cơ nhất quán:** Qua nhiều nghiên cứu, **tuổi**, **mức đường huyết trung bình** và **BMI** liên tục được xác định là ba đặc trưng dự đoán quan trọng nhất, phù hợp với các mô hình lâm sàng truyền thống như Framingham Stroke Risk Score.

5. **Deep Learning không nhất thiết vượt trội ML truyền thống:** Nghiên cứu của Moulaei et al. (2024) cho thấy RF vẫn là mô hình ML tốt nhất tổng thể, trong khi LSTM đạt Sensitivity cao hơn (96,15%), phản ánh sự đánh đổi giữa hiệu năng tổng thể và khả năng phát hiện ca dương tính.

---

#### 2.1.1.5 Các Chỉ số Đánh giá

Dựa trên thực tiễn nghiên cứu và đặc điểm bài toán (dữ liệu mất cân bằng, bài toán phân loại nhị phân y tế), nhóm sử dụng các chỉ số đánh giá sau:

**Bảng 2.6. Các chỉ số đánh giá mô hình phân loại**

| Chỉ số | Ký hiệu | Công thức | Ý nghĩa |
|:-------|:-------:|:----------|:--------|
| Độ chính xác tổng thể | Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | Tỷ lệ dự đoán đúng trên tổng mẫu |
| Độ nhạy / Recall | Sensitivity | $\frac{TP}{TP + FN}$ | Tỷ lệ phát hiện đúng ca đột quỵ thực sự — quan trọng trong bài toán y tế |
| Độ đặc hiệu | Specificity | $\frac{TN}{TN + FP}$ | Tỷ lệ phân loại đúng ca không đột quỵ |
| Độ chính xác dương tính | Precision | $\frac{TP}{TP + FP}$ | Tỷ lệ dự đoán dương tính thực sự chính xác |
| F1-Score | F1 | $\frac{2 \times Precision \times Recall}{Precision + Recall}$ | Trung bình điều hòa của Precision và Recall — phù hợp cho dữ liệu mất cân bằng |
| Diện tích dưới đường ROC | ROC-AUC | — | Khả năng phân biệt hai lớp tại mọi ngưỡng quyết định; 0,5 = ngẫu nhiên, 1,0 = hoàn hảo |

---

### Tài liệu Tham khảo

[1] Sailasya, G., & Kumari, G. L. A. (2021). Analyzing the performance of stroke prediction using ML classification algorithms. *International Journal of Advanced Computer Science and Applications (IJACSA)*, 12(6), 539–545. https://doi.org/10.14569/IJACSA.2021.0120662

[2] Biswas, M., et al. (2022). A comparative analysis of machine learning classifiers for stroke prediction: A predictive analytics approach. *Healthcare Analytics* (Elsevier). https://doi.org/10.1016/S2772442522000569

[3] Guhdar, M., Melhum, A. I., & Ibrahim, A. L. (2023). Optimizing accuracy of stroke prediction using logistic regression. *Journal of Technology and Informatics*, 4(2), 41–47. https://doi.org/10.37802/joti.v4i2.278

[4] Chakraborty, P., Bandyopadhyay, A., Sahu, P. P., Burman, A., Mallik, S., Alsubaie, N., et al. (2024). Predicting stroke occurrences: a stacked machine learning approach with feature selection and data preprocessing. *BMC Bioinformatics*, 25, 329. https://doi.org/10.1186/s12859-024-05866-8

[5] Hassan, A., Gulzar Ahmad, S., Ullah Munir, E., Ali Khan, I., & Ramzan, N. (2024). Predictive modelling and identification of key risk factors for stroke using machine learning. *Scientific Reports*, 14, 11498. https://doi.org/10.1038/s41598-024-61665-4

[6] Moulaei, K., Afshari, L., Moulaei, R., Sabet, B., Mousavi, S. M., & Afrash, M. R. (2024). Explainable artificial intelligence for stroke prediction through comparison of deep learning and machine learning models. *Scientific Reports*, 14, 31392. https://doi.org/10.1038/s41598-024-82931-5

[7] Akinwumi, P. O., Ojo, S., Nathaniel, T. I., Wanliss, J., Karunwi, O., & Sulaiman, M. (2025). Evaluating machine learning models for stroke prediction based on clinical variables. *Frontiers in Neurology*, 16, 1668420. https://doi.org/10.3389/fneur.2025.1668420
