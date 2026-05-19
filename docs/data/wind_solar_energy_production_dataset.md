### 2.2.2 Wind & Solar Energy Production Dataset

---

#### 2.2.2.1 Giới thiệu Tập dữ liệu

**Wind & Solar Energy Production Dataset** là tập dữ liệu chuỗi thời gian tần suất cao, ghi lại sản lượng điện tái tạo theo từng giờ tại **Pháp** từ ngày 01/01/2020 đến ngày 30/11/2025. Dữ liệu gốc được thu thập từ **Open Data Réseaux Énergies (ODRÉ)** — cổng dữ liệu mở chính thức của ngành năng lượng Pháp — dưới giấy phép **Licence Ouverte v2.0 (Open Licence 2.0)**. Phiên bản Kaggle được biên soạn và bổ sung đặc trưng bởi Ahmed Mohamed Zaki, phát hành theo giấy phép **CC BY 4.0** (Attribution 4.0 International), cho phép sử dụng tự do trong nghiên cứu học thuật và thương mại khi trích dẫn nguồn.

Dữ liệu gốc từ ODRÉ bao gồm 4 cột: `Date`, `Heure` (giờ), `prod_eolienne_MWh` (sản lượng điện gió) và `prod_solaire_MWh` (sản lượng điện mặt trời). Tác giả Kaggle đã **kỹ nghệ hóa thêm 5 đặc trưng thời gian** (`Day_of_Year`, `Day_Name`, `Month_Name`, `Season`, `Source`), đồng thời tổng hợp sản lượng hai nguồn thành cột `Production` (MWh). Cột `Source` phân loại mỗi giờ đo theo nguồn năng lượng chiếm ưu thế: `Wind` nếu sản lượng điện gió > điện mặt trời, `Solar` nếu ngược lại, `Mixed` nếu xấp xỉ bằng nhau.

**Phạm vi ứng dụng trong dự án**: Tập dữ liệu này được sử dụng cho bài toán **hồi quy chuỗi thời gian** — dự báo sản lượng điện tái tạo theo từng giờ trong tương lai dựa trên các mẫu lịch sử. Ba thuật toán được triển khai là **Linear Regression**, **ARIMA** và **XGBoost**, tương ứng với cách tiếp cận hồi quy tuyến tính cơ sở, mô hình chuỗi thời gian thống kê và học máy dựa trên cây quyết định tăng cường.

| Thuộc tính | Thông tin |
|---|---|
| **Tên dataset** | Wind & Solar Energy Production Dataset |
| **Nguồn Kaggle** | [https://www.kaggle.com/datasets/ahmeduzaki/wind-and-solar-energy-production-dataset](https://www.kaggle.com/datasets/ahmeduzaki/wind-and-solar-energy-production-dataset) |
| **Dữ liệu gốc** | Open Data Réseaux Énergies (ODRÉ) — Pháp |
| **URL gốc** | [https://odre.opendatasoft.com/explore/dataset/courbes-de-production-mensuelles-eolien-solaire-complement-de-remuneration/](https://odre.opendatasoft.com/explore/dataset/courbes-de-production-mensuelles-eolien-solaire-complement-de-remuneration/) |
| **Giấy phép** | CC BY 4.0 (Kaggle) / Licence Ouverte v2.0 (ODRÉ) |
| **Usability (Kaggle)** | 10.00 / 10 |
| **Tác giả** | Ahmed Mohamed Zaki |
| **Loại dữ liệu** | Chuỗi thời gian — đo lường thực tế (real-world) |
| **Bài toán** | Hồi quy chuỗi thời gian (Time Series Regression) |

---

#### 2.2.2.2 Thống kê Số lượng Mẫu

Tập dữ liệu gồm **một tệp CSV duy nhất** (`Energy Production Dataset.csv`) với cấu trúc như sau:

**Bảng 2.22 — Thống kê cơ bản tập dữ liệu Wind & Solar Energy Production**

| Thuộc tính | Giá trị |
|---|---|
| **Số lượng mẫu (hàng)** | 51,864 |
| **Số đặc trưng (cột)** | 9 |
| **Số tệp CSV** | 1 (`Energy Production Dataset.csv`) |
| **Khoảng thời gian** | 01/01/2020 – 30/11/2025 (5 năm 11 tháng) |
| **Tần suất đo lường** | Theo giờ (hourly) — 24 bản ghi/ngày |
| **Số ngày duy nhất** | 2,161 ngày |
| **Số lượng giá trị thiếu** | 0 (hoàn chỉnh 100%) |
| **Biến mục tiêu** | `Production` (MWh) |
| **Kiểu bài toán** | Hồi quy chuỗi thời gian |

> **Lưu ý:** Tập dữ liệu đã ở tần suất giờ — không cần tổng hợp (resample) trước khi huấn luyện mô hình, khác với Superstore Sales Dataset. Để đưa vào ARIMA, cần sắp xếp theo `Date` + `Start_Hour` để tạo chuỗi thời gian liên tục.

---

##### 2.2.2.2.1 Phân phối Theo Nguồn, Mùa và Sản lượng

**Bảng 2.23 — Phân phối mẫu và sản lượng theo Nguồn năng lượng và Mùa**

| Nhóm phân loại | Giá trị | Số mẫu | Tỷ lệ (%) | Sản lượng TB (MWh) | Sản lượng Min (MWh) | Sản lượng Max (MWh) |
|---|---|---|---|---|---|---|
| **Nguồn (Source)** | Wind | 42,484 | 81.93% | 6,308.26 | 58 | 23,446 |
| | Solar | 9,378 | 18.09% | 5,793.85 | 267 | 16,578 |
| | Mixed | 2 | 0.004% | 1,737.00 | 734 | 2,740 |
| **Mùa (Season)** | Winter (Đông) | 12,264 | 23.65% | 7,341.79 | 166 | 22,634 |
| | Spring (Xuân) | 13,242 | 25.54% | 6,425.50 | 274 | 23,264 |
| | Fall (Thu) | 13,110 | 25.28% | 6,266.11 | 58 | 23,446 |
| | Summer (Hạ) | 13,248 | 25.55% | 4,911.19 | 140 | 20,606 |

**Thống kê mô tả cột `Production` (MWh):**

| Chỉ số thống kê | Giá trị (MWh) |
|---|---|
| Min | 58 |
| Q1 (25%) | 3,111 |
| Trung vị (Q2) | 5,372 |
| Trung bình (mean) | 6,215.07 |
| Q3 (75%) | 8,501 |
| Max | 23,446 |
| Độ lệch chuẩn | 3,978.37 |

> **Lưu ý giải thích:** Điện gió chiếm ưu thế hoàn toàn (81.9%) do Pháp có tiềm năng gió mạnh quanh năm. Mùa Đông có sản lượng cao nhất (trung bình 7,341.79 MWh) vì tốc độ gió cao, trong khi Mùa Hạ thấp nhất (4,911.19 MWh) dù điện mặt trời đạt đỉnh vào mùa này — điều này phản ánh thực tế rằng tổng sản lượng bị chi phối bởi điện gió. `Production` có phân phối lệch phải (mean 6,215 > median 5,372), cho thấy tồn tại các giờ sản lượng rất cao (max 23,446 MWh).

---

#### 2.2.2.3 Mô tả Đặc trưng

**Bảng 2.24 — Mô tả chi tiết các đặc trưng tập dữ liệu Wind & Solar Energy Production**

| STT | Tên cột | Kiểu dữ liệu | Nhóm | Mô tả | Giá trị / Phạm vi |
|---|---|---|---|---|---|
| 1 | `Date` | `object` (str) | Thời gian | Ngày đo lường, định dạng M/D/YYYY trong tệp CSV | 1/1/2020 – 11/30/2025; 2,161 ngày duy nhất |
| 2 | `Start_Hour` | `int64` | Thời gian | Giờ bắt đầu khoảng đo (0–23) | 0–23; mỗi giá trị xuất hiện đúng 2,161 lần |
| 3 | `End_Hour` | `int64` | Thời gian | Giờ kết thúc (= Start_Hour + 1); khi Start_Hour=23 thì End_Hour=0 | 0–23 |
| 4 | `Source` | `object` (str) | Phân loại nguồn | Nguồn năng lượng chiếm ưu thế trong giờ đó (kỹ nghệ hóa từ dữ liệu gốc) | Wind (42,484), Solar (9,378), Mixed (2) |
| 5 | `Day_of_Year` | `int64` | Đặc trưng tuần hoàn | Thứ tự ngày trong năm (1–366); mã hóa chu kỳ năm | 1–366 |
| 6 | `Day_Name` | `object` (str) | Đặc trưng thời gian | Tên ngày trong tuần (tiếng Anh) | Monday–Sunday; ~7,392–7,416 mẫu/ngày |
| 7 | `Month_Name` | `object` (str) | Đặc trưng thời gian | Tên tháng trong năm (tiếng Anh) | January–December; 3,720–4,470 mẫu/tháng |
| 8 | `Season` | `object` (str) | Đặc trưng thời gian | Mùa trong năm | Winter (12,264), Spring (13,242), Summer (13,248), Fall (13,110) |
| 9 | `Production` | `int64` | **Biến mục tiêu** | Tổng sản lượng điện tái tạo trong giờ đó (MWh) | 58–23,446 MWh; mean=6,215.07 |

---

##### 2.2.2.3.1 Đặc điểm Kỹ thuật và Lưu ý Phân tích

**1. Nguồn gốc đặc trưng — dữ liệu kỹ nghệ hóa:**
Dữ liệu gốc ODRÉ có 4 cột (`Date`, `Heure`, `prod_eolienne_MWh`, `prod_solaire_MWh`). Phiên bản Kaggle đã kỹ nghệ hóa thêm 5 đặc trưng (`Day_of_Year`, `Day_Name`, `Month_Name`, `Season`, `Source`) và tổng hợp sản lượng hai nguồn thành cột `Production` duy nhất. **Lưu ý:** phiên bản Kaggle không còn lưu riêng `prod_eolienne_MWh` và `prod_solaire_MWh`, nên không thể phân tách sản lượng gió và mặt trời trong phân tích.

**2. Ý nghĩa cột `Source`:**
`Source` là đặc trưng phân loại được kỹ nghệ hóa — gán nhãn `Wind` khi sản lượng điện gió > điện mặt trời cho giờ đó, `Solar` khi ngược lại. Chỉ có **2 bản ghi `Mixed`** (chiếm 0.004%), phản ánh trường hợp hai nguồn có sản lượng xấp xỉ nhau — thực tế rất hiếm. `Source` có thể dùng như đặc trưng đầu vào cho XGBoost (sau one-hot encoding).

**3. Tính chu kỳ và tính mùa vụ:**
Cột `Day_of_Year`, `Day_Name`, `Month_Name`, và `Season` **đã được kỹ nghệ hóa sẵn** — XGBoost có thể sử dụng trực tiếp các đặc trưng này như đầu vào mà không cần bước feature engineering bổ sung. ARIMA sử dụng cột `Production` như chuỗi thời gian đơn biến (sau khi sắp xếp theo `Date` + `Start_Hour`).

**4. Chu kỳ hàng ngày (diurnal cycle):**
`Start_Hour` là đặc trưng quan trọng: điện mặt trời chỉ sản xuất ban ngày (giờ 6–18), trong khi điện gió sản xuất cả ngày lẫn đêm. Cột `Start_Hour` có mối tương quan mạnh với `Source` và `Production` — cần giữ lại trong tập đặc trưng cho XGBoost.

**5. Xử lý cột `Date`:**
Cột `Date` lưu dạng chuỗi M/D/YYYY trong tệp CSV (ví dụ: `11/30/2025`). Trước khi đưa vào mô hình, cần chuyển đổi sang kiểu `datetime` bằng `pd.to_datetime(df['Date'], format='mixed')` để đảm bảo sắp xếp đúng thứ tự thời gian.

**6. Không có giá trị thiếu:**
Tập dữ liệu hoàn chỉnh 100% (0 null trên toàn bộ 51,864 × 9 = 466,776 ô dữ liệu) — không cần bước imputation.

---

#### 2.2.2.4 Tổng quan Nghiên cứu Liên quan

**Bảng 2.25 — Các nghiên cứu liên quan đến dự báo sản lượng điện gió và điện mặt trời**

| STT | Tác giả & Năm | Tạp chí / Hội nghị | Bối cảnh nghiên cứu | Phương pháp chính | Kết quả tiêu biểu | Ghi chú |
|---|---|---|---|---|---|---|
| 1 | ÖrenÇ et al. (2024) | *2024 Global Energy Conference*, IEEE | Dự báo sản lượng điện gió và mặt trời từ **dữ liệu giờ tại Pháp (2020)** — cùng nguồn ODRÉ với dataset này | Regression: XGBoost, Random Forest, Linear Regression | XGBoost mô hình hóa hiệu quả các mẫu phi tuyến trong dữ liệu gió và mặt trời | ★ Dùng **cùng nguồn dữ liệu** (Pháp, 2020) |
| 2 | Lindas et al. (2025) | *Environmental Data Science*, Cambridge Univ. Press | Xây dựng bộ dữ liệu và benchmark mô hình ML cho **dự báo điện gió & mặt trời tại Pháp** — quy mô quốc gia | XGBoost, LASSO, Random Forest, Neural Network | Benchmark toàn diện các mô hình ML trên dữ liệu Pháp; XGBoost và LASSO cho kết quả cạnh tranh | ★ **Pháp**, cùng loại bài toán |
| 3 | Krechowicz et al. (2022) | *Energies* (MDPI); 79 trích dẫn | Dự báo sản lượng điện từ **nhiều nguồn tái tạo** (gió, mặt trời, thủy điện) | XGBoost, ANN, SVM, ARIMA | **XGBoost đạt R² = 0.9997** — vượt trội ANN, SVM và ARIMA; phù hợp nhất cho dự báo sản lượng điện | Nghiên cứu cơ sở cho XGBoost |
| 4 | Benrabia & Söffker (2025) | *Energies* (MDPI); 3 trích dẫn | Mô hình hóa và đánh giá các mô hình dự báo sản lượng điện **gió và quang điện PV** theo giờ | ARIMA, SARIMA, Prophet, LSTM | ARIMA phù hợp cho chuỗi tuyến tính; điện mặt trời có độ biến động lớn hơn điện gió | Phù hợp cho nhánh **ARIMA** |
| 5 | Abdelsalam et al. (2025) | *Energy Storage and Saving*, Elsevier; 3 trích dẫn | Dự báo sản lượng hệ thống **điện mặt trời IoT** — so sánh gradient boosting với mô hình chuỗi thời gian | XGBoost/Gradient Boosting, ARIMA, Linear Regression | Gradient boosting vượt trội ARIMA và linear regression trên dữ liệu sản lượng điện mặt trời | So sánh trực tiếp **3 thuật toán** dự án |
| 6 | Teixeira et al. (2024) | *Energies* (MDPI); 72 trích dẫn | Tổng quan toàn diện các phương pháp dự báo **điện tái tạo** (gió và mặt trời) | Review: ARIMA, SARIMA, XGBoost, LSTM, Transformer | ARIMA vẫn là cơ sở vững chắc; XGBoost nổi trội cho dữ liệu tần số cao; học sâu tốt cho dự báo ngắn hạn | Review 72 trích dẫn |
| 7 | Bülüç et al. (2025) | *Gazi University Journal of Science Part A* | Phân tích chuỗi thời gian **sản lượng điện mặt trời** theo điều kiện thời tiết | ARIMA, LSTM, FB-Prophet, NNAR, ELM, Linear Regression | ARIMA đạt kết quả tốt cho chuỗi có tính mùa vụ rõ ràng; so sánh trực tiếp với linear regression và học sâu | So sánh rõ ràng **ARIMA vs Linear Regression** |

---

##### 2.2.2.4.1 Nghiên cứu Tiêu biểu

**Nghiên cứu trực tiếp liên quan nhất** là **ÖrenÇ et al. (2024)** — nghiên cứu này sử dụng dữ liệu giờ về sản lượng điện gió và mặt trời từ **Pháp (từ năm 2020)**, có nguồn gốc từ cùng hệ thống ODRÉ là nguồn của dataset này. Nghiên cứu áp dụng các mô hình hồi quy bao gồm XGBoost, đồng thời đánh giá khả năng mô hình hóa các mẫu phi tuyến trong dữ liệu năng lượng tái tạo.

**Lindas et al. (2025)** từ *Environmental Data Science* (Cambridge University Press) cung cấp bộ benchmark toàn diện nhất cho bài toán dự báo điện gió và mặt trời tại Pháp ở quy mô quốc gia, bao gồm cả quy trình xây dựng tập dữ liệu và so sánh nhiều mô hình ML. Nghiên cứu này có bối cảnh địa lý và loại bài toán **gần nhất với tập dữ liệu hiện tại**.

**Krechowicz et al. (2022)** là nghiên cứu được trích dẫn nhiều nhất (79 lần), cung cấp bằng chứng định lượng mạnh mẽ nhất cho việc lựa chọn XGBoost: **R² = 0.9997** cho bài toán dự báo sản lượng điện từ các nguồn tái tạo, vượt trội đáng kể so với ANN, SVM và ARIMA truyền thống.

---

##### 2.2.2.4.2 Phân tích Tổng hợp

Tổng hợp các nghiên cứu liên quan cho thấy ba xu hướng chính trong dự báo sản lượng điện tái tạo:

**① Về XGBoost:** Được xác nhận là phương pháp hiệu quả nhất cho dự báo sản lượng điện từ tái tạo trong nhiều bối cảnh — từ dữ liệu giờ (ÖrenÇ et al., 2024) đến dữ liệu đa nguồn (Krechowicz et al., 2022) và hệ thống IoT (Abdelsalam et al., 2025). XGBoost đặc biệt phù hợp với tập dữ liệu này vì **5 đặc trưng thời gian đã được kỹ nghệ hóa sẵn** (`Start_Hour`, `Day_of_Year`, `Day_Name`, `Month_Name`, `Season`) phục vụ trực tiếp làm đầu vào.

**② Về ARIMA:** Được công nhận là phương pháp cơ sở (baseline) vững chắc — đặc biệt hiệu quả cho các chuỗi có tính mùa vụ rõ ràng và xu hướng tuyến tính (Benrabia & Söffker, 2025; Bülüç et al., 2025). Với dataset Wind & Solar, tính mùa vụ mạnh (Winter/Summer chênh lệch ~50% sản lượng trung bình) là điều kiện thuận lợi cho ARIMA. Tuy nhiên ARIMA chỉ sử dụng cột `Production` như chuỗi đơn biến, bỏ qua các đặc trưng phân loại.

**③ Về Linear Regression:** Đóng vai trò là mô hình cơ sở (baseline) đơn giản nhất — được sử dụng trong các nghiên cứu so sánh (Abdelsalam et al., 2025; Bülüç et al., 2025) để thiết lập ngưỡng hiệu suất tối thiểu. Linear regression không phù hợp cho các mối quan hệ phi tuyến trong dữ liệu năng lượng (biến động theo giờ, mùa) nhưng hữu ích để đánh giá mức độ cải tiến của ARIMA và XGBoost.

---

#### 2.2.2.5 Các Chỉ số Đánh giá

Bài toán hồi quy chuỗi thời gian — dự báo sản lượng điện tái tạo (MWh) — được đánh giá bằng các chỉ số hồi quy chuẩn. Do biến mục tiêu `Production` có đơn vị MWh và phạm vi rộng (58–23,446 MWh), các chỉ số tỷ lệ phần trăm (MAPE, SMAPE) bổ sung thêm góc nhìn đánh giá tương đối.

**Bảng 2.26 — Các chỉ số đánh giá mô hình hồi quy cho Wind & Solar Energy Production Dataset**

| STT | Chỉ số | Ký hiệu | Công thức | Ý nghĩa | Đặc điểm với dataset này |
|---|---|---|---|---|---|
| 1 | Trung bình sai số tuyệt đối | MAE | $\dfrac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$ | Sai số trung bình tuyệt đối (đơn vị MWh) | Dễ diễn giải; không nhạy với outlier; cùng đơn vị với Production |
| 2 | Căn bậc hai trung bình bình phương sai số | RMSE | $\sqrt{\dfrac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | Phạt mạnh các sai số lớn (đơn vị MWh) | Nhạy hơn MAE với các giờ sản lượng đột biến (max 23,446 MWh); là chỉ số chính |
| 3 | Phần trăm sai số tuyệt đối trung bình | MAPE | $\dfrac{100\%}{n}\sum_{i=1}^{n}\left\|\dfrac{y_i - \hat{y}_i}{y_i}\right\|$ | Sai số tương đối (%) — dễ so sánh giữa các mô hình | Cần lưu ý khi $y_i$ nhỏ (min=58 MWh) có thể gây MAPE phóng đại |
| 4 | Phần trăm sai số tuyệt đối đối xứng | SMAPE | $\dfrac{100\%}{n}\sum_{i=1}^{n}\dfrac{2\|y_i - \hat{y}_i\|}{|y_i| + |\hat{y}_i|}$ | Sai số đối xứng (%) — khắc phục nhược điểm MAPE khi $y$ nhỏ | Phù hợp hơn MAPE khi sản lượng có phạm vi rộng (58–23,446 MWh) |
| 5 | Hệ số xác định | R² | $1 - \dfrac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Tỷ lệ phương sai được giải thích bởi mô hình (0–1) | R² gần 1 = mô hình tốt; Krechowicz et al. (2022) đạt R²=0.9997 với XGBoost |

> **Lưu ý tính MAPE:** Một số bản ghi có `Production` rất nhỏ (min=58 MWh, đặc biệt là các giờ ban đêm khi chỉ có điện gió yếu). Khi giá trị thực tiến về 0, MAPE có thể đạt giá trị vô cùng lớn. Trong trường hợp này, SMAPE là chỉ số tỷ lệ phần trăm phù hợp hơn. Có thể bổ sung bộ lọc loại bỏ các mẫu có `Production < 100 MWh` khi tính MAPE.

---

### Tài liệu Tham khảo

[1] ÖrenÇ, S., Acar, M. S., Özerdem, B. et al. (2024). "Prediction of Electricity Production from Wind and Solar Energy by Employing Regression Models." *2024 Global Energy Conference (GEC)*, IEEE. DOI: [10.1109/GEC64401.2024.10881522](https://ieeexplore.ieee.org/abstract/document/10881522/)

[2] Lindas, N., Goude, Y., & Ciais, P. (2025). "Toward accurate forecasting of renewable energy: Building datasets and benchmarking machine learning models for solar and wind power in France." *Environmental Data Science*, Cambridge University Press. URL: [https://www.cambridge.org/core/journals/environmental-data-science/article/toward-accurate-forecasting-of-renewable-energy-building-datasets-and-benchmarking-machine-learning-models-for-solar-and-wind-power-in-france/F3B5DDBC5A9BE156AA15790CB7406403](https://www.cambridge.org/core/journals/environmental-data-science/article/toward-accurate-forecasting-of-renewable-energy-building-datasets-and-benchmarking-machine-learning-models-for-solar-and-wind-power-in-france/F3B5DDBC5A9BE156AA15790CB7406403)

[3] Krechowicz, A., Krechowicz, M., & Poczeta, K. (2022). "Machine learning approaches to predict electricity production from renewable energy sources." *Energies*, 15(23), 9146, MDPI. DOI: [10.3390/en15239146](https://www.mdpi.com/1996-1073/15/23/9146)

[4] Benrabia, Y., & Söffker, D. (2025). "Modeling and Evaluation of Forecasting Models for Energy Production in Wind and Photovoltaic Systems." *Energies*, 18(3), 625, MDPI. DOI: [10.3390/en18030625](https://www.mdpi.com/1996-1073/18/3/625)

[5] Abdelsalam, A., Souri, I., & İnanç, N. (2025). "A time series forecasting approach based on gradient boosting method for IoT-based solar energy production systems." *Energy Storage and Saving*, Elsevier. DOI: [10.1016/j.enss.2025.01.009](https://www.sciencedirect.com/science/article/pii/S2772683525000366)

[6] Teixeira, R., Cerveira, A., Pires, E. J. S., & Baptista, J. (2024). "Advancing renewable energy forecasting: A comprehensive review of renewable energy forecasting methods." *Energies*, 17(14), 3480, MDPI. DOI: [10.3390/en17143480](https://www.mdpi.com/1996-1073/17/14/3480)

[7] Bülüç, M., Sevli, O., & Yünlü, L. (2025). "Time Series Analysis of Solar Energy Production Based on Weather Conditions." *Gazi University Journal of Science Part A: Engineering and Innovation*, 2025. URL: [https://dergipark.org.tr/en/pub/gujsa/article/1797659](https://dergipark.org.tr/en/pub/gujsa/article/1797659)
