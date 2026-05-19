## 2.2 Nhóm B: Dữ liệu dạng Chuỗi thời gian (Time-Series Data)

### 2.2.1 Superstore Sales Dataset

#### 2.2.1.1 Giới thiệu Tập dữ liệu

**Tên tập dữ liệu:** Superstore Sales Dataset  
**Nguồn:** Kaggle — Rohit Sahoo, [https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)  
**Gốc dữ liệu:** Kế thừa từ bộ dữ liệu mẫu Sample Superstore của Tableau (dữ liệu bán lẻ thực tế của một chuỗi cửa hàng toàn cầu tại Hoa Kỳ)  
**Giấy phép:** GPL 2 (GNU General Public License v2.0)  
**Điểm khả dụng Kaggle:** 10.00 / 10  
**Nhiệm vụ học máy:** Hồi quy chuỗi thời gian (Time-Series Regression) — dự đoán doanh số bán hàng tổng hợp theo ngày  
**Biến mục tiêu:** `Sales` — doanh thu trên mỗi giao dịch (USD)

Tập dữ liệu này ghi lại **9.800 giao dịch bán lẻ** của một chuỗi siêu thị toàn cầu trong vòng **4 năm (từ ngày 03/01/2015 đến 30/12/2018)** tại thị trường Hoa Kỳ. Mỗi bản ghi mô tả một đơn hàng cụ thể với thông tin đầy đủ về khách hàng, sản phẩm, địa lý, phương thức vận chuyển và giá trị doanh thu tương ứng.

Câu hỏi nghiên cứu trung tâm là: **"Dựa trên dữ liệu giao dịch lịch sử, mô hình có thể dự đoán chính xác doanh số bán hàng của 7 ngày tiếp theo hay không?"** — đây là bài toán hồi quy chuỗi thời gian, yêu cầu tổng hợp dữ liệu giao dịch theo ngày, phát hiện xu hướng dài hạn và tính chu kỳ (seasonality) theo tháng/quý/năm, trước khi áp dụng các mô hình dự báo như Linear Regression, ARIMA và XGBoost.

---

#### 2.2.1.2 Thống kê Số lượng Mẫu

Tập dữ liệu được cung cấp dưới dạng **một tệp CSV duy nhất** (`train.csv`), không có tập kiểm tra tách biệt theo cấu trúc Kaggle thông thường. Bảng 2.17 trình bày thống kê tổng quát.

**Bảng 2.17. Thống kê số lượng mẫu**

| Thông tin | Giá trị |
|:----------|--------:|
| Tổng số giao dịch (dòng) | 9.800 |
| Tổng số cột | 18 |
| Số đặc trưng đầu vào | 17 |
| Biến mục tiêu | `Sales` |
| Khoảng thời gian | 03/01/2015 → 30/12/2018 (4 năm) |
| Số đơn hàng duy nhất (`Order ID`) | 4.922 |
| Số khách hàng duy nhất (`Customer ID`) | 793 |
| Số sản phẩm duy nhất (`Product ID`) | 1.861 |
| Giá trị thiếu | `Postal Code`: 11 dòng (~0,11%); 17 cột còn lại: 0 |
| Kiểu tệp | CSV (một tệp duy nhất: `train.csv`) |

> **Lưu ý:** Dữ liệu ở cấp **giao dịch** (transaction-level), không phải chuỗi thời gian đã tổng hợp theo ngày. Để áp dụng các mô hình chuỗi thời gian, cần thực hiện bước tổng hợp (resample/aggregate) theo ngày/tuần trước khi huấn luyện — đây là bước tiền xử lý bắt buộc và sẽ làm giảm đáng kể số điểm dữ liệu thực tế trong chuỗi.

##### 2.2.1.2.1 Phân phối Theo Danh mục

Bảng 2.18 trình bày phân phối số lượng giao dịch theo ba chiều phân loại chính: danh mục sản phẩm, phân khúc khách hàng và khu vực địa lý.

**Bảng 2.18. Phân phối giao dịch theo danh mục**

| Chiều phân loại | Giá trị | Số giao dịch | Tỷ lệ |
|:----------------|:--------|-------------:|------:|
| **Danh mục sản phẩm** (`Category`) | Office Supplies | 5.909 | 60,3% |
| | Furniture | 2.078 | 21,2% |
| | Technology | 1.813 | 18,5% |
| **Phân khúc khách hàng** (`Segment`) | Consumer | 5.101 | 52,1% |
| | Corporate | 2.953 | 30,1% |
| | Home Office | 1.746 | 17,8% |
| **Khu vực** (`Region`) | West | 3.140 | 32,0% |
| | East | 2.785 | 28,4% |
| | Central | 2.277 | 23,2% |
| | South | 1.598 | 16,3% |
| **Phương thức vận chuyển** (`Ship Mode`) | Standard Class | — | ~60% ước tính |
| | Second Class | — | ~20% ước tính |
| | First Class | — | ~15% ước tính |
| | Same Day | — | ~5% ước tính |

> **Lưu ý:** Phân phối `Ship Mode` là ước tính dựa trên đặc điểm điển hình của bộ dữ liệu Superstore; các giá trị chính xác sẽ được xác nhận trong bước EDA.

**Nhận xét:** Bộ dữ liệu có sự **mất cân bằng rõ rệt** theo danh mục sản phẩm — Office Supplies chiếm hơn 60% tổng giao dịch, trong khi Technology chỉ chiếm 18,5%. Điều này không gây vấn đề cho bài toán hồi quy tổng hợp theo ngày, nhưng cần lưu ý khi phân tích dự báo theo từng danh mục riêng lẻ. Khu vực West và East chiếm ~60% tổng giao dịch, phản ánh mật độ dân số và thương mại cao hơn ở hai vùng này.

---

#### 2.2.1.3 Mô tả Đặc trưng

Tập dữ liệu gồm 17 đặc trưng đầu vào và 1 biến mục tiêu, được phân chia thành bảy nhóm chức năng. Bảng 2.19 mô tả chi tiết từng cột.

**Bảng 2.19. Mô tả đặc trưng của tập dữ liệu Superstore Sales Dataset**

| STT | Tên cột | Kiểu dữ liệu | Nhóm | Mô tả | Giá trị / Khoảng giá trị |
|:---:|:--------|:-------------|:-----|:------|:--------------------------|
| 1 | `Row ID` | Số nguyên (int64) | Định danh | Chỉ số dòng tuần tự, duy nhất | 1 – 9.800; không sử dụng trong mô hình |
| 2 | `Order ID` | Chuỗi ký tự (object) | Định danh | Mã đơn hàng duy nhất | Ví dụ: `CA-2015-152156`; 4.922 đơn hàng phân biệt |
| 3 | `Order Date` | Chuỗi ký tự → datetime | Thời gian | Ngày đặt hàng — **cột thời gian chính** | 03/01/2015 → 30/12/2018; định dạng `DD/MM/YYYY` |
| 4 | `Ship Date` | Chuỗi ký tự (object) | Thời gian | Ngày vận chuyển hàng hóa | Sau `Order Date` từ 0 đến 7 ngày |
| 5 | `Ship Mode` | Phân loại danh nghĩa | Vận chuyển | Phương thức/tốc độ giao hàng | `"Standard Class"`, `"Second Class"`, `"First Class"`, `"Same Day"` |
| 6 | `Customer ID` | Chuỗi ký tự (object) | Khách hàng | Mã định danh khách hàng | 793 khách hàng phân biệt |
| 7 | `Customer Name` | Chuỗi ký tự (object) | Khách hàng | Họ tên đầy đủ của khách hàng | Không sử dụng trực tiếp trong mô hình |
| 8 | `Segment` | Phân loại danh nghĩa | Khách hàng | Phân khúc khách hàng theo loại hình | `"Consumer"` (52,1%), `"Corporate"` (30,1%), `"Home Office"` (17,8%) |
| 9 | `Country` | Phân loại danh nghĩa | Địa lý | Quốc gia của đơn hàng | `"United States"` (100% — giá trị hằng số) |
| 10 | `City` | Phân loại danh nghĩa | Địa lý | Thành phố giao hàng | Nhiều giá trị trên toàn Hoa Kỳ |
| 11 | `State` | Phân loại danh nghĩa | Địa lý | Bang/tiểu bang giao hàng | 49 bang trên tổng 50 bang của Hoa Kỳ |
| 12 | `Postal Code` | Số thực (float64) | Địa lý | Mã bưu chính | 11 giá trị null (~0,11%); còn lại là mã số hợp lệ |
| 13 | `Region` | Phân loại danh nghĩa | Địa lý | Khu vực thị trường | `"West"` (32%), `"East"` (28,4%), `"Central"` (23,2%), `"South"` (16,3%) |
| 14 | `Product ID` | Chuỗi ký tự (object) | Sản phẩm | Mã định danh sản phẩm | 1.861 sản phẩm phân biệt |
| 15 | `Category` | Phân loại danh nghĩa | Sản phẩm | Danh mục sản phẩm cấp 1 | `"Office Supplies"` (60,3%), `"Furniture"` (21,2%), `"Technology"` (18,5%) |
| 16 | `Sub-Category` | Phân loại danh nghĩa | Sản phẩm | Danh mục sản phẩm cấp 2 | 17 giá trị phân biệt; phổ biến nhất: Binders (1.492), Paper (1.338), Furnishings (931) |
| 17 | `Product Name` | Chuỗi ký tự (object) | Sản phẩm | Tên đầy đủ của sản phẩm | Mô tả chi tiết; không mã hóa trực tiếp vào mô hình |
| 18 | `Sales` | Số thực (float64) | **Biến mục tiêu** | Doanh thu thực tế trên mỗi giao dịch (USD) | Min=0,44 \| Q1=17,25 \| Median=54,49 \| Mean=230,77 \| Q3=210,61 \| Max=22.638,48 \| Std=626,65 |

##### 2.2.1.3.1 Đặc điểm Dữ liệu và Lưu ý Phân tích

Tập dữ liệu Superstore Sales có một số đặc điểm quan trọng cần xem xét kỹ trước khi xây dựng mô hình chuỗi thời gian:

1. **Yêu cầu tổng hợp dữ liệu theo thời gian:** Dữ liệu gốc ở cấp giao dịch — mỗi dòng là một sản phẩm trong một đơn hàng. Để tạo chuỗi thời gian, cần nhóm và tổng hợp (`groupby + resample`) cột `Sales` theo ngày (hoặc tuần), tạo ra time series với tần suất nhất quán. Sau khi tổng hợp theo ngày, số điểm dữ liệu giảm từ 9.800 giao dịch xuống còn khoảng 1.000–1.100 ngày trong 4 năm — đây mới là chuỗi đầu vào thực sự cho các mô hình ARIMA và XGBoost.

2. **Biến mục tiêu `Sales` có độ lệch phải mạnh (right-skewed):** Với mean=230,77 và median=54,49 (mean >> median), cùng max=22.638,48 và std=626,65 — phân phối của `Sales` bị kéo lệch mạnh về phía phải bởi các giao dịch giá trị cao bất thường (outliers). Cần xem xét biến đổi logarithm (`log1p`) hoặc xử lý outliers trước khi huấn luyện mô hình để tránh làm méo lỗi huấn luyện.

3. **Cột `Country` là hằng số (constant feature):** Toàn bộ 9.800 bản ghi đều có `Country = "United States"`. Cột này không cung cấp thông tin phân biệt và cần được loại bỏ khỏi tập đặc trưng đầu vào.

4. **`Postal Code` có 11 giá trị null (0,11%):** Tỷ lệ thiếu rất nhỏ. Có thể điền giá trị bằng tra cứu qua tổ hợp `City`/`State`, hoặc đơn giản hơn — loại bỏ cột này khỏi pipeline hồi quy chuỗi thời gian vì mã bưu chính không đóng góp trực tiếp vào dự báo doanh thu tổng hợp.

5. **Cần feature engineering từ `Order Date` cho XGBoost:** Không giống ARIMA là mô hình thuần thống kê xử lý chuỗi thời gian ngầm định, XGBoost yêu cầu tạo các đặc trưng thời gian tường minh như: `day_of_week`, `month`, `quarter`, `year`, `lag_7`, `lag_14`, `rolling_mean_7`, `rolling_mean_30` để học được các chu kỳ và xu hướng trong chuỗi doanh số.

6. **Giấy phép GPL 2:** Khác với CC0 hay CC BY thường thấy trên Kaggle, giấy phép GPL 2 yêu cầu bất kỳ phần mềm nào phân phối dữ liệu này cũng phải mở mã nguồn theo điều khoản tương tự. Trong phạm vi nghiên cứu học thuật nội bộ, điều này không gây trở ngại, nhưng cần lưu ý nếu kết quả được thương mại hóa.

---

#### 2.2.1.4 Tổng quan Các Nghiên cứu Liên quan

Dự báo doanh số bán lẻ là bài toán kinh điển trong lĩnh vực phân tích chuỗi thời gian, thu hút sự quan tâm của nhiều nhóm nghiên cứu từ thống kê kinh tế đến học máy sâu. Bộ dữ liệu Superstore (gốc từ Tableau) đã được sử dụng trực tiếp trong một số nghiên cứu gần đây, trong khi nhiều công trình khác giải quyết bài toán tương tự trên các bộ dữ liệu bán lẻ khác với phương pháp có thể đối chiếu trực tiếp.

> **Lưu ý:** Các bài báo trong bảng dưới đây được phân thành hai nhóm — (a) nghiên cứu **trực tiếp sử dụng Superstore dataset** (được đánh dấu ★) và (b) nghiên cứu về **bài toán dự báo doanh số bán lẻ tương tự** với phương pháp phù hợp để đối sánh kết quả.

##### 2.2.1.4.1 Các nghiên cứu tiêu biểu

**Bảng 2.20. Tổng hợp các nghiên cứu liên quan về dự báo doanh số bán lẻ theo chuỗi thời gian**

| STT | Tác giả & Năm | Tạp chí / Hội nghị | Bối cảnh / Dữ liệu | Phương pháp chính | Kết quả chính | Ghi chú |
|:---:|:--------------|:-------------------|:-------------------|:------------------|:--------------|:--------|
| 1 | AbdElminaam et al. (2024) ★ [1] | *2024 International Conference on Smart Applications, Communications and Networking (SmartNets)*, IEEE | ★ Hai bộ dữ liệu: Superstore Sales (Tableau) + bộ dữ liệu cửa hàng bán lẻ thứ hai | ML truyền thống + ARIMA cho bài toán dự báo chuỗi thời gian; so sánh nhiều mô hình | ARIMA đạt hiệu năng nổi bật trong dự báo chuỗi thời gian với R² cao; các mô hình ML phù hợp bài toán phân loại và hồi quy phi tuyến | **Nghiên cứu duy nhất trong bảng trực tiếp dùng Superstore; 6 trích dẫn; chi tiết số liệu từ abstract** |
| 2 | Reddy et al. (2024) ★ [2] | *Proceedings on Communications and Networking* (Springer), Chương 162 trong LNNS series, 2024 | ★ Superstore Sales Dataset (2015–2018); phân tích EDA chuyên sâu | EDA + ARIMA + Facebook Prophet; phân tích tính chu kỳ và xu hướng theo mùa | ARIMA và Prophet đều nắm bắt được seasonality trong dữ liệu Superstore; Prophet linh hoạt hơn trong xử lý khoảng trống dữ liệu | **Trực tiếp dùng cùng dataset; phân tích ARIMA + Prophet; kết quả từ abstract** |
| 3 | Kaviya et al. (2025) ★ [3] | *2025 International Conference on Machine Learning and Digital Transformation*, IEEE | ★ Superstore database 2015–2018; quản lý tồn kho siêu thị | ARIMA + SARIMAX; phân tích seasonality, trend; so sánh với các phương pháp thống kê khác | SARIMAX phù hợp nhất cho bộ dữ liệu Superstore nhờ xử lý tốt thành phần mùa vụ (seasonal component) | **Trực tiếp dùng cùng dataset; ARIMA vs SARIMAX; 1 trích dẫn; kết quả từ abstract** |
| 4 | Srivastava (2026) ★ [4] | *International Journal of Emerging Research in Engineering and Technology (IJERET)*, Vol. 7, No. 2, 2026. DOI: 10.63282/3050-922X.IJERET-V7I2P115 | ★ Kaggle Superstore sales data; quản lý chuỗi cung ứng bán lẻ tích hợp ERP | DNN (Deep Neural Network) vs. Decision Tree vs. Random Forest | **DNN tốt nhất: MAE=2,277 \| RMSE=2,814 \| MAPE=13,72% \| R²=92,0%**; DNN vượt trội Decision Tree và Random Forest | **Trực tiếp dùng cùng dataset Kaggle; có số liệu đầy đủ; DNN không thuộc pipeline của nhóm → dùng làm tham chiếu kết quả** |
| 5 | Paliari et al. (2021) [5] | *2021 12th International Conference on Information, Intelligence, Systems and Applications (IISA)*, IEEE. DOI: 10.1109/IISA52426.2021.9555520 | Nhiều bộ dữ liệu kinh tế–xã hội từ cơ sở dữ liệu công khai; dự báo chuỗi thời gian tổng quát | LSTM (tối ưu) vs. XGBoost (tối ưu) vs. ARIMA (tối ưu); hyperparameter tuning cho cả ba mô hình | XGBoost và LSTM đều vượt ARIMA trên dữ liệu phi tuyến; XGBoost có lợi thế về tốc độ huấn luyện; ARIMA phù hợp dữ liệu tuyến tính, ít noise | **84 trích dẫn; điểm chuẩn so sánh 3 phương pháp chính của nhóm; kết quả từ abstract** |
| 6 | Khalid et al. (2025) [6] | *Journal of Emerging Technology and Digital Transformation (JETDT)*, Vol. 4, No. 3, 2025. ISSN Online: 3006-9726 | Dữ liệu thương mại điện tử thực tế; dự báo doanh số e-commerce | LSTM vs. SARIMA vs. XGBoost; đánh giá trên MAE, RMSE, R² | **LSTM tốt nhất: MAE=59.553 \| RMSE=75.859 \| R²=0,561**; SARIMA R²=−1,41; XGBoost R²=−2,33 — cả hai âm trên tập này | SARIMA và XGBoost thất bại khi dữ liệu phi tuyến mạnh; LSTM vượt trội nhờ nhớ dài hạn (long-term memory) |
| 7 | Dankorpho (2024) [7] | *Journal of Computer Science and Technology Studies (JCSTS)*, Al-Kindi Publishers, 2024 | Dữ liệu kinh doanh bán lẻ thực tế; bài toán dự báo doanh số theo sản phẩm | XGBoost so sánh với phương pháp truyền thống (ARIMA, hồi quy tuyến tính) | **XGBoost: MAE=1,30**; vượt trội so với các phương pháp truyền thống trong dự báo doanh số phi tuyến | 14 trích dẫn; xác nhận ưu thế của XGBoost cho bài toán bán lẻ; chi tiết từ abstract |

##### 2.2.1.4.2 Phân tích Tổng hợp

Từ việc tổng hợp các công trình nghiên cứu, có thể rút ra một số nhận định chung phục vụ nghiên cứu của nhóm:

1. **ARIMA là đường cơ sở mạnh cho chuỗi thời gian tuyến tính:** AbdElminaam et al. (2024) và Kaviya et al. (2025) đều xác nhận ARIMA/SARIMAX đạt hiệu năng tốt trên chính bộ dữ liệu Superstore — đặc biệt khi dữ liệu đã được tổng hợp theo ngày và có thành phần xu hướng rõ ràng. Tuy nhiên, ARIMA gặp khó khăn khi chuỗi có nhiều điểm gián đoạn hoặc biến động đột ngột không theo quy luật tuyến tính.

2. **XGBoost vượt trội trong bài toán phi tuyến:** Dankorpho (2024) và Paliari et al. (2021) xác nhận XGBoost vượt các phương pháp thống kê truyền thống khi dữ liệu bán lẻ có tính phi tuyến cao và nhiều đặc trưng ngoại sinh. Kết quả của Khalid et al. (2025) cho thấy XGBoost có thể thất bại (R² âm) khi thiếu feature engineering phù hợp — điều này nhấn mạnh tầm quan trọng của việc tạo lag features và rolling statistics từ cột `Order Date`.

3. **Hồi quy tuyến tính là đường cơ sở tối giản:** Không có nghiên cứu nào trong danh sách sử dụng Linear Regression đơn thuần cho dự báo chuỗi thời gian bán lẻ, vì phân phối `Sales` lệch phải mạnh và các mối quan hệ phi tuyến là phổ biến. Linear Regression trong pipeline của nhóm đóng vai trò **đường cơ sở tối thiểu (minimal baseline)** để xác định mức độ cải thiện của ARIMA và XGBoost.

4. **Tổng hợp theo ngày là bước thiết yếu:** Cả ba nghiên cứu trực tiếp trên Superstore (AbdElminaam, Reddy, Kaviya) đều thực hiện tổng hợp dữ liệu theo ngày trước khi áp dụng mô hình. Đây là quy trình chuẩn không thể bỏ qua cho bộ dữ liệu này.

5. **Kết quả tham chiếu từ Srivastava (2026):** DNN đạt R²=92,0% trên cùng bộ dữ liệu Kaggle Superstore. Đây là mức trần thực tế (practical ceiling) để đánh giá so sánh — nhóm kỳ vọng XGBoost với feature engineering tốt có thể đạt R² > 80%, trong khi ARIMA dao động trong khoảng 60–80% tùy thuộc vào cách xử lý seasonality.

---

#### 2.2.1.5 Các Chỉ số Đánh giá

Bài toán hồi quy chuỗi thời gian yêu cầu sử dụng các chỉ số đánh giá phù hợp với bản chất dự báo liên tục. Đặc biệt, do `Sales` có độ lệch phải lớn và khoảng giá trị rộng (0,44 – 22.638), các chỉ số đánh giá cần xét trên cả giá trị tuyệt đối và tỷ lệ phần trăm. Nhóm áp dụng các chỉ số sau:

**Bảng 2.21. Các chỉ số đánh giá mô hình hồi quy chuỗi thời gian**

| Chỉ số | Ký hiệu | Công thức | Ý nghĩa |
|:-------|:-------:|:----------|:--------|
| Sai số tuyệt đối trung bình | MAE | $\frac{1}{n}\sum_{i=1}^{n}\lvert y_i - \hat{y}_i\rvert$ | Trung bình sai lệch tuyệt đối; dễ diễn giải theo đơn vị USD; ít nhạy với outliers |
| Căn bậc hai sai số bình phương trung bình | RMSE | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | Trừng phạt nặng sai số lớn; nhạy với outliers; cùng đơn vị USD với `Sales` |
| Sai số phần trăm tuyệt đối trung bình | MAPE | $\frac{100\%}{n}\sum_{i=1}^{n}\left\lvert\frac{y_i - \hat{y}_i}{y_i}\right\rvert$ | Sai số theo tỷ lệ phần trăm; dễ diễn giải cho quản lý kinh doanh; không xác định khi $y_i = 0$ |
| Sai số phần trăm tuyệt đối đối xứng | SMAPE | $\frac{100\%}{n}\sum_{i=1}^{n}\frac{2\lvert y_i - \hat{y}_i\rvert}{\lvert y_i\rvert + \lvert\hat{y}_i\rvert}$ | Phiên bản cải tiến của MAPE; xử lý được trường hợp giá trị thực gần 0; phạm vi [0%, 200%] |
| Hệ số xác định | R² | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Tỷ lệ phương sai được mô hình giải thích; R²=1 là hoàn hảo; R² âm = tệ hơn đường trung bình |

> **Lưu ý:** MAPE không xác định khi giá trị thực tế bằng 0. Mặc dù bộ dữ liệu này có min(`Sales`) = 0,444 > 0 ở cấp giao dịch, sau khi tổng hợp theo ngày có thể xuất hiện ngày không có doanh thu (tổng = 0) — khi đó SMAPE là chỉ số phần trăm ưu tiên hơn MAPE.

---

### Tài liệu Tham khảo

[1] AbdElminaam, D. S., Mohamed, M., et al. (2024). Leveraging Machine Learning for Accurate Store Sales Prediction: A Comparative Study. In *2024 International Conference on Smart Applications, Communications and Networking (SmartNets)*. IEEE. https://ieeexplore.ieee.org/abstract/document/10783509/

[2] Reddy, J. B., Ashritha, N., Navya, K. R., & Kakulapati, V. (2024). A Novel Approach of Superstore Sales Data by EDA and ARIMA. In *Proceedings of the International Conference on Communications and Networking* (Lecture Notes in Networks and Systems). Springer. https://link.springer.com/chapter/10.1007/978-981-95-0269-1_162

[3] Kaviya, S., AK, S. S., Yuvasri, S., et al. (2025). Enhancing Supermarket Inventory Management Through Arima and Sarimax-Based Demand Forecasting. In *2025 International Conference on Machine Learning and Digital Transformation*. IEEE. https://ieeexplore.ieee.org/abstract/document/11041518/

[4] Srivastava, P. K. (2026). Machine Learning-Based Retail Supply Chain Management Using ERP and Sales Data Analytics. *International Journal of Emerging Research in Engineering and Technology (IJERET)*, 7(2), 119–127. https://doi.org/10.63282/3050-922X.IJERET-V7I2P115

[5] Paliari, I., Karanikola, A., & Kotsiantis, S. (2021). A comparison of the optimized LSTM, XGBOOST and ARIMA in Time Series forecasting. In *2021 12th International Conference on Information, Intelligence, Systems and Applications (IISA)* (pp. 1–7). IEEE. https://doi.org/10.1109/IISA52426.2021.9555520

[6] Khalid, I., Ahmad, J., Mustafa, S., & Rafique, S. (2025). E-Commerce Sales Forecasting by Comparing LSTM, SARIMA, and XGBoost Models. *Journal of Emerging Technology and Digital Transformation*, 4(3). https://www.journalofemergingtechnologyanddigitaltransformation.com/index.php/3/article/view/41

[7] Dankorpho, et al. (2024). Sales forecasting for retail business using XGBoost algorithm. *Journal of Computer Science and Technology Studies (JCSTS)*. Al-Kindi Publishers. https://al-kindipublishers.org/index.php/jcsts/article/view/7381
