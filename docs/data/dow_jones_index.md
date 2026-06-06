## 2.2 Nhóm B: Dữ liệu Chuỗi Thời gian (Time-Series Data)

Nhóm dữ liệu B trong dự án bao gồm các bộ dữ liệu chuỗi thời gian, trong đó mỗi quan sát được gắn với một mốc thời gian cụ thể và thứ tự các quan sát mang ý nghĩa phân tích quan trọng. Không giống dữ liệu dạng bảng thông thường, chuỗi thời gian yêu cầu các kỹ thuật xử lý riêng biệt — bao gồm phân tích xu hướng (*trend*), tính mùa vụ (*seasonality*), tự tương quan (*autocorrelation*) — cùng chiến lược phân chia tập huấn luyện/kiểm tra theo mốc thời gian thay vì phân chia ngẫu nhiên. Ba bộ dữ liệu được lựa chọn phủ các lĩnh vực ứng dụng khác nhau và tần suất thu thập khác nhau: tài chính (tuần), nhận dạng hoạt động (giây), và sản xuất điện (giờ).

---

### 2.2.1 Dow Jones Index Dataset

---

#### 2.2.1.1 Giới thiệu Tập dữ liệu

**Tên tập dữ liệu:** Dow Jones Index  
**Nguồn:** UCI Machine Learning Repository — [https://archive.ics.uci.edu/dataset/312/dow+jones+index](https://archive.ics.uci.edu/dataset/312/dow+jones+index)  
**Tác giả tạo lập:** Michael S. Brown, Mary Pelosi, Henry Dirska  
**Năm đóng góp lên UCI:** 2014 (dữ liệu từ năm 2011)  
**Giấy phép:** CC BY 4.0 (Creative Commons Attribution 4.0 International)  
**Bản chất dữ liệu:** Dữ liệu thực (*real-world*) — thu thập từ các sở giao dịch chứng khoán lớn của Hoa Kỳ  
**Tần suất thu thập:** Tuần (*weekly*) — mỗi bản ghi đại diện cho một tuần giao dịch  
**Nhiệm vụ học máy gốc:** Dự báo lợi suất cổ phiếu (*financial forecasting*); Classification, Clustering  
**Biến mục tiêu chính (trong dự án):** `percent_change_next_weeks_price` — phần trăm thay đổi giá cổ phiếu trong tuần kế tiếp

Chỉ số Công nghiệp Dow Jones (*Dow Jones Industrial Average — DJIA*) là một trong những chỉ số chứng khoán lâu đời và được theo dõi rộng rãi nhất thế giới, được tính toán từ giá cổ phiếu của 30 công ty lớn niêm yết trên sàn New York Stock Exchange (NYSE) và NASDAQ. Được giới thiệu lần đầu vào năm 1896 bởi Charles Dow, DJIA là thước đo sức khỏe tổng thể của thị trường chứng khoán Mỹ và thường được dùng như một proxy cho nền kinh tế quốc gia.

Tập dữ liệu này được xây dựng và sử dụng trong nghiên cứu của Brown, Pelosi và Dirska (2013) [1], với mục tiêu phát triển và đánh giá các thuật toán học máy có khả năng dự báo cổ phiếu nào trong rổ DJIA sẽ mang lại tỷ suất sinh lời cao nhất trong tuần tiếp theo. Sau đó, tập dữ liệu được tái sử dụng trong nghiên cứu ShapeSearch của Siddiqui và cộng sự (2020) [2] — một hệ thống khám phá dữ liệu trực quan dựa trên hình dạng xu hướng (*trendline shape-based exploration*) — như một case study thực tiễn để đánh giá hiệu quả của công cụ.

**Phạm vi ứng dụng trong dự án:** Trong khuôn khổ dự án CTH621, tập dữ liệu được sử dụng để thực nghiệm các bài toán **hồi quy chuỗi thời gian** (Regression — Nhóm B), với mục tiêu dự báo `percent_change_next_weeks_price` theo thứ tự thời gian bằng các thuật toán Linear Regression, ARIMA/SARIMA và XGBoost. Tập dữ liệu cũng được sử dụng cho bài toán **phân cụm** (Clustering), nhằm xác định các nhóm cổ phiếu có hành vi giao dịch tương đồng.

**Bảng 2.17 — Thông tin tổng quan tập dữ liệu Dow Jones Index**

| Thuộc tính | Thông tin |
|:-----------|:----------|
| **Tên dataset** | Dow Jones Index |
| **Nguồn (UCI)** | [https://archive.ics.uci.edu/dataset/312/...](https://archive.ics.uci.edu/dataset/312/dow+jones+index) |
| **DOI dataset (UCI)** | [10.24432/C5788V](https://doi.org/10.24432/C5788V) |
| **Bài báo giới thiệu** | Brown, Pelosi & Dirska (2013) |
| **Bài báo tham chiếu (README)** | Siddiqui et al. (2020), *SIGMOD '20*, DOI: [10.1145/3318464.3389722](https://doi.org/10.1145/3318464.3389722) |
| **Năm thu thập dữ liệu** | 2011 (Quý 1 và Quý 2) |
| **Giấy phép** | CC BY 4.0 |
| **Loại dữ liệu** | Chuỗi thời gian (Time-Series) — dữ liệu tài chính |
| **Tần suất thu thập** | Hàng tuần (*weekly*) — ghi nhận vào ngày làm việc cuối tuần (thường là thứ Sáu) |
| **Số bản ghi** | 750 |
| **Số đặc trưng** | 16 cột |
| **Số cổ phiếu** | 30 (toàn bộ thành phần DJIA năm 2011) |
| **Giá trị thiếu** | Không (ngoại trừ 30 giá trị trong 2 cột tính toán — xem mục 2.2.1.3.1) |

---

#### 2.2.1.2 Thống kê Số lượng Mẫu và Cấu trúc Chuỗi Thời gian

Tập dữ liệu bao gồm dữ liệu giao dịch tuần của **30 cổ phiếu** trong rổ DJIA, trải dài trong **hai quý đầu năm 2011** (từ tuần kết thúc ngày 07/01/2011 đến tuần kết thúc ngày 03/06/2011). Thiết kế dữ liệu theo dạng **bảng dọc (long format)**: mỗi hàng là một cổ phiếu trong một tuần cụ thể, tạo thành chuỗi 25 tuần cho mỗi cổ phiếu.

**Bảng 2.18 — Thống kê số lượng mẫu**

| Thông tin | Giá trị |
|:----------|--------:|
| Tổng số bản ghi | 750 |
| Tổng số cột | 16 |
| Số cổ phiếu | 30 |
| Số tuần giao dịch | 25 (trải trên 2 quý) |
| Giai đoạn thời gian | 07/01/2011 — 03/06/2011 |
| Bản ghi Quý 1 (Jan–Mar) | 360 |
| Bản ghi Quý 2 (Apr–Jun) | 390 |
| Giá trị thiếu (thực chất) | 30 (chỉ ảnh hưởng 2 cột tính toán) |

> **Ghi chú về cấu trúc:** Với 30 cổ phiếu và 25 tuần giao dịch, mỗi cổ phiếu đóng góp đúng 25 bản ghi. 30 giá trị `NaN` trong cột `percent_change_volume_over_last_wk` và `previous_weeks_volume` chính là tuần đầu tiên của từng cổ phiếu (30 × 1 = 30) — do không có dữ liệu tuần trước để tính biến động khối lượng. Đây là đặc điểm cố hữu của dữ liệu chuỗi thời gian, không phải lỗi thu thập.

**Cách phân chia dữ liệu trong nghiên cứu gốc [1]:**

Brown, Pelosi & Dirska (2013) sử dụng **Quý 1 (Jan–Mar) làm tập huấn luyện** và **Quý 2 (Apr–Jun) làm tập kiểm tra** — đây là phương pháp Chronological Split, đảm bảo mô hình không "nhìn trước" dữ liệu tương lai. Chiến lược này được kế thừa trong dự án CTH621, phù hợp với ràng buộc bắt buộc áp dụng Chronological Split cho mọi bài toán hồi quy chuỗi thời gian.

---

#### 2.2.1.3 Mô tả Đặc trưng

Tập dữ liệu gồm 16 cột, bao phủ bốn nhóm thông tin: **(1)** định danh và phân kỳ, **(2)** giá OHLC (*Open-High-Low-Close*) và khối lượng giao dịch tuần hiện tại, **(3)** dữ liệu tuần tiếp theo (look-ahead), và **(4)** các chỉ số phái sinh và cổ tức. Bảng 2.19 mô tả chi tiết từng cột.

**Bảng 2.19 — Mô tả đặc trưng của tập dữ liệu Dow Jones Index**

| STT | Tên cột | Kiểu dữ liệu (CSV) | Nhóm | Mô tả | Ghi chú |
|:---:|:--------|:-------------------|:-----|:------|:--------|
| 1 | `quarter` | Integer | Định danh | Quý trong năm | `1` = Jan–Mar; `2` = Apr–Jun |
| 2 | `stock` | Categorical (string) | Định danh | Ký hiệu mã cổ phiếu | 30 mã: AA, AXP, BA, BAC, CAT, CSCO, CVX, DD, DIS, GE, HD, HPQ, IBM, INTC, JNJ, JPM, KO, KRFT, MCD, MMM, MRK, MSFT, PFE, PG, T, TRV, UTX, VZ, WMT, XOM |
| 3 | `date` | String (MM/DD/YYYY) | Định danh | Ngày làm việc cuối tuần (thường là thứ Sáu) | Cần chuyển đổi sang kiểu `datetime` trước khi xử lý |
| 4 | `open` | String (có tiền tố `$`) | OHLC | Giá mở cửa đầu tuần | Cần xóa ký tự `$` và chuyển về `float` |
| 5 | `high` | String (có tiền tố `$`) | OHLC | Giá cao nhất trong tuần | Cần xóa ký tự `$` và chuyển về `float` |
| 6 | `low` | String (có tiền tố `$`) | OHLC | Giá thấp nhất trong tuần | Cần xóa ký tự `$` và chuyển về `float` |
| 7 | `close` | String (có tiền tố `$`) | OHLC | Giá đóng cửa cuối tuần | Cần xóa ký tự `$` và chuyển về `float` |
| 8 | `volume` | Integer | Khối lượng | Tổng khối lượng cổ phiếu giao dịch trong tuần (số lượng cổ phiếu đổi tay) | Đơn vị: cổ phiếu |
| 9 | `percent_change_price` | Float | Chỉ số phái sinh | Phần trăm thay đổi giá trong tuần hiện tại | $\frac{close - open}{open} \times 100$ |
| 10 | `percent_change_volume_over_last_wk` | Float | Chỉ số phái sinh | Phần trăm thay đổi khối lượng giao dịch so với tuần trước | **30 giá trị NaN** — tuần đầu tiên của mỗi cổ phiếu |
| 11 | `previous_weeks_volume` | Float | Khối lượng tham chiếu | Khối lượng giao dịch của tuần trước | **30 giá trị NaN** — tuần đầu tiên của mỗi cổ phiếu |
| 12 | `next_weeks_open` | String (có tiền tố `$`) | Look-ahead | Giá mở cửa đầu tuần sau | Dữ liệu look-ahead — chỉ dùng để tính biến mục tiêu, không dùng làm đặc trưng đầu vào |
| 13 | `next_weeks_close` | String (có tiền tố `$`) | Look-ahead | Giá đóng cửa cuối tuần sau | Dữ liệu look-ahead — chỉ dùng để tính biến mục tiêu, không dùng làm đặc trưng đầu vào |
| 14 | `percent_change_next_weeks_price` | Float | **Biến mục tiêu** | Phần trăm thay đổi giá cổ phiếu trong tuần kế tiếp | $\frac{next\_weeks\_close - next\_weeks\_open}{next\_weeks\_open} \times 100$; Trung bình: +0,24%; Min: −15,42%; Max: +9,88% |
| 15 | `days_to_next_dividend` | Integer | Cổ tức | Số ngày cho đến ngày chia cổ tức kế tiếp | |
| 16 | `percent_return_next_dividend` | Float | Cổ tức | Tỷ suất sinh lời từ cổ tức kế tiếp (%) | |

##### 2.2.1.3.1 Vấn đề Chất lượng Dữ liệu

Phân tích sơ bộ xác định hai nhóm vấn đề chính cần xử lý trong giai đoạn tiền xử lý.

Thứ nhất, **định dạng kiểu dữ liệu không chuẩn:** Sáu cột giá (`open`, `high`, `low`, `close`, `next_weeks_open`, `next_weeks_close`) được lưu dưới dạng chuỗi với tiền tố ký hiệu dollar (`$`). Mặc dù về bản chất là số thực, pandas sẽ đọc các cột này như `object`. Cần loại bỏ ký tự `$` và chuyển đổi kiểu sang `float64` trước khi huấn luyện mô hình. Tương tự, cột `date` cần được chuyển sang kiểu `datetime` để cho phép lập chỉ số thời gian.

Thứ hai, **giá trị thiếu mang tính cấu trúc:** Hai cột `percent_change_volume_over_last_wk` và `previous_weeks_volume` có đúng **30 giá trị `NaN`** — tương ứng với bản ghi đầu tiên của mỗi cổ phiếu (tuần kết thúc 07/01/2011). Đây là hiện tượng tất yếu của dữ liệu sai phân bậc-1 (*first-order differencing*): không thể tính phần trăm thay đổi khi không có tuần tham chiếu trước. Chiến lược xử lý phù hợp là điền bằng 0 (không có biến động) hoặc loại bỏ các hàng đầu tiên này trước khi huấn luyện mô hình.

Thứ ba, cần đặc biệt chú ý đến **rủi ro data leakage**: các cột `next_weeks_open`, `next_weeks_close` là dữ liệu *look-ahead* (thông tin tương lai) và **không được phép dùng làm đặc trưng đầu vào** trong mô hình. Các cột này chỉ phục vụ tính toán biến mục tiêu `percent_change_next_weeks_price`.

**Bảng 2.20 — Tổng hợp vấn đề chất lượng dữ liệu**

| Vấn đề | Các cột ảnh hưởng | Mức độ | Cách xử lý |
|:-------|:-----------------|:------:|:-----------|
| Kiểu chuỗi có ký tự `$` | `open`, `high`, `low`, `close`, `next_weeks_open`, `next_weeks_close` | Cao (bắt buộc xử lý) | Loại bỏ `$`, chuyển sang `float64` |
| Cột ngày dạng chuỗi | `date` | Cao | Chuyển sang `datetime`, đặt làm chỉ số thời gian |
| NaN cấu trúc | `percent_change_volume_over_last_wk`, `previous_weeks_volume` | Trung bình | Điền 0 hoặc bỏ hàng đầu của mỗi cổ phiếu |
| Rủi ro data leakage | `next_weeks_open`, `next_weeks_close` | Cao (bắt buộc loại) | Loại khỏi tập đặc trưng đầu vào |

---

#### 2.2.1.4 Tổng quan Các Nghiên cứu Liên quan

##### 2.2.1.4.1 Nghiên cứu gốc — Brown, Pelosi & Dirska (2013)

Tập dữ liệu Dow Jones Index được giới thiệu lần đầu trong công trình của Brown, Pelosi và Dirska (2013) [1], công bố tại Hội nghị IAPR International Conference on Machine Learning and Data Mining in Pattern Recognition. Nghiên cứu đặt ra bài toán thực tiễn: với dữ liệu giao dịch tuần của 30 cổ phiếu DJIA, thuật toán học máy nào có thể xác định cổ phiếu mang lại tỷ suất sinh lời cao nhất trong tuần tiếp theo?

Để giải quyết bài toán này, nhóm tác giả đề xuất thuật toán **Dynamic-Radius Species-Conserving Genetic Algorithm (DRSCGA)** — một biến thể của Genetic Algorithm (GA) truyền thống, trong đó bán kính loài (*species radius*) được điều chỉnh động để duy trì sự đa dạng quần thể (*population diversity*) và tránh hội tụ sớm. Thuật toán hoạt động bằng cách tìm kiếm không gian giải pháp gồm các bộ quy tắc (*rule sets*) tổ hợp từ các đặc trưng như `percent_change_price`, `percent_change_volume_over_last_wk`, `days_to_next_dividend` và `percent_return_next_dividend`, nhằm phân loại cổ phiếu thành các nhóm có lợi suất tương lai cao hay thấp.

Phương pháp thực nghiệm sử dụng **Quý 1 làm tập huấn luyện và Quý 2 làm tập kiểm tra**, tận dụng đặc điểm thời gian tuyến tính của dữ liệu. Tác giả lưu ý một điểm dữ liệu đáng chú ý: trong tuần kết thúc ngày 27/05/2011 (thuộc tập kiểm tra Quý 2), **toàn bộ 30 cổ phiếu DJIA đều ghi nhận tỷ suất sinh lời âm** — phản ánh đợt bán tháo thị trường rộng khắp trong giai đoạn này.

##### 2.2.1.4.2 Tái sử dụng Dataset — ShapeSearch (Siddiqui et al., 2020)

Trong công trình đoạt giải "Best Paper" tại SIGMOD 2020, Siddiqui và cộng sự [2] giới thiệu **ShapeSearch** — một hệ thống truy vấn linh hoạt cho phép người dùng tìm kiếm các xu hướng (*trendlines*) với hình dạng mong muốn trong tập dữ liệu thị trường tài chính. Tập dữ liệu Dow Jones Index được sử dụng như một trong những case study chứng minh tính thực tiễn của hệ thống.

ShapeSearch cho phép người dùng biểu diễn truy vấn theo ba cơ chế: phác thảo (*sketch*), ngôn ngữ tự nhiên (*natural language*), và biểu thức chính quy trực quan (*visual regular expression*). Hệ thống xây dựng một đại số hình dạng (*shape querying algebra*) với tập hợp các toán hạng nguyên thủy tối giản để biểu diễn đa dạng các mẫu hình xu hướng. Trên bộ dữ liệu Dow Jones, ShapeSearch có khả năng xác định các mẫu giá điển hình như "đầu và vai" (*head-and-shoulders*), "đỉnh ba" (*triple top*), hay các xu hướng tăng/giảm liên tục — đây là các tín hiệu kỹ thuật (*technical signals*) được sử dụng rộng rãi trong phân tích chứng khoán.

Kết quả đánh giá cho thấy ShapeSearch đạt thời gian phản hồi tương tác (*interactive response times*), vượt trội đáng kể so với các phương pháp khớp hình dạng chuỗi thời gian hiện có (*state-of-the-art trendline shape matching*). Đây là nghiên cứu tiêu biểu thuộc lĩnh vực *Visual Analytics* và *Data Management* ứng dụng trên dữ liệu chuỗi thời gian tài chính, khác biệt về phạm vi so với các nghiên cứu dự báo giá truyền thống.

##### 2.2.1.4.3 Phân tích Tổng hợp

Hai công trình trên phản ánh hai hướng tiếp cận khác nhau đối với dữ liệu chuỗi thời gian tài chính: Brown et al. (2013) [1] tiếp cận theo hướng **dự báo định lượng** với mục tiêu tối đa hóa lợi suất đầu tư, trong khi ShapeSearch [2] tiếp cận theo hướng **khám phá dữ liệu trực quan** nhằm hỗ trợ phân tích kỹ thuật.

Trong khuôn khổ dự án CTH621, hướng tiếp cận phù hợp nhất là **hồi quy chuỗi thời gian** (*time series regression*) nhằm dự báo `percent_change_next_weeks_price`. Đây là bài toán dự báo giá trị liên tục trong tương lai dựa trên các đặc trưng kỳ hiện tại và quá khứ, thuộc phạm vi các thuật toán Linear Regression, ARIMA và XGBoost được triển khai trong pipeline.

Một số lưu ý thực tiễn khi áp dụng mô hình hồi quy trên bộ dữ liệu này: **(1)** với chỉ 25 tuần dữ liệu mỗi cổ phiếu, ARIMA cần được cấu hình với bộ tham số đơn giản để tránh overfitting; **(2)** các đặc trưng lag (*lagged features*) cần được tính từ cột `close` và `volume` sau khi đã chuyển đổi kiểu dữ liệu; **(3)** nên phân tích từng cổ phiếu riêng lẻ hoặc gộp toàn bộ 30 cổ phiếu như một chuỗi duy nhất — mỗi cách có ưu điểm riêng về độ phong phú dữ liệu và tính đặc thù của từng cổ phiếu.

---

#### 2.2.1.5 Các Chỉ số Đánh giá

Với bài toán hồi quy chuỗi thời gian (*time series regression*), nhóm sử dụng các chỉ số đánh giá sau:

**Bảng 2.21 — Các chỉ số đánh giá mô hình hồi quy chuỗi thời gian**

| Chỉ số | Ký hiệu | Công thức | Ý nghĩa |
|:-------|:-------:|:----------|:--------|
| Sai số tuyệt đối trung bình | MAE | $\frac{1}{n}\sum_{i=1}^{n}\lvert y_i - \hat{y}_i \rvert$ | Sai số trung bình tính theo đơn vị gốc (điểm phần trăm); dễ diễn giải |
| Căn bậc hai sai số bình phương trung bình | RMSE | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | Phạt nặng hơn cho các sai số lớn; nhạy cảm với ngoại lệ (*outliers*) |
| Hệ số xác định | $R^2$ | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Tỷ lệ phương sai được giải thích bởi mô hình; 1,0 = hoàn hảo, 0 = tương đương baseline |

> **Lưu ý:** Trong bài toán dự báo tài chính, $R^2$ thường thấp (dưới 0,5) do tính ngẫu nhiên cao của thị trường (*market noise*). MAE và RMSE được tính theo đơn vị điểm phần trăm (%), có thể so sánh trực tiếp với biên độ biến động thực tế của dữ liệu (độ lệch chuẩn thực tế ≈ 2,52 điểm phần trăm).

---

### Tài liệu Tham khảo

[1] Brown, M. S., Pelosi, M., & Dirska, H. (2013). Dynamic-Radius Species-Conserving Genetic Algorithm for the Financial Forecasting of Dow Jones Index Stocks. In *Machine Learning and Data Mining in Pattern Recognition*, Lecture Notes in Computer Science, vol. 7988, pp. 27–41. Springer. (Dataset citation: Brown, M. (2013). Dow Jones Index [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5788V)

[2] Siddiqui, T., Luh, P., Wang, Z., Karahalios, K., & Parameswaran, A. (2020). ShapeSearch: A Flexible and Efficient System for Shape-based Exploration of Trendlines. In *Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data (SIGMOD '20)*, pp. 51–65. ACM. https://doi.org/10.1145/3318464.3389722

---

### Trích dẫn LaTeX

```latex
% BibTeX entry cho bài báo giới thiệu dataset (Brown et al., 2013)
@inproceedings{Brown2013,
  author    = {Brown, Michael S. and Pelosi, Mary and Dirska, Henry},
  title     = {Dynamic-Radius Species-Conserving Genetic Algorithm for the
               Financial Forecasting of {Dow Jones Index} Stocks},
  booktitle = {Machine Learning and Data Mining in Pattern Recognition},
  series    = {Lecture Notes in Computer Science},
  volume    = {7988},
  pages     = {27--41},
  year      = {2013},
  publisher = {Springer},
  doi       = {10.1007/978-3-642-39712-7_3}
}

% BibTeX entry cho dataset trên UCI ML Repository
@misc{Brown2013dataset,
  author    = {Brown, Michael},
  title     = {{Dow Jones Index} [Dataset]},
  year      = {2013},
  publisher = {UCI Machine Learning Repository},
  doi       = {10.24432/C5788V},
  url       = {https://archive.ics.uci.edu/dataset/312/dow+jones+index}
}

% BibTeX entry cho bài báo ShapeSearch (Siddiqui et al., 2020)
@inproceedings{Siddiqui2020,
  author    = {Siddiqui, Tarique and Luh, Paul and Wang, Zesheng
               and Karahalios, Karrie and Parameswaran, Aditya},
  title     = {{ShapeSearch}: A Flexible and Efficient System for
               Shape-based Exploration of Trendlines},
  booktitle = {Proceedings of the 2020 {ACM SIGMOD} International Conference
               on Management of Data},
  series    = {SIGMOD '20},
  pages     = {51--65},
  year      = {2020},
  publisher = {ACM},
  doi       = {10.1145/3318464.3389722},
  url       = {https://doi.org/10.1145/3318464.3389722}
}
```

**Trích dẫn trong văn bản (IEEE / APA style):**

- IEEE (Brown et al.): `[1] M. S. Brown, M. Pelosi, and H. Dirska, "Dynamic-Radius Species-Conserving Genetic Algorithm for the Financial Forecasting of Dow Jones Index Stocks," in *Machine Learning and Data Mining in Pattern Recognition*, LNCS vol. 7988, Springer, 2013, pp. 27–41.`

- IEEE (Siddiqui et al.): `[2] T. Siddiqui, P. Luh, Z. Wang, K. Karahalios, and A. Parameswaran, "ShapeSearch: A Flexible and Efficient System for Shape-based Exploration of Trendlines," in *Proc. ACM SIGMOD '20*, 2020, pp. 51–65, doi: 10.1145/3318464.3389722.`

- APA (Brown et al.): `Brown, M. S., Pelosi, M., & Dirska, H. (2013). Dynamic-Radius Species-Conserving Genetic Algorithm for the financial forecasting of Dow Jones Index stocks. *Machine Learning and Data Mining in Pattern Recognition*, LNCS 7988, 27–41. Springer.`

- APA (Siddiqui et al.): `Siddiqui, T., Luh, P., Wang, Z., Karahalios, K., & Parameswaran, A. (2020). ShapeSearch: A flexible and efficient system for shape-based exploration of trendlines. *Proceedings of ACM SIGMOD '20*, 51–65. https://doi.org/10.1145/3318464.3389722`
