### 2.2.3 Cinema Tickets Dataset

---

#### 2.2.3.1 Giới thiệu Tập dữ liệu

**Cinema Tickets Dataset** là tập dữ liệu lịch sử bán vé thực tế của một chuỗi rạp chiếu phim, được thu thập trong khoảng **8,5 tháng của năm 2018** (từ ngày 21/02/2018 đến 04/11/2018). Dữ liệu gốc đã qua bước ẩn danh hóa và mã hóa (*encoded anonymized locations*) trước khi phát hành, do đó các thông tin nhận dạng như tên rạp, tên bộ phim và địa điểm được thay thế bằng mã số nguyên. Tập dữ liệu được công bố trên Kaggle bởi tác giả **Möbius (arashnic)** theo giấy phép **CC BY-NC-SA 4.0** (Attribution-NonCommercial-ShareAlike 4.0 International) — cho phép sử dụng và chia sẻ tự do trong các dự án học thuật phi thương mại với điều kiện trích dẫn nguồn và giữ nguyên giấy phép cho các tác phẩm phái sinh.

Điểm đặc trưng quan trọng của tập dữ liệu này là **độ phân giải thời gian ở cấp độ suất chiếu** (*show-time level*): mỗi bản ghi đại diện cho một lần chiếu phim tại một rạp cụ thể, với một bộ phim xác định và trong một khung giờ nhất định. Cấu trúc này khác biệt so với các tập dữ liệu chuỗi thời gian tổng hợp theo ngày hoặc theo tháng — dẫn đến yêu cầu kỹ thuật bổ sung là **tổng hợp (aggregation)** dữ liệu theo ngày trước khi huấn luyện các mô hình chuỗi thời gian truyền thống.

**Phạm vi ứng dụng trong dự án**: Tập dữ liệu này được sử dụng cho bài toán **hồi quy chuỗi thời gian** — dự báo tổng doanh thu bán vé theo ngày trong tương lai dựa trên các mẫu lịch sử. Cột `total_sales` được chọn làm biến mục tiêu; để xây dựng chuỗi thời gian ngày, dữ liệu được tổng hợp theo cột `date`, thu được 234 điểm dữ liệu theo ngày. Ba thuật toán được triển khai là **Linear Regression**, **ARIMA** và **XGBoost** — tương ứng với hướng tiếp cận hồi quy tuyến tính cơ sở, mô hình thống kê chuỗi thời gian và học máy dựa trên cây quyết định tăng cường.

| Thuộc tính | Thông tin |
|---|---|
| **Tên dataset** | Cinema Tickets Dataset |
| **Nguồn Kaggle** | [https://www.kaggle.com/datasets/arashnic/cinema-ticket](https://www.kaggle.com/datasets/arashnic/cinema-ticket) |
| **Tác giả** | Möbius (arashnic) |
| **Giấy phép** | CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0) |
| **Usability (Kaggle)** | 10.00 / 10 |
| **Loại dữ liệu** | Chuỗi thời gian — dữ liệu thực tế đã ẩn danh hóa (*real-world anonymized*) |
| **Bài toán** | Hồi quy chuỗi thời gian (Time Series Regression) |

---

#### 2.2.3.2 Thống kê Số lượng Mẫu

Tập dữ liệu gồm **một tệp CSV duy nhất** (`cinemaTicket_Ref.csv`) với cấu trúc như sau:

**Bảng 2.27 — Thống kê cơ bản tập dữ liệu Cinema Tickets**

| Thuộc tính | Giá trị |
|---|---|
| **Số lượng mẫu (hàng)** | 142,524 |
| **Số đặc trưng (cột)** | 14 |
| **Số tệp CSV** | 1 (`cinemaTicket_Ref.csv`) |
| **Khoảng thời gian** | 21/02/2018 – 04/11/2018 (~8,5 tháng) |
| **Độ phân giải thời gian** | Cấp độ suất chiếu (*show-time level*) — mỗi hàng = 1 lần chiếu tại 1 rạp |
| **Số ngày duy nhất** | 234 ngày |
| **Số bộ phim duy nhất** | 48 (`film_code`: 1471–1589) |
| **Số rạp chiếu duy nhất** | 246 (`cinema_code`: 32–637) |
| **Số lượng giá trị thiếu** | 250 (`occu_perc`: 125, `capacity`: 125) |
| **Biến mục tiêu** | `total_sales` (doanh thu mỗi suất chiếu) |
| **Kiểu bài toán** | Hồi quy chuỗi thời gian |

> **Lưu ý:** Tập dữ liệu ở độ phân giải suất chiếu — cần **tổng hợp theo ngày** (groupby `date`, sum `total_sales`) để tạo chuỗi thời gian 234 điểm trước khi huấn luyện ARIMA và Linear Regression. Đối với XGBoost, có thể sử dụng trực tiếp toàn bộ 142,524 hàng với đầy đủ 14 đặc trưng, hoặc sử dụng chuỗi đã tổng hợp theo ngày với đặc trưng lag bổ sung.

---

##### 2.2.3.2.1 Phân phối Theo Quý và Tháng

**Bảng 2.28 — Phân phối mẫu và doanh thu hằng ngày theo Quý và Tháng**

| Quý | Tháng | Số mẫu (suất chiếu) | Tỷ lệ (%) | Số phim duy nhất | Doanh thu ngày TB (nghìn đơn vị) |
|---|---|---|---|---|---|
| **Q1** | Tháng 2 (21–28) | ~1,500 | ~1,1% | 1 | Rất thấp (giai đoạn khởi đầu) |
| | Tháng 3 | ~6,500 | ~4,6% | 7 | Thấp |
| **Q2** | Tháng 4 | ~14,000 | ~9,8% | 8 | Trung bình |
| | Tháng 5 | ~24,000 | ~16,8% | 15 | Cao |
| | Tháng 6 | ~20,300 | ~14,2% | 23 | Cao |
| **Q3** | Tháng 7 | ~19,500 | ~13,7% | 25 | Cao |
| | Tháng 8 | ~19,000 | ~13,3% | 30 | Cao |
| | Tháng 9 | ~15,600 | ~10,9% | 29 | Trung bình–Cao |
| **Q4** | Tháng 10 | ~19,200 | ~13,5% | 32 | Trung bình–Cao |
| | Tháng 11 (1–4) | ~3,000 | ~2,1% | 23 | Trung bình |

**Phân phối theo Quý (tổng hợp):**

| Quý | Số mẫu | Tỷ lệ (%) |
|---|---|---|
| Q1 (Tháng 1–3) | 7,996 | 5,6% |
| Q2 (Tháng 4–6) | 58,270 | 40,9% |
| Q3 (Tháng 7–9) | 54,057 | 37,9% |
| Q4 (Tháng 10–12) | 22,201 | 15,6% |

**Thống kê mô tả cột `total_sales` theo suất chiếu:**

| Chỉ số thống kê | Giá trị |
|---|---|
| Min | 20,000 |
| Q1 (25%) | 1,260,000 |
| Trung vị (Q2) | 3,720,000 |
| Trung bình (mean) | 12,347,280 |
| Q3 (75%) | 11,100,000 |
| Max | 1,262,820,000 |
| Độ lệch chuẩn | 30,654,860 |

**Thống kê mô tả `total_sales` sau tổng hợp theo ngày (234 điểm):**

| Chỉ số thống kê | Giá trị |
|---|---|
| Min | 180,000 |
| Q1 (25%) | 3,636,723,000 |
| Trung vị (Q2) | 6,156,187,000 |
| Trung bình (mean) | 7,520,441,000 |
| Q3 (75%) | 10,794,160,000 |
| Max | 21,319,390,000 |
| Độ lệch chuẩn | 4,935,529,000 |

> **Lưu ý về phân phối:** Phân phối `total_sales` ở cấp độ suất chiếu **lệch phải rất mạnh** (mean 12,3 triệu >> median 3,7 triệu; max 1,26 tỷ): phần lớn suất chiếu có doanh thu thấp–trung bình, trong khi một số rất ít suất chiếu đặc biệt (phim bom tấn, rạp lớn, giờ cao điểm) tạo ra doanh thu cực cao. Hiện tượng này ít rõ ràng hơn ở chuỗi tổng hợp theo ngày (mean 7,5 tỷ so với median 6,2 tỷ). Trong quá trình huấn luyện, cần cân nhắc **biến đổi logarithm** hoặc **loại bỏ ngoại lệ cực đoan** để đảm bảo ổn định mô hình.

---

#### 2.2.3.3 Mô tả Đặc trưng

**Bảng 2.29 — Mô tả chi tiết các đặc trưng tập dữ liệu Cinema Tickets**

| STT | Tên cột | Kiểu dữ liệu | Nhóm | Mô tả | Giá trị / Phạm vi |
|---|---|---|---|---|---|
| 1 | `film_code` | `int64` | Định danh | Mã bộ phim (ẩn danh hóa) — thay thế cho tên phim thực tế | 48 giá trị duy nhất (1471–1589) |
| 2 | `cinema_code` | `int64` | Định danh | Mã rạp chiếu (ẩn danh hóa) — thay thế cho tên và địa chỉ rạp | 246 giá trị duy nhất (32–637) |
| 3 | `date` | `object` (str) | Thời gian | Ngày chiếu phim, định dạng YYYY-MM-DD | 21/02/2018–04/11/2018; 234 ngày duy nhất |
| 4 | `month` | `int64` | Thời gian | Tháng trong năm (trích xuất từ `date`) | 2–11 (Tháng 2 đến Tháng 11) |
| 5 | `quarter` | `int64` | Thời gian | Quý trong năm (trích xuất từ `date`) | 1–4 |
| 6 | `day` | `int64` | Thời gian | Ngày trong tháng (trích xuất từ `date`) | 1–31 |
| 7 | `show_time` | `int64` | Đặc trưng suất chiếu | Chỉ số suất chiếu trong ngày (được mã hóa, không phải giờ đồng hồ trực tiếp) | 1–60; 51 giá trị duy nhất; mean=3,93; ~95% giá trị trong khoảng 1–8 |
| 8 | `ticket_price` | `float64` | Đặc trưng vé | Giá vé trung bình mỗi suất chiếu (đơn vị tiền tệ không được chỉ định) | 483,87–700.000; mean=81.234 |
| 9 | `capacity` | `float64` | Đặc trưng rạp | Sức chứa của rạp chiếu (125 giá trị thiếu; tồn tại giá trị âm do mã hóa) | –2–9.692; 125 null |
| 10 | `occu_perc` | `float64` | Đặc trưng rạp | Tỷ lệ lấp đầy rạp (%); có thể >100% do cách tính trong dữ liệu ẩn danh hóa | 0–147,5%; mean=19,97%; median=10,35%; 125 null |
| 11 | `tickets_sold` | `int64` | Chỉ số bán vé | Số vé bán ra trong suất chiếu | 1–8.499; mean=140,14; median=50 |
| 12 | `tickets_out` | `int64` | Chỉ số bán vé | Số vé bị hoàn trả hoặc hủy | 0–311; mean=0,24 |
| 13 | `ticket_use` | `int64` | Chỉ số sử dụng | Số vé thực tế được sử dụng vào rạp; tồn tại giá trị âm do mã hóa | –219–8.499; 61 hàng có giá trị < 0 |
| 14 | `total_sales` | `int64` | **Biến mục tiêu** | Tổng doanh thu bán vé mỗi suất chiếu | 20.000–1.262.820.000; mean=12.347.280 |

---

##### 2.2.3.3.1 Đặc điểm Kỹ thuật và Lưu ý Phân tích

**1. Bản chất dữ liệu ẩn danh hóa:**
Toàn bộ thông tin định danh trong tập dữ liệu đã được ẩn danh hóa — `film_code` (48 mã nguyên trong khoảng 1471–1589) và `cinema_code` (246 mã nguyên trong khoảng 32–637) không thể ánh xạ ngược về tên phim hay địa điểm rạp thực tế. Quy trình mã hóa này giải thích một số bất thường trong dữ liệu: giá trị âm ở `ticket_use` (−219 đến −1, tổng cộng 61 hàng) và `capacity` (giá trị −2), cũng như `occu_perc` vượt ngưỡng 100% (tối đa 147,5%) — không phải lỗi thu thập dữ liệu mà là kết quả của bước mã hóa.

**2. Ý nghĩa cột `show_time`:**
Cột `show_time` có giá trị từ 1 đến 60 với 51 giá trị duy nhất, nhưng **không phải là giờ đồng hồ** (0–23). Phân phối tập trung mạnh ở giá trị 1–8 (chiếm >95% số hàng), trong đó giá trị 3 xuất hiện nhiều nhất (28.785 lần). Cột này nhiều khả năng là **chỉ số suất chiếu trong ngày** (suất 1 = suất sáng sớm nhất, suất 2, 3... = các suất tiếp theo trong ngày), với một số rạp có nhiều hơn 8 suất/ngày (biểu diễn ở các giá trị 9–60).

**3. Chiến lược tổng hợp cho mô hình chuỗi thời gian:**
Do độ phân giải ở cấp suất chiếu, cần bước tiền xử lý **tổng hợp theo ngày** (`groupby('date').sum()`) để tạo chuỗi thời gian 234 điểm. Chuỗi tổng hợp có tính **mùa vụ rõ ràng**: doanh thu tăng dần từ Q1 (ít phim, mùa thấp điểm) đến đỉnh điểm Q2–Q3 (mùa phim hè), sau đó điều chỉnh vào Q4. Chuỗi 234 điểm là **chuỗi tương đối ngắn** đối với ARIMA — cần lựa chọn bậc (p, d, q) cẩn thận để tránh overfitting.

**4. Đặc trưng cho XGBoost:**
Với XGBoost, `film_code` và `cinema_code` là các đặc trưng phân loại quan trọng — cần **mã hóa (encoding)** trước khi đưa vào mô hình. Kết hợp `date`-derived features (`month`, `quarter`, `day`) với các đặc trưng mô tả suất chiếu (`show_time`, `ticket_price`, `capacity`, `occu_perc`) cho phép XGBoost khai thác các tương tác phức tạp giữa thời gian, rạp chiếu và bộ phim.

**5. Xử lý giá trị thiếu:**
`occu_perc` và `capacity` có **125 giá trị thiếu cùng vị trí** (cùng hàng), chiếm 0,09% tập dữ liệu — có thể được impute bằng median theo nhóm `cinema_code` hoặc loại bỏ nếu sử dụng ARIMA/Linear Regression trên chuỗi tổng hợp.

**6. Đơn vị `total_sales`:**
Đơn vị tiền tệ không được ghi rõ trong mô tả dataset. Dựa vào tương quan giữa `ticket_price` (mean ≈ 81.234), `tickets_sold` (mean ≈ 140) và `total_sales` (mean ≈ 12.347.280 ≈ 140 × 81.234 × (1 − tickets_out_ratio)), các giá trị nhất quán về mặt số học. Trong báo cáo, biến mục tiêu được ký hiệu theo đơn vị tiền tệ địa phương không xác định.

---

#### 2.2.3.4 Tổng quan Nghiên cứu Liên quan

**Bảng 2.30 — Các nghiên cứu liên quan đến dự báo doanh thu và tỷ lệ lấp đầy rạp chiếu phim**

| STT | Tác giả & Năm | Tạp chí / Hội nghị | Bối cảnh nghiên cứu | Phương pháp chính | Kết quả tiêu biểu | Ghi chú |
|---|---|---|---|---|---|---|
| 1 | Baranowski, Korczak & Zając (2020) | *Business Systems Research*, 11(1):73–88; 15 trích dẫn | Dự báo **lượng khán giả tại cấp độ suất chiếu** từ 179.103 suất phim tại **Ba Lan** — cùng độ phân giải với dataset này | Multiple Linear Regression + các mô hình dự báo ngắn hạn khác | Dự báo ngắn hạn (cập nhật tuần) cho suất chiếu với các đặc trưng: số màn chiếu, sức chứa rạp | ★ **Cùng độ phân giải** (show-level); gần nhất với cấu trúc dữ liệu này |
| 2 | Venkataramani, Ramesh, Jain & Kumar (2019) | *IEEE ICDM Workshops*; 4 trích dẫn | Xây dựng hệ thống dự báo **tỷ lệ lấp đầy rạp và số vé bán** (`occu_perc`, `tickets_sold`) cho từng suất chiếu — thích nghi động theo thời gian | Multiple Linear Regression, Neural Networks, và các mô hình ML khác | Các mô hình ML đạt hiệu suất tương đương Multiple Linear Regression cho bài toán dự báo tỷ lệ lấp đầy | ★ Dự báo **cùng biến mục tiêu** (`tickets_sold`, `occu_perc`) với dataset này |
| 3 | Tang (2024) | *PLoS ONE*, 19(10): e0309227; 15 trích dẫn | Dự báo **doanh thu phòng vé (box office)** từ đặc trưng marketing và phân phối phim; phân tích xu hướng thị trường phim theo thời gian | XGBoost tối ưu hóa (vs. DNN, CatBoost, LightGBM, RF, GBDT) | **Accuracy=0,90; F1=0,90; Precision=0,89; Recall=0,91** (tại n=2.500); XGBoost vượt trội tất cả baseline | Minh chứng mạnh nhất cho **XGBoost trong dự báo doanh thu phim** |
| 4 | Shahid & Islam (2023) | *PeerJ Computer Science*; 8 trích dẫn | Dự báo **thành công phòng vé** dựa trên đặc trưng độ phổ biến thể loại phim được trích xuất qua **chuỗi thời gian** | ARIMA (trích xuất đặc trưng xu hướng thể loại) + XGBoost, SVM, Gradient Boosting | Đặc trưng chuỗi thời gian từ ARIMA cải thiện đáng kể độ chính xác dự báo so với không có đặc trưng thời gian | Minh chứng cho kết hợp **ARIMA + ML** trong bài toán phim |
| 5 | Ampountolas & Legg (2021) | *International Journal of Contemporary Hospitality Management*, 33(6):2001; 72 trích dẫn | Dự báo **tỷ lệ lấp đầy phòng khách sạn** từ dữ liệu đặt chỗ và mạng xã hội — bài toán tương đồng với dự báo `occu_perc` | ARIMA (đặt chỗ nâng cao) → đầu vào XGBoost; so sánh với Genetic Algorithm và Linear Regression | Mô hình kết hợp ARIMA+XGBoost outperform các mô hình độc lập; ARIMA cung cấp tín hiệu cơ sở hiệu quả | Kết hợp **ARIMA + XGBoost** cho bài toán occupancy |
| 6 | Nguyen, Karg & Valadkhani (2022) | *Applied Economics* (Taylor & Francis); 24 trích dẫn | **Ứng dụng đầu tiên của XGBoost** cho bài toán dự báo lượng khán giả tại các sự kiện giải trí theo từng sự kiện riêng lẻ (*step-forward approach*) | XGBoost, Random Forest, Decision Tree, Logistic Regression — lựa chọn đặc trưng tiến (step-forward) | XGBoost vượt trội tất cả mô hình so sánh (cây quyết định, hồi quy) trong dự báo lượng khán giả | Đầu tiên áp dụng **XGBoost cho event attendance** — cùng lĩnh vực giải trí |
| 7 | Pulickakunnel Eldhose (2025) | *DIVA-portal* (Luận văn Thạc sĩ, Linköping University); 1 trích dẫn | So sánh **mô hình ensemble vs. mô hình truyền thống** cho bài toán dự báo doanh thu bán lẻ với hiệu ứng thời vụ và khuyến mãi | Random Forest, XGBoost (ensemble) vs. SARIMA (ARIMA mùa vụ) | XGBoost và Random Forest outperform SARIMA trong hầu hết các kịch bản; SARIMA vẫn cạnh tranh khi chuỗi có tính mùa vụ mạnh | So sánh trực tiếp **XGBoost vs. ARIMA** trong dự báo doanh số |

---

##### 2.2.3.4.1 Nghiên cứu Tiêu biểu

**Nghiên cứu trực tiếp liên quan nhất** là **Baranowski, Korczak & Zając (2020)** — đây là nghiên cứu hiếm hoi trong văn liệu khoa học thực hiện dự báo lượng khán giả ở **cùng độ phân giải với tập dữ liệu Cinema Tickets**: cấp độ từng suất chiếu (*individual show level*) thay vì tổng hợp theo ngày hay theo phim. Nghiên cứu sử dụng dữ liệu 179.103 suất phim tại Ba Lan, áp dụng mô hình **Multiple Linear Regression** với các đặc trưng về số màn chiếu và sức chứa rạp — tương đồng trực tiếp với các cột `show_time`, `capacity` và `cinema_code` trong tập dữ liệu hiện tại. Kết quả cho thấy dự báo ngắn hạn (cập nhật theo chương trình tuần) là khả thi và có giá trị thực tiễn cao.

**Venkataramani et al. (2019)** là nghiên cứu kỹ thuật quan trọng xây dựng hệ thống dự báo **cùng hai biến quan tâm nhất** trong tập dữ liệu Cinema Tickets: `tickets_sold` (số vé bán) và `occu_perc` (tỷ lệ lấp đầy rạp). Kết quả đáng chú ý: các mô hình học máy đạt hiệu suất **tương đương với Multiple Linear Regression** — cho thấy đặc trưng dữ liệu chiếu phim có thể được mô hình hóa tốt ngay cả với các phương pháp tuyến tính khi đặc trưng được lựa chọn phù hợp.

**Tang (2024)** cung cấp bằng chứng định lượng mạnh nhất cho lựa chọn XGBoost: **Accuracy=0,90, F1=0,90** — vượt trội DNN, CatBoost, LightGBM và Random Forest trên dữ liệu dự báo doanh thu phim. Đây là kết quả quan trọng để so sánh khi đánh giá hiệu suất của XGBoost trên tập dữ liệu Cinema Tickets.

---

##### 2.2.3.4.2 Phân tích Tổng hợp

Tổng hợp các nghiên cứu liên quan cho thấy ba xu hướng chính trong dự báo doanh thu và lượng khán giả rạp chiếu phim:

**① Về XGBoost:** Được xác nhận là phương pháp hiệu quả nhất cho bài toán dự báo doanh thu và lượng khán giả trong ngành giải trí (Tang 2024; Nguyen et al. 2022; Shahid & Islam 2023). XGBoost đặc biệt phù hợp với tập dữ liệu Cinema Tickets vì **dữ liệu có cấu trúc bảng với nhiều đặc trưng tương tác** (`film_code`, `cinema_code`, `show_time`, `ticket_price`, `capacity`, các đặc trưng thời gian) — đây chính là điều kiện XGBoost khai thác hiệu quả nhất thông qua quá trình boosting gradient.

**② Về ARIMA:** Được công nhận là phương pháp cơ sở (*baseline*) hiệu quả cho chuỗi thời gian tổng hợp. Trên dữ liệu Cinema Tickets (sau tổng hợp theo ngày), chuỗi 234 điểm có tính mùa vụ rõ ràng (Q2–Q3 cao điểm; Q1 thấp điểm) — điều kiện thuận lợi cho ARIMA. Shahid & Islam (2023) còn đề xuất sử dụng ARIMA để **trích xuất đặc trưng xu hướng** rồi đưa vào mô hình ML — hướng tiếp cận có thể bổ sung vào pipeline.

**③ Về Linear Regression:** Đóng vai trò mô hình cơ sở đơn giản nhất. Điểm đáng chú ý là kết quả của Venkataramani et al. (2019) cho thấy Linear Regression đạt hiệu suất **tương đương với các mô hình phức tạp hơn** cho bài toán dự báo tỷ lệ lấp đầy rạp chiếu — điều này gợi ý rằng các đặc trưng thời gian tuyến tính (`month`, `quarter`, `day`) có sức mạnh giải thích đáng kể trong ngữ cảnh này.

---

#### 2.2.3.5 Các Chỉ số Đánh giá

Bài toán hồi quy chuỗi thời gian — dự báo tổng doanh thu bán vé theo ngày hoặc theo suất chiếu — được đánh giá bằng các chỉ số hồi quy chuẩn. Do biến mục tiêu `total_sales` có phân phối **lệch phải rất mạnh** (mean 12,3 triệu >> median 3,7 triệu; max 1,26 tỷ ở cấp suất chiếu; max 21,3 tỷ ở cấp ngày), các chỉ số phần trăm (MAPE, SMAPE) đặc biệt quan trọng để đánh giá tương đối giữa các mô hình.

**Bảng 2.31 — Các chỉ số đánh giá mô hình hồi quy cho Cinema Tickets Dataset**

| STT | Chỉ số | Ký hiệu | Công thức | Ý nghĩa | Đặc điểm với dataset này |
|---|---|---|---|---|---|
| 1 | Trung bình sai số tuyệt đối | MAE | $\dfrac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$ | Sai số trung bình tuyệt đối (cùng đơn vị với `total_sales`) | Dễ diễn giải về mặt kinh doanh (sai số trung bình bao nhiêu đơn vị tiền tệ/ngày); ít nhạy với ngoại lệ |
| 2 | Căn bậc hai trung bình bình phương sai số | RMSE | $\sqrt{\dfrac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | Phạt mạnh các sai số lớn — nhạy cảm với ngày doanh thu đột biến | Quan trọng để phát hiện mô hình gặp khó khăn với các ngày cao điểm (max 21,3 tỷ/ngày); là chỉ số chính |
| 3 | Phần trăm sai số tuyệt đối trung bình | MAPE | $\dfrac{100\%}{n}\sum_{i=1}^{n}\left\|\dfrac{y_i - \hat{y}_i}{y_i}\right\|$ | Sai số tương đối (%) — cho phép so sánh trực tiếp giữa các mô hình | Cần loại trừ các ngày doanh thu rất nhỏ (min 180.000) để tránh MAPE phóng đại; dùng ngưỡng lọc nếu cần |
| 4 | Phần trăm sai số tuyệt đối đối xứng | SMAPE | $\dfrac{100\%}{n}\sum_{i=1}^{n}\dfrac{2\|y_i - \hat{y}_i\|}{|y_i| + |\hat{y}_i|}$ | Sai số đối xứng (%) — khắc phục nhược điểm MAPE khi $y$ nhỏ | Phù hợp hơn MAPE khi doanh thu có phạm vi rộng (phân phối lệch phải mạnh); ưu tiên sử dụng |
| 5 | Hệ số xác định | R² | $1 - \dfrac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Tỷ lệ phương sai được giải thích bởi mô hình (0–1) | R² gần 1 = mô hình tốt; tuy nhiên với chuỗi lệch phải mạnh, R² cao chưa đủ — cần kết hợp SMAPE để đánh giá toàn diện |

> **Lưu ý tính MAPE:** Ngày đầu tiên trong dataset (21/02/2018) chỉ có **1 suất chiếu** — tổng hợp theo ngày cho giá trị rất thấp (180.000) so với trung bình 7,5 tỷ. Khi tính MAPE trên chuỗi ngày, nên loại bỏ các ngày đặc biệt có ít hơn 10 suất chiếu (< 10 bản ghi) để tránh nhiễu. SMAPE vẫn là chỉ số tỷ lệ phần trăm ổn định hơn trong trường hợp này.

---

### Tài liệu Tham khảo

[1] Baranowski, P., Korczak, K., & Zając, J. (2020). "Forecasting cinema attendance at the movie show level: Evidence from Poland." *Business Systems Research: International journal of the Society for Advancing Innovation and Research in Economy*, 11(1), 73–88. URL: [https://hrcak.srce.hr/ojs/index.php/bsr/article/view/12668](https://hrcak.srce.hr/ojs/index.php/bsr/article/view/12668)

[2] Venkataramani, S., Ramesh, A., Jain, A. K., & Kumar, G. (2019). "A dynamically adaptive movie occupancy forecasting system with feature optimization." *2019 IEEE International Conference on Data Mining (ICDM) Workshops*. DOI: [10.1109/ICDMW.2019.00047](https://ieeexplore.ieee.org/abstract/document/8955583/)

[3] Tang, Y. (2024). "The box office prediction model based on the optimized XGBoost algorithm in the context of film marketing and distribution." *PLOS ONE*, 19(10): e0309227. DOI: [10.1371/journal.pone.0309227](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0309227)

[4] Shahid, M. H., & Islam, M. A. (2023). "Investigation of time series-based genre popularity features for box office success prediction." *PeerJ Computer Science*. DOI: [10.7717/peerj-cs.1603](https://peerj.com/articles/cs-1603/)

[5] Ampountolas, A., & Legg, M. P. (2021). "A segmented machine learning modeling approach of social media for predicting occupancy." *International Journal of Contemporary Hospitality Management*, 33(6), 2001–2023, Emerald. DOI: [10.1108/IJCHM-06-2020-0611](https://www.emerald.com/ijchm/article/33/6/2001/161110)

[6] Nguyen, H., Karg, A., & Valadkhani, A. (2022). "Predicting individual event attendance with machine learning: a 'step-forward' approach." *Applied Economics*, Taylor & Francis. DOI: [10.1080/00036846.2021.2003747](https://www.tandfonline.com/doi/abs/10.1080/00036846.2021.2003747)

[7] Pulickakunnel Eldhose, J. (2025). "Promotional Timing and Sales Forecasting: A Comparative Forecasting Study Using Ensemble and Traditional Models." *DIVA-portal* (Master's Thesis, Linköping University). URL: [https://www.diva-portal.org/smash/record.jsf?pid=diva2:1986632](https://www.diva-portal.org/smash/record.jsf?pid=diva2:1986632)
