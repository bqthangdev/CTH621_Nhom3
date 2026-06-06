### 2.2.3 PGCB Hourly Generation Dataset (Bangladesh)

---

#### 2.2.3.1 Giới thiệu Tập dữ liệu

**Tên tập dữ liệu:** PGCB Hourly Generation Dataset (Bangladesh)  
**Nguồn:** UCI Machine Learning Repository — [https://archive.ics.uci.edu/dataset/1175/pgcb+hourly+generation+dataset+(bangladesh)](https://archive.ics.uci.edu/dataset/1175/pgcb+hourly+generation+dataset+(bangladesh))  
**Tác giả tạo lập:** Power Grid Company of Bangladesh (PGCB)  
**Năm đóng góp lên UCI:** 2025 (dữ liệu thu thập từ tháng 4/2015 đến tháng 6/2025)  
**Giấy phép:** CC BY 4.0 (Creative Commons Attribution 4.0 International)  
**Bản chất dữ liệu:** Dữ liệu thực (*real-world*) — công bố chính thức bởi đơn vị vận hành lưới truyền tải quốc gia Bangladesh  
**Tần suất thu thập:** Chủ yếu hàng giờ (1 bản ghi/giờ); có thêm bản ghi 30 phút trong các khung giờ cao điểm  
**Nhiệm vụ học máy gốc:** Dự báo phụ tải điện (*load forecasting*); Regression  
**Biến mục tiêu (trong dự án):** `demand_mw` — tổng nhu cầu điện quốc gia theo giờ (đơn vị: MW)

Hệ thống điện Bangladesh là một trong những hệ thống lưới điện đang phát triển nhanh nhất ở Nam và Đông Nam Á. Công ty Lưới điện Bangladesh (*Power Grid Company of Bangladesh — PGCB*) là đơn vị vận hành lưới truyền tải quốc gia duy nhất, chịu trách nhiệm điều phối toàn bộ nguồn điện từ các nhà máy thuộc nhiều công nghệ khác nhau và phân phối đến các nhà cung cấp điện khu vực. Với công suất lắp đặt khoảng 25.700 MW tại thời điểm xuất bản dữ liệu, PGCB vận hành một lưới điện hỗn hợp đa nguồn bao gồm nhiệt điện khí, nhiệt điện dầu, nhiệt điện than, thủy điện, điện mặt trời, điện gió và điện nhập khẩu từ Ấn Độ và Nepal.

Tập dữ liệu được thu thập liên tục trong **hơn 10 năm** (từ tháng 4/2015 đến tháng 6/2025), ghi lại từng giờ tổng sản lượng điện, nhu cầu điện, lượng cắt tải (*load shedding*), và sản lượng chi tiết của từng nguồn phát. Đây là một trong số ít bộ dữ liệu vận hành lưới điện có tầm nhìn dài hạn, được công bố công khai ở cấp độ quốc gia từ một quốc gia đang phát triển, tạo cơ hội cho nghiên cứu dự báo phụ tải ngắn hạn và trung hạn trên dữ liệu thực tế.

Bài báo gốc của Islam, Turja và Habib (2025) [1] đề xuất một pipeline tiền xử lý và trích xuất đặc trưng kết hợp dữ liệu phụ tải, dữ liệu thời tiết và dữ liệu kinh tế, nhằm cải thiện độ chính xác dự báo phụ tải điện ngắn hạn và trung hạn cho Bangladesh.

**Phạm vi ứng dụng trong dự án:** Trong khuôn khổ dự án CTH621, tập dữ liệu được sử dụng cho bài toán **hồi quy chuỗi thời gian** (Regression — Nhóm B), với mục tiêu dự báo `demand_mw` theo thứ tự thời gian bằng các thuật toán Linear Regression, ARIMA/SARIMA và XGBoost. Tập dữ liệu cũng được sử dụng cho bài toán **phân cụm** (Clustering), nhằm xác định các nhóm thời điểm có hành vi tiêu thụ điện tương đồng.

**Bảng 2.27 — Thông tin tổng quan tập dữ liệu PGCB Hourly Generation**

| Thuộc tính | Thông tin |
|:-----------|:----------|
| **Tên dataset** | PGCB Hourly Generation Dataset (Bangladesh) |
| **Nguồn (UCI)** | [https://archive.uci.edu/dataset/1175/...](https://archive.ics.uci.edu/dataset/1175/pgcb+hourly+generation+dataset+(bangladesh)) |
| **DOI dataset (UCI)** | [10.24432/C59P6V](https://doi.org/10.24432/C59P6V) |
| **Bài báo giới thiệu** | Islam, Turja & Habib (2025), *Journal of Data, Information and Management* |
| **DOI bài báo** | [10.1007/s42488-025-00140-9](https://doi.org/10.1007/s42488-025-00140-9) |
| **Giai đoạn dữ liệu** | Tháng 4/2015 — Tháng 6/2025 (~10 năm) |
| **Giấy phép** | CC BY 4.0 |
| **Loại dữ liệu** | Chuỗi thời gian đa biến (*multivariate time-series*) — dữ liệu vận hành lưới điện |
| **Tần suất thu thập** | Chủ yếu hàng giờ; bổ sung bán giờ (30 phút) trong giờ cao điểm |
| **Số bản ghi (tổng)** | 92.650 |
| **Số cột** | 15 (1 timestamp + 13 đặc trưng số + 1 nhãn phân loại) |
| **Giá trị thiếu** | Có (xem mục 2.2.3.3.1) |

---

#### 2.2.3.2 Thống kê Số lượng Mẫu và Cấu trúc Chuỗi Thời gian

Tập dữ liệu được lưu trong một tệp CSV duy nhất (`PGCB_date_power_demand.csv`), với mỗi hàng là một quan sát theo thời gian. Toàn bộ 92.650 bản ghi trải dài **hơn 10 năm**, từ ngày 19/04/2015 đến ngày 17/06/2025. Phần lớn các bản ghi có khoảng cách 1 giờ (83.649 khoảng cách 1 giờ, chiếm ~90,5% tổng số), tuy nhiên tồn tại 8.336 khoảng cách 30 phút, tương ứng với các bản ghi bổ sung trong khung giờ cao điểm buổi chiều (thường trong khoảng 18:00–19:00) được đánh dấu qua cột `remarks` với giá trị "Evening_Peak" hoặc "Day_Peak".

**Bảng 2.28 — Thống kê số lượng mẫu và cấu trúc chuỗi thời gian**

| Thông tin | Giá trị |
|:----------|--------:|
| Tổng số bản ghi | 92.650 |
| Tổng số cột | 15 |
| Giai đoạn thời gian | 19/04/2015 — 17/06/2025 |
| Tần suất chủ yếu | 1 giờ/bản ghi (83.649 khoảng cách 1 giờ) |
| Khoảng cách 30 phút (giờ cao điểm) | 8.336 |
| Số mốc thời gian duy nhất | 92.218 |
| Bản ghi trùng mốc thời gian | 813 (liên quan đến 432 cặp trùng) |
| Số năm có dữ liệu đầy đủ | 9 (2016–2024, ~9.000–9.500 bản ghi/năm) |
| Bản ghi có load shedding > 0 | 14.808 (16,0%) |
| Nhu cầu điện trung bình (demand_mw) | ~8.813 MW (loại trừ outlier) |
| Nhu cầu điện nhỏ nhất / lớn nhất (dữ liệu sạch) | ~1.005 MW / ~20.587 MW |

> **Ghi chú về cấu trúc năm:** Năm 2015 chỉ có 6.659 bản ghi (bắt đầu từ tháng 4) và năm 2025 chỉ có 4.188 bản ghi (đến giữa tháng 6), trong khi các năm 2016–2024 đều có từ 8.486 đến 9.148 bản ghi — phù hợp với tần suất hàng giờ của 365 ngày × 24 giờ = 8.760 bản ghi lý thuyết mỗi năm đầy đủ.

---

#### 2.2.3.3 Mô tả Đặc trưng

Tập dữ liệu ghi lại tổng hợp vận hành lưới điện quốc gia Bangladesh theo giờ. Các cột được tổ chức thành bốn nhóm thông tin: **(1)** thời gian, **(2)** tổng hợp phụ tải và cắt tải, **(3)** sản lượng theo nguồn phát nội địa, và **(4)** nhập khẩu điện từ nước ngoài và nhãn phụ. Bảng 2.29 mô tả chi tiết từng cột.

**Bảng 2.29 — Mô tả đặc trưng của tập dữ liệu PGCB Hourly Generation**

| STT | Tên cột | Kiểu dữ liệu | Đơn vị | Nhóm | Mô tả |
|:---:|:--------|:------------:|:------:|:----:|:------|
| 1 | `datetime` | string | — | Thời gian | Dấu thời gian theo định dạng `YYYY-MM-DD HH:MM:SS`; chủ yếu ghi vào đầu mỗi giờ, bổ sung bản ghi :30 vào giờ cao điểm |
| 2 | `generation_mw` | float64 | MW | Tổng hợp | Tổng sản lượng điện được phát lên lưới (MW); *lưu ý:* có 1 giá trị cực đại bất thường (~64.526.500 MW ngày 2023-01-15 — rõ ràng là lỗi nhập liệu, xem mục 2.2.3.3.1) |
| 3 | `demand_mw` | int64 | MW | Tổng hợp | **Biến mục tiêu** — tổng nhu cầu điện quốc gia; thường trùng với `generation_mw` trừ khi có cắt tải hoặc lỗi dữ liệu |
| 4 | `load_shedding` | int64 | MW | Tổng hợp | Lượng phụ tải bị cắt do nguồn cung không đủ (MW); bằng 0 khi không có cắt tải |
| 5 | `gas` | int64 | MW | Nội địa | Sản lượng nhiệt điện khí |
| 6 | `liquid_fuel` | int64 | MW | Nội địa | Sản lượng nhiệt điện dầu (diesel, furnace oil, HFO) |
| 7 | `coal` | int64 | MW | Nội địa | Sản lượng nhiệt điện than |
| 8 | `hydro` | int64 | MW | Nội địa | Sản lượng thủy điện |
| 9 | `solar` | float64 | MW | Nội địa | Sản lượng điện mặt trời; có 22.133 giá trị `NaN` (23,9%) tương ứng với giai đoạn trước khi hạ tầng năng lượng mặt trời được triển khai hoặc trước khi PGCB bắt đầu theo dõi chỉ tiêu này |
| 10 | `wind` | float64 | MW | Nội địa | Sản lượng điện gió; có 73.974 giá trị `NaN` (79,8%) — điện gió Bangladesh còn ở giai đoạn đầu phát triển |
| 11 | `india_bheramara_hvdc` | int64 | MW | Nhập khẩu | Điện nhập khẩu từ Ấn Độ qua đường dây HVDC tại Bheramara |
| 12 | `india_tripura` | int64 | MW | Nhập khẩu | Điện nhập khẩu từ Ấn Độ qua đường dây xoay chiều phía Tripura |
| 13 | `india_adani` | float64 | MW | Nhập khẩu | Điện nhập khẩu từ tập đoàn Adani (Ấn Độ) theo hợp đồng mua bán điện; 85.312 giá trị `NaN` (92,1%) do hợp đồng ký kết và đường dây vận hành từ giai đoạn muộn |
| 14 | `nepal` | float64 | MW | Nhập khẩu | Điện nhập khẩu từ Nepal; 87.299 giá trị `NaN` (94,2%) do kết nối lưới Nepal–Bangladesh chỉ mới vận hành trong thời gian gần đây |
| 15 | `remarks` | string | — | Nhãn phụ | Nhãn phân loại thời điểm đặc biệt; nhận hai giá trị: `Day_Peak` hoặc `Evening_Peak`; 86.257 giá trị `NaN` (93,1%) — chỉ các mốc giờ cao điểm mới được gán nhãn |

> **Lưu ý về mối quan hệ giữa các cột:** Về lý thuyết, `generation_mw` = `gas` + `liquid_fuel` + `coal` + `hydro` + `solar` + `wind` + `india_bheramara_hvdc` + `india_tripura` + `india_adani` + `nepal`. Và `demand_mw` = `generation_mw` − `load_shedding`. Tuy nhiên, các ràng buộc này không được thỏa mãn hoàn toàn trong tập dữ liệu, một phần do giá trị `NaN` trong các cột nguồn phát và một phần do lỗi dữ liệu hiện diện.

---

#### 2.2.3.3.1 Vấn đề Chất lượng Dữ liệu

Mặc dù trang UCI ghi nhận "không có giá trị thiếu", phân tích tệp CSV thực tế cho thấy tập dữ liệu chứa một số vấn đề chất lượng đáng chú ý.

**Giá trị thiếu có cấu trúc (*structural missing values*).** Các cột `solar`, `wind`, `india_adani` và `nepal` có tỷ lệ `NaN` rất cao (từ 23,9% đến 94,2%). Đây không phải lỗi thu thập ngẫu nhiên mà là giá trị thiếu có cấu trúc: các nguồn phát điện và đường dây nhập khẩu này chưa được vận hành hoặc chưa được PGCB theo dõi trong phần lớn thời gian của chuỗi. Trong quá trình tiền xử lý, các cột này cần được điền giá trị 0 cho các giai đoạn trước khi vận hành, thay vì xử lý bằng imputation thông thường.

**Một ngoại lệ cực đoan trong `generation_mw`.** Ngày 2023-01-15 04:00:00 ghi nhận giá trị `generation_mw` là 64.526.500 MW — cao hơn 6.000 lần so với tổng công suất lắp đặt của Bangladesh (~25.700 MW). Giá trị `demand_mw` tương ứng là 6.452 MW (bình thường), cho thấy đây là lỗi nhập liệu có thể do thừa số 0 hoặc nhầm đơn vị (kW thay vì MW). Bản ghi này cần được loại bỏ hoặc điều chỉnh trước khi huấn luyện mô hình.

**Các giá trị `demand_mw` bất thường.** Có 22 bản ghi với `demand_mw` vượt quá 50.000 MW (cao hơn gấp đôi công suất lắp đặt tối đa), với giá trị lớn nhất lên đến 156.050 MW. Những bản ghi này nhiều khả năng là lỗi nhập liệu và cần được xử lý trong bước làm sạch dữ liệu.

**Dấu thời gian trùng lặp.** Có 813 bản ghi tham gia vào 432 cặp trùng mốc thời gian, tức là một số thời điểm có hai bản ghi với giá trị khác nhau. Nguyên nhân chưa rõ — có thể là bản ghi sửa đổi (amended reading) không xóa bản cũ. Cần chiến lược deduplication rõ ràng (ví dụ: giữ bản ghi có `generation_mw` cao hơn, hoặc giữ bản cuối theo thứ tự xuất hiện).

**Tần suất không đều.** Do bổ sung bản ghi 30 phút vào giờ cao điểm, chuỗi thời gian không có tần suất cố định (*non-uniform frequency*). Các mô hình yêu cầu đầu vào đều nhau (như ARIMA gốc) cần bước tái lấy mẫu (*resampling*) về tần suất 1 giờ trước khi huấn luyện.

**Bảng 2.30 — Tóm tắt vấn đề chất lượng dữ liệu**

| Vấn đề | Quy mô | Cột liên quan | Hướng xử lý đề xuất |
|:-------|-------:|:--------------|:--------------------|
| Giá trị thiếu có cấu trúc (NaN do chưa vận hành) | 22.133–87.299 bản ghi | `solar`, `wind`, `india_adani`, `nepal` | Điền 0 cho giai đoạn trước vận hành; kiểm tra ngày bắt đầu của từng nguồn |
| Ngoại lệ cực đoan `generation_mw` | 1 bản ghi | `generation_mw` | Loại bỏ hoặc điều chỉnh theo `demand_mw` tương ứng |
| Giá trị `demand_mw` bất thường (>50.000 MW) | ~22 bản ghi | `demand_mw` | Loại bỏ hoặc đánh dấu outlier |
| Dấu thời gian trùng lặp | 813 bản ghi | `datetime` | Deduplication theo tiêu chí nhất quán |
| Tần suất không đều (1h + 30min) | 8.336 khoảng cách 30 phút | `datetime` | Resampling về tần suất 1 giờ (`resample('1h').mean()` hoặc `.last()`) |

---

#### 2.2.3.4 Tổng quan Nghiên cứu Liên quan

##### 2.2.3.4.1 Bài báo giới thiệu tập dữ liệu

Islam, Turja và Habib (2025) [1] trình bày phương pháp dự báo phụ tải trong bài báo *"Enhanced power demand forecasting for Bangladesh: using feature engineering associated with environmental and economic impact"*, đăng trên tạp chí **Journal of Data, Information and Management** (Springer), năm 2025. Đây là bài báo tạo lập tập dữ liệu PGCB và đặt nền tảng lý thuyết cho các nghiên cứu dự báo phụ tải điện Bangladesh.

**Bối cảnh và động lực:** Bangladesh là quốc gia đang phát triển với tốc độ tăng trưởng phụ tải điện cao, song dữ liệu vận hành lưới điện hiếm khi được công bố công khai. Các nghiên cứu dự báo phụ tải ngắn hạn (*short-term load forecasting — STLF*) hiện có chủ yếu tập trung vào các nền kinh tế phát triển với lưới điện ổn định; Bangladesh với đặc thù cắt tải thường xuyên (*load shedding*) và cơ cấu nguồn điện đa dạng (khí, dầu, than, nhập khẩu) đặt ra những thách thức riêng biệt chưa được nghiên cứu đầy đủ.

**Phương pháp đề xuất:** Nhóm tác giả đề xuất một **pipeline tiền xử lý và trích xuất đặc trưng** kết hợp ba nguồn dữ liệu: (1) dữ liệu phụ tải từ PGCB, (2) dữ liệu thời tiết (*environmental features* — nhiệt độ, độ ẩm, lượng mưa), và (3) dữ liệu kinh tế (*economic features* — chỉ số giá tiêu dùng, tốc độ tăng trưởng GDP). Phương pháp tạo ra bộ đặc trưng phong phú gồm: đặc trưng thời gian (*temporal features*: giờ trong ngày, ngày trong tuần, tháng, ngày lễ), đặc trưng trễ (*lag features*: phụ tải giờ trước, hôm trước, tuần trước), và đặc trưng trung bình động (*rolling mean features*). Kết hợp đặc trưng đa nguồn này nhằm cải thiện dự báo cả ngắn hạn (giờ tiếp theo) lẫn trung hạn (nhiều ngày đến vài tuần).

**Kết quả:** Mô hình đề xuất đạt **MAPE 2,3%** trên tập dữ liệu PGCB khi dự báo phụ tải tháng 1 [1]. Đây là kết quả tốt nhất được báo cáo trong nghiên cứu, phản ánh hiệu quả của chiến lược kết hợp đặc trưng đa nguồn so với các baseline chỉ sử dụng dữ liệu phụ tải lịch sử đơn thuần. Chi tiết đầy đủ về bảng so sánh các mô hình, kết quả RMSE/MAE theo từng kịch bản thời gian và phân tích tầm quan trọng đặc trưng đang chờ xác nhận từ toàn văn bài báo (xem Ghi chú).

**Ý nghĩa:** Nghiên cứu đặt nền tảng cho các hệ thống điều phối lưới điện thông minh (*smart grid dispatch*) tại Bangladesh, hỗ trợ lập kế hoạch vận hành ngắn hạn và tối ưu hóa cơ cấu nguồn điện để giảm chi phí và cắt tải.

---

#### 2.2.3.5 Chỉ số Đánh giá

Bài toán dự báo phụ tải điện là bài toán **hồi quy chuỗi thời gian** (*time-series regression*), với biến mục tiêu là `demand_mw` — một biến liên tục có tính chu kỳ theo giờ trong ngày, ngày trong tuần và mùa trong năm. Các chỉ số đánh giá phù hợp bao gồm:

**Bảng 2.31 — Chỉ số đánh giá cho bài toán dự báo phụ tải điện**

| Chỉ số | Ký hiệu | Công thức | Mô tả | Ghi chú |
|:-------|:-------:|:---------:|:------|:--------|
| Mean Absolute Error | *MAE* | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Sai số tuyệt đối trung bình (MW) | Trực quan, không phạt nặng ngoại lệ |
| Root Mean Squared Error | *RMSE* | $\sqrt{\frac{1}{n}\sum(y_i-\hat{y}_i)^2}$ | Căn trung bình bình phương sai số (MW) | Phạt nặng hơn với sai số lớn; nhạy với outlier |
| Mean Absolute Percentage Error | *MAPE* | $\frac{100}{n}\sum\left|\frac{y_i-\hat{y}_i}{y_i}\right|$ | Sai số phần trăm trung bình (%) | Cho phép so sánh tương đối giữa các hệ thống điện quy mô khác nhau; bài báo [1] sử dụng MAPE là chỉ số chính |
| Coefficient of Determination | $R^2$ | $1 - \frac{SS_{res}}{SS_{tot}}$ | Tỷ lệ phương sai được giải thích bởi mô hình | Từ 0 đến 1; giá trị gần 1 nghĩa là mô hình khớp tốt với dữ liệu |

Trong khuôn khổ dự án CTH621, **RMSE và MAPE** là hai chỉ số ưu tiên, phù hợp với thông lệ trong lĩnh vực dự báo phụ tải điện. $R^2$ được báo cáo bổ sung. Phân chia dữ liệu theo phương pháp **Chronological Split**: dữ liệu năm 2015–2023 dùng làm tập huấn luyện, năm 2024 dùng làm tập kiểm tra — đảm bảo không có rò rỉ thông tin tương lai vào quá trình học.

---

#### Tài liệu Tham khảo (mục 2.2.3)

[1] M. T. Islam, S. A. Turja, and A. Habib, "Enhanced power demand forecasting for Bangladesh: using feature engineering associated with environmental and economic impact," *Journal of Data, Information and Management*, Springer, 2025, doi: [10.1007/s42488-025-00140-9](https://doi.org/10.1007/s42488-025-00140-9).

---

**Trích dẫn BibTeX:**

```bibtex
@article{Islam2025,
  author    = {Islam, Muhammad Tanveer and Turja, Sartaj Aziz and Habib, Ahsan},
  title     = {Enhanced power demand forecasting for {Bangladesh}: using feature
               engineering associated with environmental and economic impact},
  journal   = {Journal of Data, Information and Management},
  year      = {2025},
  publisher = {Springer},
  doi       = {10.1007/s42488-025-00140-9},
  url       = {https://doi.org/10.1007/s42488-025-00140-9}
}

@misc{PGCB2025dataset,
  author       = {{Power Grid Company of Bangladesh (PGCB)}},
  title        = {{PGCB} Hourly Generation Dataset ({Bangladesh}) [Dataset]},
  year         = {2025},
  howpublished = {UCI Machine Learning Repository},
  doi          = {10.24432/C59P6V},
  url          = {https://doi.org/10.24432/C59P6V}
}
```

**Trích dẫn trong văn bản (IEEE / APA style):**

- IEEE: `[1] M. T. Islam, S. A. Turja, and A. Habib, "Enhanced power demand forecasting for Bangladesh: using feature engineering associated with environmental and economic impact," *Journal of Data, Information and Management*, Springer, 2025, doi: 10.1007/s42488-025-00140-9.`

- APA: `Islam, M. T., Turja, S. A., & Habib, A. (2025). Enhanced power demand forecasting for Bangladesh: using feature engineering associated with environmental and economic impact. *Journal of Data, Information and Management*. Springer. https://doi.org/10.1007/s42488-025-00140-9`

---

## Ghi chú cần bổ sung

**[GHI CHÚ 1 — Kết quả đầy đủ từ bài báo gốc]**

Trang Springer yêu cầu đăng nhập để truy cập toàn văn bài báo [1] — không thể trích xuất tự động tại thời điểm soạn thảo. Kết quả duy nhất truy xuất được từ đoạn trích Google Scholar là **MAPE = 2,3%** trên tập kiểm tra tháng 1. Các thông tin sau cần được bổ sung thủ công sau khi đọc toàn văn:

- Mô hình học máy được sử dụng (Random Forest, LSTM, XGBoost, ...?)
- Bảng so sánh hiệu suất: RMSE, MAE, $R^2$ theo từng mô hình và horizon dự báo
- Danh sách cụ thể các đặc trưng thời tiết và kinh tế được tích hợp
- Chiến lược phân chia tập huấn luyện/kiểm tra trong nghiên cứu gốc
- Kết quả dự báo trung hạn (nhiều ngày / nhiều tuần)

**[GHI CHÚ 2 — Năm xuất bản và tên tạp chí]**

Bài báo [1] được xác nhận đăng trên **Journal of Data, Information and Management** (Springer) năm 2025 (từ Google Scholar và trang UCI). Tuy nhiên, số tập và số trang cụ thể chưa truy xuất được — cần bổ sung khi có quyền truy cập toàn văn.
