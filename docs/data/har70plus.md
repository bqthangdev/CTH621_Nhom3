### 2.2.2 HAR70+ Dataset

---

#### 2.2.2.1 Giới thiệu Tập dữ liệu

**Tên tập dữ liệu:** HAR70+ (Human Activity Recognition 70+)  
**Nguồn:** UCI Machine Learning Repository — [https://archive.ics.uci.edu/dataset/780/har70](https://archive.ics.uci.edu/dataset/780/har70)  
**Tác giả tạo lập:** Aleksej Logacjov, Astrid Ustad — Norwegian University of Science and Technology (NTNU), Na Uy  
**Năm đóng góp lên UCI:** 2023 (dữ liệu thu thập năm 2021)  
**Giấy phép:** CC BY 4.0 (Creative Commons Attribution 4.0 International)  
**Bản chất dữ liệu:** Dữ liệu thực (*real-world*) — được thu thập và ghi chú chuyên nghiệp trong điều kiện phòng thí nghiệm thực địa  
**Tần suất thu thập:** 50 Hz (mỗi 20 mili-giây một bản ghi)  
**Nhiệm vụ học máy gốc:** Phân loại hoạt động thể chất (*Human Activity Recognition — HAR*); Classification  
**Biến mục tiêu:** `label` — mã hoạt động được chú thích thủ công từ video

Nhận dạng hoạt động thể chất từ dữ liệu cảm biến đeo tay (*wearable sensor-based HAR*) là một lĩnh vực nghiên cứu đang phát triển mạnh trong bối cảnh lão hóa dân số toàn cầu và nhu cầu ngày càng cao về giám sát sức khỏe người cao tuổi. Phần lớn các tập dữ liệu HAR được xây dựng từ đối tượng trẻ tuổi, khỏe mạnh — dẫn đến khoảng cách đáng kể về tính đại diện của mô hình khi triển khai cho nhóm người già và người có hạn chế vận động.

Tập dữ liệu HAR70+ được phát triển bởi nhóm nghiên cứu tại NTNU để lấp đầy khoảng trống đó: 18 đối tượng tham gia ở độ tuổi từ 70 đến 95, trải đều từ mức độ thể chất *fit* (linh hoạt) đến *frail* (yếu ớt), trong đó năm người sử dụng dụng cụ hỗ trợ đi lại (walking aids) trong quá trình thu thập. Mỗi đối tượng đeo hai cảm biến gia tốc kế 3 trục Axivity AX3: một gắn ở **lưng dưới** (*lower back*) và một gắn ở **đùi phải** (*right thigh*), thu thập liên tục trong khoảng 40 phút theo giao thức tự do có kiểm soát (*semi-structured free-living protocol*). Toàn bộ hoạt động được ghi lại bằng camera video gắn ngực và chú thích từng khung hình (*frame-by-frame annotation*).

Bài báo gốc của Ustad và cộng sự (2023) [1] xây dựng và kiểm định mô hình phân loại HAR70+, so sánh với các mô hình được huấn luyện trên đối tượng trẻ tuổi, và chứng minh sự cần thiết của dữ liệu chuyên biệt cho nhóm tuổi này.

**Phạm vi ứng dụng trong dự án:** Trong khuôn khổ dự án CTH621, tập dữ liệu HAR70+ được sử dụng cho bài toán **phân loại chuỗi thời gian** (Classification — Nhóm B), với mục tiêu nhận dạng hoạt động từ chuỗi tín hiệu gia tốc kế của 18 đối tượng. Tập dữ liệu cũng được sử dụng cho bài toán **phân cụm** (Clustering), nhằm xác định nhóm đối tượng có hành vi vận động tương đồng.

**Bảng 2.22 — Thông tin tổng quan tập dữ liệu HAR70+**

| Thuộc tính | Thông tin |
|:-----------|:----------|
| **Tên dataset** | HAR70+ (Human Activity Recognition 70+) |
| **Nguồn (UCI)** | [https://archive.ics.uci.edu/dataset/780/har70](https://archive.ics.uci.edu/dataset/780/har70) |
| **DOI dataset (UCI)** | [10.24432/C5CW3D](https://doi.org/10.24432/C5CW3D) |
| **Bài báo giới thiệu** | Ustad et al. (2023), *Sensors* 23(5), 2368 |
| **DOI bài báo** | [10.3390/s23052368](https://doi.org/10.3390/s23052368) |
| **Năm thu thập dữ liệu** | 2021 |
| **Giấy phép** | CC BY 4.0 |
| **Loại dữ liệu** | Chuỗi thời gian đa biến (*multivariate time-series*) — dữ liệu cảm biến sinh trắc học |
| **Tần suất thu thập** | 50 Hz (mỗi 20 ms một bản ghi) |
| **Thiết bị đo** | Axivity AX3 (gia tốc kế 3 trục) × 2 vị trí: lưng dưới, đùi phải |
| **Số bản ghi (tổng)** | 2.259.597 |
| **Số cột** | 8 (1 timestamp + 6 đặc trưng + 1 nhãn) |
| **Số đối tượng** | 18 (ID: 501–518), tuổi 70–95 |
| **Số lớp hoạt động** | 7 |
| **Giá trị thiếu** | Không có |

---

#### 2.2.2.2 Thống kê Số lượng Mẫu và Cấu trúc Dữ liệu

Mỗi tệp `.csv` tương ứng với một đối tượng tham gia (*participant*) và được đặt tên theo mã số từ `501.csv` đến `518.csv`. Không có tệp tổng hợp — người dùng cần tự nối (*concatenate*) khi huấn luyện toàn tập. Tổng số bản ghi sau khi nối toàn bộ 18 tệp là **2.259.597 dòng**, tương đương khoảng **37,7 giờ** dữ liệu cảm biến liên tục ở 50 Hz.

**Bảng 2.23 — Thống kê số lượng mẫu theo đối tượng**

| Mã đối tượng | Số bản ghi | Thời gian ghi xấp xỉ (phút) |
|:------------:|----------:|----------------------------:|
| 501 | 103.860 | ~34,6 |
| 502 | 131.367 | ~43,8 |
| 503 | 116.413 | ~38,8 |
| 504 | 150.758 | ~50,3 |
| 505 | 87.006 | ~29,0 |
| 506 | 122.714 | ~40,9 |
| 507 | 120.125 | ~40,0 |
| 508 | 130.494 | ~43,5 |
| 509 | 121.763 | ~40,6 |
| 510 | 122.061 | ~40,7 |
| 511 | 128.063 | ~42,7 |
| 512 | 119.310 | ~39,8 |
| 513 | 123.599 | ~41,2 |
| 514 | 101.510 | ~33,8 |
| 515 | 153.517 | ~51,2 |
| 516 | 138.278 | ~46,1 |
| 517 | 147.045 | ~49,0 |
| 518 | 141.714 | ~47,2 |
| **Tổng** | **2.259.597** | **~753,2** |

> **Ghi chú:** Thời gian ghi xấp xỉ được tính theo công thức: *n_rows / (50 Hz × 60 s/phút)*. Biến động thời gian giữa các đối tượng (từ ~29 phút đến ~51 phút) phản ánh sự linh hoạt của giao thức thực địa (*semi-structured free-living*) — không áp đặt thời lượng cứng nhắc mà để đối tượng thực hiện hoạt động theo điều kiện thực tế.

**Bảng 2.24 — Phân phối nhãn hoạt động (toàn bộ 18 đối tượng)**

| Mã nhãn | Tên hoạt động | Số bản ghi | Tỷ lệ (%) |
|:-------:|:-------------|----------:|----------:|
| 1 | Walking (đi bộ) | 1.079.312 | 47,77 |
| 3 | Shuffling (đi lê bước, chân không nhấc khỏi mặt đất) | 66.058 | 2,92 |
| 4 | Stairs ascending (leo cầu thang lên) | 4.560 | 0,20 |
| 5 | Stairs descending (leo cầu thang xuống) | 4.978 | 0,22 |
| 6 | Standing (đứng) | 418.055 | 18,50 |
| 7 | Sitting (ngồi) | 483.452 | 21,40 |
| 8 | Lying (nằm) | 203.182 | 8,99 |
| — | **Tổng** | **2.259.597** | **100,00** |

> **Nhận xét mất cân bằng lớp:** Tập dữ liệu có mức độ mất cân bằng lớp (*class imbalance*) nghiêm trọng. Hoạt động *walking* chiếm gần một nửa tổng số bản ghi (47,77%), trong khi hai hoạt động trên cầu thang (*stairs ascending* và *stairs descending*) chỉ chiếm lần lượt 0,20% và 0,22% — tức chưa đến 5.000 mẫu mỗi lớp. Sự mất cân bằng này phản ánh thực tế sinh hoạt của người cao tuổi, nhưng đặt ra thách thức đáng kể cho các mô hình phân loại: các thuật toán có xu hướng bỏ qua lớp thiểu số nếu không có kỹ thuật xử lý phù hợp (SMOTE, class weighting, hoặc cost-sensitive learning).

---

#### 2.2.2.3 Mô tả Đặc trưng

Mỗi bản ghi trong tập dữ liệu chứa 8 cột: một cột nhãn thời gian, sáu cột tín hiệu gia tốc kế dấu phẩy động (*floating-point*), và một cột nhãn hoạt động nguyên (*integer*). Không có cột gyroscope, từ kế hay áp suất — toàn bộ thông tin chuyển động được biểu diễn qua gia tốc theo ba trục không gian tại hai vị trí cơ thể.

**Bảng 2.25 — Mô tả đặc trưng của tập dữ liệu HAR70+**

| STT | Tên cột | Kiểu dữ liệu | Đơn vị | Vị trí cảm biến | Mô tả |
|:---:|:--------|:------------:|:------:|:---------------:|:------|
| 1 | `timestamp` | string | — | — | Dấu thời gian theo định dạng `YYYY-MM-DD HH:MM:SS.mmm`; khoảng cách trung bình giữa hai bản ghi liên tiếp ≈ 20 ms (50 Hz) |
| 2 | `back_x` | float64 | *g* | Lưng dưới | Gia tốc trục X hướng xuống dưới (*down*) của cảm biến lưng |
| 3 | `back_y` | float64 | *g* | Lưng dưới | Gia tốc trục Y hướng sang trái (*left*) của cảm biến lưng |
| 4 | `back_z` | float64 | *g* | Lưng dưới | Gia tốc trục Z hướng về phía trước (*forward*) của cảm biến lưng |
| 5 | `thigh_x` | float64 | *g* | Đùi phải | Gia tốc trục X hướng xuống dưới (*down*) của cảm biến đùi |
| 6 | `thigh_y` | float64 | *g* | Đùi phải | Gia tốc trục Y hướng sang phải (*right*) của cảm biến đùi |
| 7 | `thigh_z` | float64 | *g* | Đùi phải | Gia tốc trục Z hướng về phía sau (*backward*) của cảm biến đùi |
| 8 | `label` | int64 | — | — | Mã hoạt động được chú thích thủ công từ video; nhận các giá trị: 1, 3, 4, 5, 6, 7, 8 (không có nhãn 2) |

> **Lưu ý về đơn vị và quy ước trục:** Giá trị gia tốc được biểu diễn theo đơn vị *g* (gia tốc trọng lực, 1 *g* ≈ 9,81 m/s²). Quy ước trục tọa độ của hai cảm biến không đồng nhất: trục Y của cảm biến lưng hướng sang *trái* trong khi trục Y của cảm biến đùi hướng sang *phải*. Điều này phản ánh hướng gắn vật lý của từng thiết bị và cần được lưu ý khi thiết kế đặc trưng kết hợp hai cảm biến. Nhãn `2` không xuất hiện trong bộ dữ liệu — sơ đồ mã hóa mô tả 7 hoạt động với các mã không liên tiếp (1, 3–8).

---

#### 2.2.2.3.1 Vấn đề Chất lượng Dữ liệu

Tập dữ liệu HAR70+ được đánh giá có chất lượng tổng thể tốt so với các tập HAR công khai khác. Kiểm tra toàn bộ 18 tệp bằng Python (`pandas`) xác nhận **không có giá trị thiếu** trong bất kỳ cột nào. Tuy nhiên, có một số đặc điểm cần chú ý trong quá trình xử lý:

**Mất cân bằng lớp nghiêm trọng.** Như đã trình bày ở Bảng 2.24, tỷ lệ bản ghi giữa lớp đa số (*walking*: 47,77%) và lớp thiểu số (*stairs ascending*: 0,20%) lên tới xấp xỉ 237:1. Các mô hình phân loại thông thường (như Logistic Regression hay Decision Tree không điều chỉnh) sẽ bị kéo lệch về dự đoán lớp đa số, dẫn đến *accuracy* cao giả tạo trong khi *recall* cho lớp thiểu số gần bằng 0. Cần áp dụng chiến lược phù hợp: *class_weight='balanced'*, oversampling (SMOTE/ADASYN), hoặc đánh giá bằng macro-F1 thay vì accuracy.

**Biến động số lượng bản ghi giữa các đối tượng.** Thời lượng ghi dao động từ ~29 phút (đối tượng 505, 87.006 bản ghi) đến ~51 phút (đối tượng 515, 153.517 bản ghi). Sự biến động này có thể gây bất cân bằng thứ cấp khi mô hình học từ tập huấn luyện có cấu thành đối tượng không đều.

**Không có cột số thứ tự mẫu toàn cục.** Cột `timestamp` là chuỗi ký tự (*string*) và chỉ có ý nghĩa trong phạm vi từng tệp riêng. Khi nối dữ liệu, cần tạo chỉ số dòng mới và giải quyết khoảng cách thời gian giữa hai phiên ghi.

---

#### 2.2.2.4 Tổng quan Nghiên cứu Liên quan

##### 2.2.2.4.1 Bài báo giới thiệu tập dữ liệu

Ustad và cộng sự (2023) [1] trình bày quá trình xây dựng và kiểm định mô hình HAR70+ trong bài báo *"Validation of an Activity Type Recognition Model Classifying Daily Physical Behavior in Older Adults: The HAR70+ Model"*, đăng trên tạp chí **Sensors** (MDPI), tập 23, số 5, bài 2368. Đây là bài báo mở (*open access*) được tài trợ bởi Đại học Khoa học và Công nghệ Na Uy (NTNU).

**Mục tiêu và bối cảnh:** Nghiên cứu này nhằm giải quyết hạn chế căn bản của các mô hình HAR hiện có — được huấn luyện chủ yếu trên đối tượng trẻ tuổi, khỏe mạnh — khi áp dụng cho người cao tuổi (70+), nhóm đối tượng có tốc độ di chuyển chậm hơn, mẫu dáng bước (*gait pattern*) khác biệt, và thể trạng phân tầng từ *fit* đến *frail*. Năm trong số 18 đối tượng sử dụng gậy, khung tập đi hoặc nạng trong quá trình thu thập — phản ánh đặc thù của nhóm tuổi này.

**Phương pháp:** Nhóm tác giả trích xuất đặc trưng từ các cửa sổ thời gian cố định (*fixed-length time windows*) của tín hiệu gia tốc kế và huấn luyện mô hình học máy có giám sát. Quy trình kiểm định sử dụng **Leave-One-Subject-Out Cross-Validation (LOSO-CV)** — mỗi vòng lặp, một đối tượng được giữ lại hoàn toàn cho tập kiểm tra trong khi 17 đối tượng còn lại tạo thành tập huấn luyện. Phương pháp này đảm bảo tính độc lập hoàn toàn giữa tập huấn luyện và tập kiểm tra theo đơn vị cá nhân (*subject-independent evaluation*), phù hợp với yêu cầu khái quát hóa mô hình sang người dùng mới.

**So sánh với mô hình nền (*baseline*):** Nhóm tác giả đồng thời kiểm định mô hình HAR được huấn luyện trên đối tượng trẻ tuổi (từ tập dữ liệu Acti4) trên cùng tập dữ liệu HAR70+, nhằm định lượng mức độ giảm hiệu suất khi thiếu tính đại diện về độ tuổi. Kết quả cho thấy mô hình chuyên biệt cho người cao tuổi (HAR70+ model) vượt trội so với mô hình tổng quát, xác nhận giả thuyết của nhóm nghiên cứu.

**Kết quả:** Nghiên cứu báo cáo Accuracy 91% khi mô hình được đánh giá trên HARTH và 94% trên HAR70+. Đối với nhóm người dùng dụng cụ hỗ trợ đi lại, Accuracy tăng từ 87% lên 93% sau khi bổ sung dữ liệu huấn luyện phù hợp. Các kết quả này cho thấy thiết kế đánh giá theo đối tượng và tính đại diện của tập huấn luyện có ảnh hưởng trực tiếp đến khả năng khái quát hóa.

**Ý nghĩa:** Nghiên cứu đóng góp vào lĩnh vực *ambient assisted living (AAL)* và giám sát sức khỏe từ xa (*remote health monitoring*) cho người cao tuổi. Kết quả nghiên cứu có tiềm năng ứng dụng trong hệ thống phát hiện té ngã (*fall detection*), đánh giá mức độ hoạt động thể chất, và hỗ trợ phục hồi chức năng vận động.

---

#### 2.2.2.5 Chỉ số Đánh giá

Do bài toán gốc là **phân loại đa lớp** (*multi-class classification*) với mức độ mất cân bằng lớp cao, các chỉ số đánh giá phù hợp bao gồm:

**Bảng 2.26 — Chỉ số đánh giá cho bài toán phân loại hoạt động**

| Chỉ số | Ký hiệu | Mô tả | Ghi chú |
|:-------|:-------:|:------|:--------|
| Độ chính xác tổng thể | *Accuracy* | Tỷ lệ bản ghi được phân loại đúng trên toàn bộ tập kiểm tra | Không tin cậy khi lớp mất cân bằng; cần đánh giá kèm với chỉ số khác |
| F1-score trung bình macro | *Macro-F1* | Trung bình cộng F1 của từng lớp, không tính tỷ trọng tần suất | Phù hợp khi cần đánh giá bình đẳng các lớp thiểu số (stairs) |
| F1-score trung bình có trọng số | *Weighted-F1* | Trung bình F1 có trọng số theo tần suất từng lớp | Phản ánh hiệu suất tổng thể theo phân phối thực tế |
| Độ nhạy theo lớp | *Per-class Recall (Sensitivity)* | Tỷ lệ bản ghi của lớp *k* được nhận dạng đúng | Quan trọng đặc biệt với stairs ascending/descending |
| Độ đặc hiệu theo lớp | *Per-class Specificity* | Tỷ lệ bản ghi không thuộc lớp *k* được phân loại đúng là không thuộc *k* | Dùng trong bài báo gốc [1] cùng với sensitivity |
| Ma trận nhầm lẫn | *Confusion Matrix* | Bảng TP/FP/FN/TN cho từng cặp lớp | Hữu ích để xác định các lớp dễ bị nhầm lẫn (ví dụ: walking vs. shuffling) |

Các chỉ số trên phù hợp với nhiệm vụ phân loại hoạt động gốc. Trong pipeline CTH621 hiện hành, HAR70+ được dùng cho bài toán hồi quy `back_x` với phép chia theo thời gian; `label` được giữ để diễn giải và bị loại khỏi không gian đặc trưng của bài toán tương ứng. Vì khác nhiệm vụ, Accuracy của bài báo không được so sánh trực tiếp với MAE, RMSE hoặc \(R^2\) của dự án.

---

#### Tài liệu Tham khảo (mục 2.2.2)

[1] A. Ustad, A. Logacjov, S. Ø. Trollebø, P. Thingstad, B. Vereijken, K. Bach, and N. S. Maroni, "Validation of an Activity Type Recognition Model Classifying Daily Physical Behavior in Older Adults: The HAR70+ Model," *Sensors*, vol. 23, no. 5, p. 2368, 2023, doi: [10.3390/s23052368](https://doi.org/10.3390/s23052368).

---

**Trích dẫn BibTeX:**

```bibtex
@article{Ustad2023,
  author    = {Ustad, Astrid and Logacjov, Aleksej and Trolleb{\o}, Stine {\O}verengen
               and Thingstad, Pernille and Vereijken, Beatrix
               and Bach, Kerstin and Maroni, Nina Skj{\ae}ret},
  title     = {Validation of an Activity Type Recognition Model Classifying
               Daily Physical Behavior in Older Adults: The {HAR70+} Model},
  journal   = {Sensors},
  volume    = {23},
  number    = {5},
  pages     = {2368},
  year      = {2023},
  publisher = {MDPI},
  doi       = {10.3390/s23052368},
  url       = {https://doi.org/10.3390/s23052368}
}

@misc{Logacjov2023dataset,
  author       = {Logacjov, Aleksej and Ustad, Astrid},
  title        = {{HAR70+} [Dataset]},
  year         = {2023},
  howpublished = {UCI Machine Learning Repository},
  doi          = {10.24432/C5CW3D},
  url          = {https://doi.org/10.24432/C5CW3D}
}
```

**Trích dẫn trong văn bản (IEEE / APA style):**

- IEEE: `[1] A. Ustad et al., "Validation of an Activity Type Recognition Model Classifying Daily Physical Behavior in Older Adults: The HAR70+ Model," *Sensors*, vol. 23, no. 5, p. 2368, 2023, doi: 10.3390/s23052368.`

- APA: `Ustad, A., Logacjov, A., Trollebø, S. Ø., Thingstad, P., Vereijken, B., Bach, K., & Maroni, N. S. (2023). Validation of an activity type recognition model classifying daily physical behavior in older adults: The HAR70+ Model. *Sensors*, *23*(5), 2368. https://doi.org/10.3390/s23052368`

---

## Lưu ý về mã nhãn

Sơ đồ mã hóa hoạt động trong dữ liệu dùng cho dự án gồm các nhãn 1, 3, 4, 5, 6, 7 và 8. Nhãn 2 không xuất hiện trong bất kỳ tệp nguồn nào đã thu thập; báo cáo giữ nguyên mã gốc và không suy diễn nguyên nhân của khoảng trống này.
