### 2.3.2 Heartbeat Sounds (PASCAL CHSC 2011)

---

#### 2.3.2.1 Giới thiệu Tập dữ liệu

**Tên tập dữ liệu:** Heartbeat Sounds (PASCAL Classifying Heart Sounds Challenge 2011)  
**Nguồn Kaggle:** [https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds)  
**Nguồn gốc:** PASCAL Classifying Heart Sounds Challenge 2011 (CHSC2011)  
**Tác giả thách thức gốc:** Peter Bentley, Glenn Nordehn, Miguel Coimbra, Shie Mannor, Rita Getz  
**Tổ chức bảo trợ:** PASCAL Network  
**Giấy phép:** CC0: Public Domain  
**Bản chất dữ liệu:** Dữ liệu thực (*real-world*) — thu thập từ hai nguồn độc lập: cộng đồng và cơ sở lâm sàng  
**Nhiệm vụ học máy gốc:** (1) Phân vùng âm thanh tim (*heart sound segmentation*) và (2) Phân loại âm thanh tim (*heart sound classification*)  
**Biến mục tiêu:** Nhãn loại âm thanh tim (`normal`, `murmur`, `extrahls`/`extrastole`, `artifact`)

Bệnh tim mạch là nguyên nhân tử vong hàng đầu toàn cầu — theo Tổ chức Y tế Thế giới (WHO), ước tính 17,1 triệu người tử vong do bệnh tim mạch năm 2004, chiếm 29% tổng số ca tử vong. Nghe tim bằng ống nghe (*cardiac auscultation*) là một kỹ năng thăm khám lâm sàng không xâm lấn, rẻ tiền và phổ biến — nhưng độ chính xác của phương pháp này phụ thuộc rất lớn vào năng lực và kinh nghiệm của bác sĩ. Các nghiên cứu chỉ ra rằng khả năng nhận dạng tiếng thổi tâm thu (*systolic murmur*) thông qua nghe tim có độ đồng thuận khá thấp giữa các bác sĩ (Cohen's κ = 0,30–0,48), và độ chính xác còn kém hơn đối với các bệnh lý tim khác [2]. Điều này đặt ra nhu cầu phát triển hệ thống tự động hỗ trợ sàng lọc bệnh lý tim mạch thông qua phân tích âm thanh tim.

Tập dữ liệu **Heartbeat Sounds** được xây dựng và công bố trong khuôn khổ thách thức học máy **PASCAL Classifying Heart Sounds Challenge 2011 (CHSC2011)** [1], nhằm thu hút cộng đồng nghiên cứu vào hai bài toán liên quan: (1) định vị âm thanh tim S1/S2 trong tín hiệu thô, và (2) phân loại bản ghi âm thành các nhóm bệnh lý. Điểm đặc biệt của bộ dữ liệu này là sự kết hợp giữa **hai nguồn thu thập hoàn toàn khác nhau** về thiết bị, đối tượng và điều kiện ghi âm, phản ánh cả môi trường y tế bình dân (*community setting*) lẫn môi trường lâm sàng chuyên nghiệp (*clinical setting*). Dữ liệu này sau đó được phổ biến công khai qua nền tảng Kaggle (kinguistics, 2014) và hiện là một trong những bộ dữ liệu âm thanh tim được trích dẫn nhiều nhất trong lĩnh vực phân loại tín hiệu sinh học.

**Phạm vi ứng dụng trong dự án:** Trong khuôn khổ dự án CTH621, tập dữ liệu Heartbeat Sounds được sử dụng cho bài toán **phân loại âm thanh** (Classification — Nhóm C), với mục tiêu phân loại bản ghi âm thanh tim vào các nhóm bệnh lý. Tập dữ liệu cũng được sử dụng cho bài toán **phân cụm** (Clustering), nhằm xác định các nhóm bản ghi có đặc trưng âm học tương đồng.

**Bảng 2.35 — Thông tin tổng quan tập dữ liệu Heartbeat Sounds**

| Thuộc tính | Thông tin |
|:-----------|:----------|
| **Tên dataset** | Heartbeat Sounds (PASCAL CHSC 2011) |
| **Nguồn Kaggle** | [https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds) |
| **Nguồn gốc thách thức** | PASCAL CHSC 2011 — [http://www.peterjbentley.com/heartchallenge/](http://www.peterjbentley.com/heartchallenge/) |
| **Giấy phép** | CC0: Public Domain |
| **Loại dữ liệu** | Âm thanh tim PCG (*Phonocardiogram*) — định dạng WAV, đơn kênh |
| **Số tệp WAV (tổng)** | 832 |
| **Nguồn thu thập** | Set A: iPhone app (công cộng) · Set B: DigiScope tại bệnh viện (lâm sàng) |
| **Tần số lấy mẫu** | Set A: 44.100 Hz · Set B: 4.000 Hz |
| **Độ dài mỗi bản ghi** | 1 giây đến 30 giây (biến động theo từng bản ghi) |
| **Số lớp phân loại** | Set A: 4 lớp · Set B: 3 lớp |
| **Dữ liệu định vị S1/S2** | Có — `set_a_timing.csv` (390 mốc định vị) |

---

#### 2.3.2.2 Thống kê Số lượng Mẫu và Phân phối Nhãn

Tập dữ liệu được chia thành hai tập con riêng biệt phản ánh hai nguồn thu thập khác nhau, mỗi tập lưu trong thư mục riêng (`set_a/` và `set_b/`) kèm theo tệp metadata CSV tương ứng. Tổng cộng có **832 tệp WAV**, trong đó 176 tệp thuộc Set A và 656 tệp thuộc Set B. Một phần trong mỗi tập không có nhãn (tập kiểm tra ẩn của thách thức gốc — 52 tệp ở Set A và 195 tệp ở Set B), cùng với tập huấn luyện đã được gán nhãn đầy đủ.

**Bảng 2.36 — Phân phối nhãn theo từng tập dữ liệu**

| Nhãn | Set A | Mô tả | Set B | Mô tả |
|:-----|------:|:------|------:|:------|
| `normal` | 31 | Tiếng tim bình thường | 320 | Tiếng tim bình thường (bao gồm 120 bản ghi `noisynormal`) |
| `murmur` | 34 | Tiếng thổi tim (*heart murmur*) | 95 | Tiếng thổi tim (bao gồm 29 bản ghi `noisymurmur`) |
| `extrahls` | 19 | Âm phụ thêm (*extra heart sound*) | — | Không có nhóm này trong Set B |
| `extrastole` | — | Không có trong Set A | 46 | Ngoại tâm thu (*extrasystole*) |
| `artifact` | 40 | Tạp âm / không có tiếng tim | — | Không có nhóm này trong Set B |
| Chưa gán nhãn | 52 | Tập kiểm tra ẩn | 195 | Tập kiểm tra ẩn |
| **Tổng** | **176** | | **656** | |

> **Lưu ý về nhãn phụ (`sublabel`):** Tệp `set_b.csv` có thêm cột `sublabel` — 149 trong số 656 bản ghi của Set B có nhãn phụ, gồm `noisynormal` (120 bản ghi) và `noisymurmur` (29 bản ghi). Đây là các bản ghi có lượng tiếng ồn nền đáng kể nhưng vẫn xác định được loại âm thanh tim. Trong thách thức gốc, nhãn phụ có thể được dùng hoặc bỏ qua tùy chiến lược xử lý của người tham dự.

---

#### 2.3.2.3 Cấu trúc Tệp và Đặc điểm Tín hiệu

##### 2.3.2.3.1 Tổ chức Tệp và Metadata

Tập dữ liệu được tổ chức gồm ba tệp CSV và hai thư mục âm thanh:

- **`set_a/`**: 176 tệp WAV, ghi âm qua ứng dụng iStethoscope Pro trên iPhone. Tên tệp theo dạng `<nhãn>__<timestamp>.wav` (ví dụ: `normal__201102081321.wav`, `artifact__201012172012.wav`), phản ánh trực tiếp nhãn phân loại trong tên tệp.
- **`set_b/`**: 656 tệp WAV, ghi âm qua ống nghe kỹ thuật số DigiScope tại bệnh viện. Tên tệp theo dạng `Btraining_<nhãn>_<id>_<timestamp>_<vị trí>.wav` (ví dụ: `Btraining_extrastole_127_1306764300147_C2.wav`).
- **`set_a.csv`**: 176 bản ghi, 4 cột (`dataset`, `fname`, `label`, `sublabel`) — nhãn phân loại và đường dẫn tệp.
- **`set_b.csv`**: 656 bản ghi, 4 cột — nhãn phân loại, đường dẫn tệp và nhãn phụ nhiễu.
- **`set_a_timing.csv`**: 390 bản ghi, 4 cột (`fname`, `cycle`, `sound`, `location`) — dữ liệu vàng chuẩn (*gold standard*) định vị âm thanh S1 và S2 theo số mẫu tín hiệu cho các bản ghi `normal` của Set A. Đây là dữ liệu phục vụ bài toán phân vùng (Challenge 1).

##### 2.3.2.3.2 Đặc điểm Tín hiệu theo Nguồn

Hai tập con có sự khác biệt căn bản về tần số lấy mẫu, phản ánh đặc điểm thiết bị:

**Set A (iStethoscope Pro — iPhone, 44.100 Hz):** Đây là tần số lấy mẫu âm thanh tiêu dùng phổ biến, phù hợp cho ghi âm âm nhạc và giọng nói, nhưng quá cao so với nhu cầu phân tích tiếng tim — vốn chứa phần lớn thông tin hữu ích ở dải tần thấp dưới 195 Hz. Bản ghi từ Set A có nhiều biến động về chất lượng, do người dùng phổ thông tự thực hiện không qua đào tạo: nhiều tệp chứa tiếng ồn nền (giao thông, nhạc, lời nói), tiếng cọ vải vào micro, hoặc tiếng thở. Tuy nhiên, chính sự đa dạng về điều kiện này làm cho Set A có giá trị cao trong đánh giá tính bền vững (*robustness*) của mô hình.

**Set B (DigiScope — lâm sàng, 4.000 Hz):** Tần số lấy mẫu 4.000 Hz đủ để biểu diễn tín hiệu tiếng tim dưới 2.000 Hz, tương thích với chuẩn thiết bị y tế. Bản ghi được thực hiện bởi nhân viên y tế trong điều kiện bệnh viện kiểm soát, nên chất lượng tín hiệu cao hơn đáng kể so với Set A. Tuy nhiên, tập vẫn chứa nhóm `noisynormal` và `noisymurmur` nhằm phản ánh thực tế lâm sàng, nơi tiếng ồn từ vận động bệnh nhân, hô hấp và môi trường bệnh phòng không thể loại bỏ hoàn toàn.

**Đặc điểm âm học của các nhóm nhãn.** Tiếng tim bình thường (*normal*) có nhịp rõ ràng với hai âm thanh chính S1 ("lub") và S2 ("dub"), khoảng cách từ S1 đến S2 ngắn hơn khoảng cách từ S2 đến S1 tiếp theo. Tiếng thổi tim (*murmur*) xuất hiện dưới dạng tạp âm "whooshing" hoặc "turbulent" ở khoảng giữa S1–S2 hoặc S2–S1, có thể là dấu hiệu của nhiều bệnh lý tim nghiêm trọng. Âm phụ thêm (*extra heart sound*) tạo ra nhịp điệu ba âm điển hình như "lub-lub dub" hoặc "lub dub-dub". Ngoại tâm thu (*extrasystole*) tương tự âm phụ nhưng xảy ra không đều và không theo chu kỳ cố định. Cuối cùng, nhóm `artifact` chứa toàn bộ âm thanh không phải tiếng tim: tiếng rít thiết bị, tiếng nói, nhạc, và tạp âm — không có chu kỳ tim nhận dạng được.

---

#### 2.3.2.3.3 Vấn đề Chất lượng Dữ liệu

Tập dữ liệu Heartbeat Sounds đặt ra một số thách thức chất lượng đáng kể, phần lớn xuất phát từ đặc thù thu thập trong điều kiện không kiểm soát:

**Mất cân bằng lớp đáng kể.** Trong Set B (tập lớn hơn và thường được ưu tiên), nhóm `normal` chiếm 320/461 bản ghi có nhãn (~69,4%), trong khi `extrastole` chỉ có 46 bản ghi (~10%). Cùng với đó, Set A có sự mất cân bằng giữa `artifact` (40) và `extrahls` (19). Mức độ mất cân bằng này đòi hỏi chiến lược xử lý phù hợp khi huấn luyện mô hình.

**Sự dị biệt về tần số lấy mẫu giữa hai tập.** Sự khác biệt 11 lần (44.100 Hz vs 4.000 Hz) giữa Set A và Set B có nghĩa là cùng một chuỗi tiền xử lý không thể áp dụng đồng nhất cho cả hai tập. Khi kết hợp hai tập để huấn luyện mô hình chung, bắt buộc phải thực hiện tái lấy mẫu (*resampling*) về một tần số mục tiêu đồng nhất trước.

**Nhiễu nền đa dạng.** Tiếng tim bị che khuất bởi nhiều loại nhiễu — tiếng thở, tiếng quần áo cọ sát, tiếng ồn môi trường — đặc biệt trong Set A do điều kiện thu thập không kiểm soát. Nhóm nhãn phụ `noisynormal` và `noisymurmur` trong Set B phản ánh một phần vấn đề này, nhưng không bao phủ toàn bộ sự đa dạng của nhiễu trong Set A.

**Không có nhãn cho tập kiểm tra ẩn.** 52 tệp (Set A) và 195 tệp (Set B) không có nhãn trong các tệp CSV — đây là tập kiểm tra được giữ kín cho thách thức CHSC2011. Khi sử dụng toàn bộ thư mục audio, cần lọc chỉ giữ lại các bản ghi có nhãn nếu muốn huấn luyện mô hình học có giám sát.

---

#### 2.3.2.4 Tổng quan Nghiên cứu Liên quan

##### 2.3.2.4.1 Nguồn gốc và Bối cảnh Dữ liệu

Thách thức PASCAL CHSC 2011 được tổ chức bởi Bentley và cộng sự [1], tập trung vào hai bài toán đặc thù của phân tích tiếng tim: phân vùng để định vị S1 và S2, và phân loại bản ghi thành các nhóm bệnh lý. Kết quả và dữ liệu của thách thức này đặt nền móng cho một thập kỷ nghiên cứu phân tích tiếng tim tự động, và bộ dữ liệu PASCAL tiếp tục được trích dẫn rộng rãi trong các nghiên cứu đến năm 2024–2025 [2].

Theo tổng quan hệ thống của Zhao và cộng sự (2024) [2] — bài đánh giá toàn diện về học sâu trong phân tích tiếng tim từ 2010 đến 2024 — bộ dữ liệu PASCAL là một trong bốn bộ dữ liệu công khai được sử dụng nhiều nhất trong lĩnh vực, bên cạnh CinC2016, CinC2022 và bộ dữ liệu Yaseen. Điều này xác nhận tính đại diện và giá trị học thuật của dữ liệu được sử dụng trong dự án.

##### 2.3.2.4.2 Phương pháp Trích xuất Đặc trưng Truyền thống

Trước khi học sâu trở nên phổ biến, các hệ thống phân loại tiếng tim dựa trên quy trình hai giai đoạn: trích xuất đặc trưng thủ công từ tín hiệu âm thanh, sau đó đưa vào bộ phân loại học máy cổ điển.

**Đặc trưng miền thời gian-tần số.** Phép biến đổi Fourier thời gian ngắn (*Short-Time Fourier Transform — STFT*) là phương pháp nền tảng, biến đổi tín hiệu một chiều thành biểu diễn hai chiều thời gian–tần số, từ đó có thể hình dung và đo lường thành phần tần số theo thời gian. Tuy nhiên, STFT có độ phân giải thời gian–tần số cố định, không phù hợp cho tín hiệu tiếng tim có thành phần tần số biến đổi không đều theo chu kỳ tim. Phép biến đổi wavelet liên tục (*Continuous Wavelet Transform — CWT*) khắc phục hạn chế này bằng cách cung cấp độ phân giải thay đổi — phân giải tần số cao hơn ở dải tần thấp và phân giải thời gian cao hơn ở dải tần cao — phù hợp hơn với đặc tính phi cố định (*non-stationary*) của tín hiệu tiếng tim [2].

**Mel Frequency Cepstral Coefficients (MFCC).** Được mượn từ lĩnh vực nhận dạng giọng nói, MFCC là đặc trưng được sử dụng nhiều nhất trong phân loại tiếng tim [2]. Quy trình tính MFCC bao gồm: chia tín hiệu thành khung ngắn, áp dụng cửa sổ (*windowing*), tính phổ Mel (thang tần số phi tuyến mô phỏng cảm nhận thính giác của con người), lấy log, sau đó áp dụng biến đổi Cosine rời rạc (DCT) để nén biểu diễn. Đặc trưng MFCC nắm bắt hiệu quả hình dạng phổ tổng quát của âm thanh trong khi loại bỏ thông tin pha và nhiễu cao tần. Tuy nhiên, vì MFCC được thiết kế cho giọng nói người (dải 100–8.000 Hz), hiệu quả của nó đối với tiếng tim (chủ yếu dưới 195 Hz) không được đảm bảo tuyệt đối và phụ thuộc nhiều vào cấu hình bộ lọc Mel.

Nogueira và cộng sự (2019) [3] kết hợp **MFCC, đặc trưng motif ảnh** (biểu diễn chuỗi tần số con như ảnh) và đặc trưng thời gian, phân loại tiếng tim từ PASCAL CHSC 2011 và CinC2016, cho thấy đặc trưng motif kết hợp MFCC có khả năng biểu diễn phong phú hơn MFCC đơn thuần. Chakir và cộng sự (2018) [4] xây dựng pipeline tiền xử lý tín hiệu PCG chuyên dụng cho PASCAL CHSC, kết hợp nhiều loại đặc trưng miền thời gian và tần số trước khi đưa vào bộ phân loại.

##### 2.3.2.4.3 Phương pháp Học sâu

Học sâu mang lại bước đột phá đáng kể trong phân tích tiếng tim, đặc biệt từ nửa sau thập kỷ 2010. Zhao và cộng sự (2024) [2] tổng kết rằng CNN là kiến trúc được sử dụng nhiều nhất — phần lớn các nghiên cứu đưa đặc trưng thời gian–tần số (spectrogram, Mel-spectrogram, MFCC map) vào CNN dưới dạng ảnh 2D và để mạng tự học các bộ lọc đặc trưng, thay vì thiết kế thủ công. Các kiến trúc như ResNet, DenseNet và MobileNet đã được áp dụng thành công thông qua học chuyển tiếp (*transfer learning*), tận dụng trọng số tiền huấn luyện trên ImageNet để bù đắp cho quy mô dữ liệu hạn chế.

Mạng LSTM và kiến trúc kết hợp CNN–RNN khai thác đặc tính chuỗi thời gian của tín hiệu PCG, cho phép mô hình học sự phụ thuộc dài hạn giữa các thành phần của chu kỳ tim — điều mà CNN thuần túy không làm được. Rath và cộng sự (2022) [5] nghiên cứu bài toán phát hiện bệnh tim từ dữ liệu mất cân bằng (bao gồm PASCAL CHSC), sử dụng kết hợp đặc trưng MFCC và tiền xử lý khử nhiễu, đề xuất chiến lược xử lý imbalanced learning phù hợp với bộ dữ liệu nhỏ không cân bằng. Almanifi và cộng sự (2022) [6] áp dụng học chuyển tiếp từ đặc trưng MFCC và BFCC sang mô hình CNN cho bài toán phát hiện tiếng thổi tim, đạt kết quả cạnh tranh trên tập dữ liệu gồm 489 bản ghi âm tim có và không có bệnh.

Hướng nghiên cứu gần đây của Latifi và cộng sự (2025) [7] đề xuất kiến trúc **Multi-Branch Deep Convolutional Network (MBDCN)** và mô hình kết hợp **LSTM-CNN (LSCN)**, sử dụng đầu vào là phổ năng lượng (*power spectrum*) kết hợp với nhiều kích thước bộ lọc song song để mô phỏng cơ chế xử lý thính giác của con người. Trên bộ dữ liệu PASCAL và các tập dữ liệu tiếng tim khác, mô hình LSCN đạt độ chính xác phân loại đa lớp **89,65%** và phân loại nhị phân (bình thường/bất thường) **93,93%**, vượt trội so với các phương pháp truyền thống MFCC và wavelet. Bahreini và cộng sự (2025) [8] kết hợp **đặc trưng MFCC với đặc trưng sâu CNN** theo kiến trúc lai, đạt trên 95% độ chính xác trên PASCAL CHSC — minh chứng cho hiệu quả của việc kết hợp tri thức lĩnh vực (*domain knowledge*) trong thiết kế đặc trưng với khả năng học tự động của CNN.

Lee và Kwak (2023) [9] đề xuất kết hợp **Wavelet Scattering Transform (WST)** và **Continuous Wavelet Transform (CWT)** cho hai nhánh CNN song song (1D-CNN xử lý đặc trưng WST và 2D-CNN xử lý ảnh CWT scalogram), sau đó kết hợp qua mô hình ensemble. WST là biến thể wavelet bất biến dịch vị (*translation-invariant*), phù hợp với tín hiệu PCG có cấu trúc lặp không đồng nhất. Trên tập PASCAL và CinC2016, phương pháp này vượt trội so với nhiều phương pháp trước đó. Về so sánh tổng quát, Kalimuthu và Hemanth (2025) [10] phân tích đối chiếu nhiều phương pháp học máy (SVM, kNN, Random Forest) và học sâu (CNN, ResNet, MobileNet) trên tập PCG kết hợp CinC2016 và CirCor 2022, cung cấp cơ sở tham chiếu hữu ích khi lựa chọn mô hình cho bài toán phân loại tiếng tim.

Cơ chế **chú ý** (*attention mechanism*) và kiến trúc Transformer cũng đang được thử nghiệm ngày càng nhiều trong phân tích tiếng tim. Cơ chế chú ý giúp mô hình tập trung vào các thời điểm và tần số quan trọng nhất trong chu kỳ tim, chẳng hạn như khoảng thời gian xung quanh S1/S2 và khu vực tần số chứa tiếng thổi. Xu hướng này phản ánh sự chuyển dịch từ thiết kế đặc trưng thủ công sang kiến trúc mô hình tự phát hiện cấu trúc quan trọng trong tín hiệu.

---

#### 2.3.2.5 Chỉ số Đánh giá

Bài toán phân loại tiếng tim có tính chất y tế (*medical diagnostic task*), vì vậy việc lựa chọn chỉ số đánh giá cần đặc biệt chú ý đến hệ quả lâm sàng của từng loại lỗi phân loại. Phân loại nhầm tiếng thổi tim thành bình thường (*false negative*) có hậu quả nghiêm trọng hơn phân loại nhầm bình thường thành bất thường (*false positive*).

**Bảng 2.37 — Chỉ số đánh giá cho bài toán phân loại tiếng tim**

| Chỉ số | Ký hiệu | Mô tả | Ghi chú |
|:-------|:-------:|:------|:--------|
| Độ chính xác tổng thể | *Accuracy* | Tỷ lệ bản ghi được phân loại đúng | Không tin cậy khi mất cân bằng lớp (Set B có 69% normal) |
| F1-score trung bình macro | *Macro-F1* | Trung bình F1 trên tất cả lớp, không tính trọng số tần suất | Chỉ số ưu tiên để đánh giá bình đẳng giữa các lớp thiểu số |
| Độ nhạy (*Recall*) theo lớp | *Sensitivity* | Tỷ lệ bản ghi bệnh được phát hiện đúng | Đặc biệt quan trọng cho `murmur` và `extrastole` |
| Độ đặc hiệu theo lớp | *Specificity* | Tỷ lệ bản ghi lành mạnh không bị phân loại sai thành bệnh | Quan trọng để kiểm soát tỷ lệ báo động nhầm |
| Diện tích dưới đường ROC | *AUC-ROC* | Đánh giá khả năng phân biệt tổng thể của mô hình | Phù hợp đặc biệt cho phân loại nhị phân (normal/abnormal) |
| Ma trận nhầm lẫn | *Confusion Matrix* | Bảng TP/FP/FN/TN từng cặp lớp | Xác định cặp lớp dễ nhầm: murmur↔normal, extrahls↔normal |

Trong khuôn khổ dự án CTH621, mô hình phân loại được đánh giá theo **Macro-F1** là chỉ số ưu tiên, kèm theo *Sensitivity* và *Specificity* theo từng lớp để đánh giá toàn diện hơn. Kiểm định sử dụng **stratified train/test split** (80/20), phân tầng theo cả `label` và `dataset` (A/B) để đảm bảo tỷ lệ đại diện đồng đều.

---

#### Tài liệu tham khảo

[1] P. Bentley, G. Nordehn, M. Coimbra, S. Mannor, và R. Getz, "The PASCAL Classifying Heart Sounds Challenge 2011 (CHSC2011) Results," 2011. URL: [http://www.peterjbentley.com/heartchallenge/](http://www.peterjbentley.com/heartchallenge/)

[2] Q. Zhao, S. Geng, B. Wang, Y. Sun, W. Nie, B. Bai, G. Tang, D. Zhao, J. Liu, C. Yang, F. Zhao, và S. Hong, "Deep Learning in Heart Sound Analysis: From Techniques to Clinical Applications," *Health Data Science*, tập 4, bài 0182, 2024. DOI: [10.34133/hds.0182](https://doi.org/10.34133/hds.0182)

[3] D. M. Nogueira, C. A. Ferreira, E. F. Gomes, và A. M. Jorge, "Classifying Heart Sounds Using Images of Motifs, MFCC and Temporal Features," *Journal of Medical Systems*, tập 43, số 6, bài 168, 2019. DOI: [10.1007/s10916-019-1286-5](https://doi.org/10.1007/s10916-019-1286-5)

[4] F. Chakir, A. Jilbab, C. Nacir, và A. Hammouch, "Phonocardiogram Signals Processing Approach for PASCAL Classifying Heart Sounds Challenge," *Signal, Image and Video Processing*, tập 12, số 6, tr. 1149–1155, 2018. DOI: [10.1007/s11760-018-1261-5](https://doi.org/10.1007/s11760-018-1261-5)

[5] A. Rath, D. Mishra, G. Panda, và M. Pal, "Development and Assessment of Machine Learning Based Heart Disease Detection Using Imbalanced Heart Sound Signal," *Biomedical Signal Processing and Control*, tập 76, bài 103730, 2022. DOI: [10.1016/j.bspc.2022.103730](https://doi.org/10.1016/j.bspc.2022.103730)

[6] O. R. A. Almanifi, A. F. Ab Nasir, M. A. M. Razman, M. R. Majeed, J. Musa, và J. H. Zulkifli, "Heartbeat Murmurs Detection in Phonocardiogram Recordings via Transfer Learning," *Alexandria Engineering Journal*, tập 61, số 12, tr. 10995–11002, 2022. DOI: [10.1016/j.aej.2022.04.031](https://doi.org/10.1016/j.aej.2022.04.031)

[7] S. A. Latifi, H. Ghassemian, và M. Imani, "Multi-Branch Convolutional Network and LSTM-CNN for Heart Sound Classification," *Physical and Engineering Sciences in Medicine*, 2025. DOI: [10.1007/s13246-025-01664-5](https://doi.org/10.1007/s13246-025-01664-5)

[8] M. Bahreini, R. Barati, và A. Kamali, "Cardiac Sound Classification Using a Hybrid Approach: MFCC-Based Feature Fusion and CNN Deep Features," *EURASIP Journal on Advances in Signal Processing*, tập 2025, bài 1203, 2025. DOI: [10.1186/s13634-025-01203-0](https://doi.org/10.1186/s13634-025-01203-0)

[9] J.-A. Lee và K.-C. Kwak, "Heart Sound Classification Using Wavelet Analysis Approaches and Ensemble of Deep Learning Models," *Applied Sciences*, tập 13, số 21, bài 11942, 2023. DOI: [10.3390/app132111942](https://doi.org/10.3390/app132111942)

[10] M. Kalimuthu và C. Hemanth, "A Comparative Analysis of Machine Learning and Deep Learning Approaches for Phonocardiogram Classification Using Dataset Integration," *IEEE Access*, 2025. DOI: [10.1109/ACCESS.2025.11174271](https://doi.org/10.1109/ACCESS.2025.11174271) *(Lưu ý: nghiên cứu dùng tập CinC2016 + CirCor 2022, không dùng PASCAL CHSC)*

---

#### Trích dẫn LaTeX (BibTeX)

```bibtex
% [1] Nguồn gốc thách thức PASCAL CHSC 2011 (trích dẫn chính thức cho dữ liệu)
@misc{bentley2011pascal,
  author       = {Bentley, Peter and Nordehn, Glenn and Coimbra, Miguel
                  and Mannor, Shie and Getz, Rita},
  title        = {The {PASCAL} {C}lassifying {H}eart {S}ounds {C}hallenge 2011
                  ({CHSC2011}) {R}esults},
  year         = {2011},
  howpublished = {\url{http://www.peterjbentley.com/heartchallenge/index.html}},
  note         = {Truy cập ngày 07/06/2026}
}

% [1b] Kaggle dataset (nếu cần trích dẫn phiên bản trên Kaggle)
@misc{king2014heartbeat,
  author       = {King, Ed},
  title        = {Heartbeat Sounds {[Dataset]}},
  year         = {2014},
  howpublished = {Kaggle},
  url          = {https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds},
  note         = {License: CC0 Public Domain. Truy cập ngày 07/06/2026}
}

% [2] Zhao et al. (2024) — Tổng quan học sâu trong phân tích tiếng tim
@article{zhao2024deeplearning,
  author  = {Zhao, Qinghao and Geng, Shijia and Wang, Boya and Sun, Yutong
             and Nie, Wei and Bai, Bo and Tang, Guanlan and Zhao, Dayun
             and Liu, Jianxin and Yang, Congbo and Zhao, Fangyan and Hong, Shenda},
  title   = {Deep Learning in Heart Sound Analysis: From Techniques to Clinical Applications},
  journal = {Health Data Science},
  volume  = {4},
  pages   = {Article 0182},
  year    = {2024},
  doi     = {10.34133/hds.0182},
  url     = {https://doi.org/10.34133/hds.0182}
}

% [3] Nogueira et al. (2019) — MFCC + motif features trên PASCAL
@article{nogueira2019classifying,
  author  = {Nogueira, David M. and Ferreira, Carlos A. and Gomes, Elsa F.
             and Jorge, Al\'ipio M.},
  title   = {Classifying Heart Sounds Using Images of Motifs, {MFCC} and Temporal Features},
  journal = {Journal of Medical Systems},
  volume  = {43},
  number  = {6},
  pages   = {168},
  year    = {2019},
  doi     = {10.1007/s10916-019-1286-5}
}

% [4] Chakir et al. (2018) — Phonocardiogram processing PASCAL challenge
@article{chakir2018phonocardiogram,
  author  = {Chakir, Fatima and Jilbab, Abdelhadi and Nacir, Chafik
             and Hammouch, Ahmed},
  title   = {Phonocardiogram Signals Processing Approach for {PASCAL}
             Classifying Heart Sounds Challenge},
  journal = {Signal, Image and Video Processing},
  volume  = {12},
  number  = {6},
  pages   = {1149--1155},
  year    = {2018},
  doi     = {10.1007/s11760-018-1261-5}
}

% [5] Rath et al. (2022) — Machine learning + imbalanced PASCAL data
@article{rath2022development,
  author  = {Rath, Adyasha and Mishra, Debahuti and Panda, Ganapati and Pal, Mahendra},
  title   = {Development and Assessment of Machine Learning Based Heart Disease
             Detection Using Imbalanced Heart Sound Signal},
  journal = {Biomedical Signal Processing and Control},
  volume  = {76},
  pages   = {103730},
  year    = {2022},
  doi     = {10.1016/j.bspc.2022.103730}
}

% [6] Almanifi et al. (2022) — Transfer learning MFCC/BFCC murmur detection
@article{almanifi2022heartbeat,
  author  = {Almanifi, Omar R. A. and Ab Nasir, Ahmad F. and Razman, Mohd A. M.
             and Majeed, Mahyudin R. and Musa, Rabiu and Zulkifli, Juliana H.},
  title   = {Heartbeat Murmurs Detection in Phonocardiogram Recordings via Transfer Learning},
  journal = {Alexandria Engineering Journal},
  volume  = {61},
  number  = {12},
  pages   = {10995--11002},
  year    = {2022},
  doi     = {10.1016/j.aej.2022.04.031}
}

% [9] Lee & Kwak (2023) — WST + CWT ensemble CNN trên PASCAL + CinC2016
@article{lee2023heartsound,
  author  = {Lee, Jin-A and Kwak, Keun-Chang},
  title   = {Heart Sound Classification Using Wavelet Analysis Approaches
             and Ensemble of Deep Learning Models},
  journal = {Applied Sciences},
  volume  = {13},
  number  = {21},
  pages   = {11942},
  year    = {2023},
  doi     = {10.3390/app132111942}
}

% [10] Kalimuthu & Hemanth (2025) — So sánh ML/DL trên CinC2016 + CirCor2022
@article{kalimuthu2025comparative,
  author  = {Kalimuthu, Manikandan and Hemanth, Chinnasamy},
  title   = {A Comparative Analysis of Machine Learning and Deep Learning
             Approaches for Phonocardiogram Classification Using Dataset Integration},
  journal = {IEEE Access},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.11174271},
  note    = {Sử dụng tập dữ liệu CinC2016 + CirCor 2022 (không dùng PASCAL CHSC)}
}

% [7] Latifi et al. (2025) — MBDCN + LSTM-CNN, accuracy 89.65%
@article{latifi2025multibranch,
  author  = {Latifi, Seyed Amir and Ghassemian, Hassan and Imani, Maryam},
  title   = {Multi-Branch Convolutional Network and {LSTM-CNN} for Heart Sound Classification},
  journal = {Physical and Engineering Sciences in Medicine},
  year    = {2025},
  doi     = {10.1007/s13246-025-01664-5},
  note    = {Preprint: arXiv:2407.10689}
}

% [8] Bahreini et al. (2025) — MFCC hybrid CNN, >95% accuracy on PASCAL
@article{bahreini2025cardiac,
  author  = {Bahreini, M. and Barati, R. and Kamali, A.},
  title   = {Cardiac Sound Classification Using a Hybrid Approach:
             {MFCC}-Based Feature Fusion and {CNN} Deep Features},
  journal = {EURASIP Journal on Advances in Signal Processing},
  volume  = {2025},
  pages   = {1203},
  year    = {2025},
  doi     = {10.1186/s13634-025-01203-0}
}
```
