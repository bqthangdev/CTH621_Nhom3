### 2.3.1 RealWaste Dataset

---

#### 2.3.1.1 Giới thiệu Tập dữ liệu

**Tên tập dữ liệu:** RealWaste  
**Nguồn:** UCI Machine Learning Repository — [https://archive.ics.uci.edu/dataset/908/realwaste](https://archive.ics.uci.edu/dataset/908/realwaste)  
**Tác giả tạo lập:** Sam Single, Saeid Iranmanesh, Raad Raad  
**Năm đóng góp lên UCI:** 2023  
**Giấy phép:** CC BY-NC-SA 4.0 *(Creative Commons Attribution-NonCommercial-ShareAlike 4.0 — theo README gốc trên GitHub; xem lưu ý bên dưới)*  
**Bản chất dữ liệu:** Dữ liệu thực (*real-world*) — ảnh chụp chất thải trong điều kiện bãi tiếp nhận thực tế, không qua dàn dựng  
**Nhiệm vụ học máy gốc:** Phân loại ảnh đa lớp (*multi-class image classification*)  
**Biến mục tiêu:** Nhãn loại vật liệu chất thải (*material type label*) — 9 lớp vật liệu

Phân loại chất thải tự động là một trong những bài toán ứng dụng học máy ngày càng được quan tâm trong bối cảnh khủng hoảng rác thải đô thị toàn cầu ngày càng trầm trọng. Theo ước tính của Ngân hàng Thế giới, chi phí xử lý chất thải không đúng cách trên phạm vi toàn cầu dao động từ 222 đến 370 tỷ đô la Mỹ mỗi năm, và các dự báo chỉ ra con số này có thể vượt 400 tỷ đô la vào năm 2025. Trong bối cảnh đó, khả năng phân loại chất thải tự động và chính xác đóng vai trò then chốt trong việc cải thiện hiệu quả tái chế và giảm thiểu lượng rác chôn lấp.

Phần lớn các tập dữ liệu ảnh phân loại chất thải trước đây được xây dựng trong điều kiện kiểm soát cao: vật thể đặt trên nền đồng nhất, ánh sáng đồng đều và không có tạp chất ngoại cảnh. Điển hình nhất là **TrashNet** (Yang & Thung, 2016) với 2.527 ảnh thuộc 6 danh mục — bộ dữ liệu được sử dụng rộng rãi nhất trong lĩnh vực nhưng xa rời thực tế triển khai. Khoảng cách giữa điều kiện lý tưởng đó và thực tế tại các bãi chôn lấp — nơi chất thải xuất hiện ở trạng thái lẫn lộn, bẩn, biến dạng, dưới ánh sáng không đồng đều — là nguyên nhân chính dẫn đến sự sụt giảm hiệu suất đáng kể của mô hình khi triển khai thực tế.

Tập dữ liệu **RealWaste** được phát triển bởi Single và cộng sự (2023) [1] với mục đích lấp đầy khoảng trống đó: toàn bộ 4.752 hình ảnh được chụp tại điểm tiếp nhận của cơ sở **Whyte's Gully Waste and Resource Recovery** (Wollongong, New South Wales, Úc) — tức là ảnh chụp ngay khi phương tiện vận chuyển đổ rác, trong điều kiện thực địa chưa qua tiền xử lý hay phân loại thủ công trước đó. Bộ dữ liệu bao gồm 9 danh mục vật liệu phù hợp với chuẩn quản lý chất thải đô thị, tương đồng với các thành phần chính được Ngân hàng Thế giới xác định trong báo cáo "What a Waste 2.0" (2018).

**Phạm vi ứng dụng trong dự án:** Trong khuôn khổ dự án CTH621, tập dữ liệu RealWaste được sử dụng cho bài toán **phân loại ảnh** (Classification — Nhóm C), với mục tiêu phân loại ảnh chất thải vào một trong 9 danh mục vật liệu. Tập dữ liệu cũng được sử dụng cho bài toán **phân cụm** (Clustering), nhằm xác định nhóm hình ảnh có đặc trưng thị giác tương đồng.

**Bảng 2.32 — Thông tin tổng quan tập dữ liệu RealWaste**

| Thuộc tính | Thông tin |
|:-----------|:----------|
| **Tên dataset** | RealWaste |
| **Nguồn (UCI)** | [https://archive.ics.uci.edu/dataset/908/realwaste](https://archive.ics.uci.edu/dataset/908/realwaste) |
| **DOI dataset (UCI)** | [10.24432/C5SS4G](https://doi.org/10.24432/C5SS4G) |
| **Bài báo giới thiệu** | Single et al. (2023), *Information* 14(12), 633 |
| **DOI bài báo** | [10.3390/info14120633](https://doi.org/10.3390/info14120633) |
| **Năm đóng góp dữ liệu** | 2023 |
| **Giấy phép** | CC BY-NC-SA 4.0 *(xem lưu ý)* |
| **Loại dữ liệu** | Ảnh màu RGB (*color images*) — định dạng JPEG, 524×524 px |
| **Địa điểm thu thập** | Whyte's Gully Waste and Resource Recovery, Wollongong, NSW, Úc |
| **Số ảnh (tổng)** | 4.752 |
| **Số lớp phân loại** | 9 |
| **Giá trị thiếu** | Không có |

> **Lưu ý về giấy phép:** README trên kho mã nguồn GitHub chính thức của tác giả ([github.com/sam-single/realwaste](https://github.com/sam-single/realwaste)) ghi rõ giấy phép **CC BY-NC-SA 4.0** — cấm sử dụng cho mục đích thương mại và bắt buộc chia sẻ tương tự khi phân phối lại. Trong khi đó, trang UCI Machine Learning Repository ghi **CC BY 4.0** (không có ràng buộc phi thương mại). Trường hợp có mâu thuẫn giữa hai nguồn, tuyên bố của tác giả gốc (CC BY-NC-SA 4.0) cần được ưu tiên áp dụng. Người dùng nên kiểm tra điều khoản trực tiếp từ tác giả khi có nhu cầu sử dụng ngoài phạm vi học thuật.

---

#### 2.3.1.2 Thống kê Số lượng Mẫu và Phân phối Lớp

Tập dữ liệu được tổ chức theo cấu trúc thư mục phẳng (*flat folder structure*): mỗi danh mục vật liệu tương ứng với một thư mục con, tất cả ảnh đặt trực tiếp trong thư mục đó. Quy ước đặt tên tệp theo dạng `<CategoryName>_<id>.jpg`. Không có tệp metadata CSV kèm theo — nhãn lớp được suy ra trực tiếp từ tên thư mục chứa ảnh (*folder-based labeling*). Đây là quy ước phổ biến trong các tập dữ liệu ảnh phân loại, tương thích với các thư viện huấn luyện mô hình học sâu phổ biến như `ImageDataGenerator` (Keras/TensorFlow) hoặc `ImageFolder` (PyTorch).

Tổng cộng có **4.752 ảnh** phân bố trên **9 danh mục**. Bảng 2.33 trình bày phân phối chi tiết theo từng lớp vật liệu.

**Bảng 2.33 — Phân phối số lượng ảnh theo danh mục vật liệu**

| STT | Danh mục | Tên tiếng Việt | Số ảnh | Tỷ lệ (%) |
|:---:|:---------|:---------------|-------:|----------:|
| 1 | Plastic | Nhựa | 921 | 19,38 |
| 2 | Metal | Kim loại | 790 | 16,62 |
| 3 | Miscellaneous Trash | Rác hỗn hợp | 495 | 10,42 |
| 4 | Paper | Giấy | 500 | 10,52 |
| 5 | Cardboard | Bìa cứng / Carton | 461 | 9,70 |
| 6 | Vegetation | Thực vật / Rác vườn | 436 | 9,17 |
| 7 | Glass | Thủy tinh | 420 | 8,84 |
| 8 | Food Organics | Thực phẩm hữu cơ | 411 | 8,65 |
| 9 | Textile Trash | Rác dệt may | 318 | 6,69 |
| — | **Tổng** | — | **4.752** | **100,00** |

> **Nhận xét phân phối lớp:** Tập dữ liệu có mức độ mất cân bằng lớp (*class imbalance*) ở mức vừa phải. Lớp đa số là *Plastic* (921 ảnh, 19,38%) trong khi lớp thiểu số là *Textile Trash* (318 ảnh, 6,69%), tạo ra tỷ lệ chênh lệch khoảng 2,9:1 — thấp hơn đáng kể so với các tập dữ liệu chuỗi thời gian như HAR70+ (tỷ lệ 237:1). Ở mức này, mất cân bằng lớp không đòi hỏi các kỹ thuật xử lý đặc biệt phức tạp, nhưng việc sử dụng *weighted cross-entropy loss* hoặc đánh giá theo Macro-F1 thay vì accuracy thuần túy vẫn được khuyến nghị để đảm bảo hiệu suất nhất quán trên các lớp thiểu số.

---

#### 2.3.1.3 Cấu trúc Tệp và Định dạng Ảnh

Tất cả ảnh trong tập dữ liệu RealWaste đều được lưu dưới **định dạng JPEG** (`.jpg`) với độ phân giải chuẩn **524×524 pixel** — đây là độ phân giải được sử dụng trong bài báo nghiên cứu gốc của Single và cộng sự (2023) [1]. Ảnh có độ phân giải đầy đủ (*full-size resolution*) có thể được yêu cầu riêng từ tác giả tương ứng. Ảnh là ảnh màu 3 kênh RGB, không có kênh alpha, và kích thước tệp dao động từ khoảng 75 KB đến hơn 200 KB tùy theo nội dung ảnh.

Về đặc điểm thị giác của dữ liệu, do được thu thập trong môi trường thực địa, tập dữ liệu mang những đặc trưng riêng biệt so với các bộ dữ liệu ảnh chất thải truyền thống. Nhiều ảnh chứa đồng thời nhiều loại vật liệu khác nhau, trong đó chỉ có một loại được gán nhãn chính theo vật liệu chiếm ưu thế. Độ sáng và màu sắc biến thiên theo điều kiện thời tiết và thời điểm trong ngày tại điểm chụp. Đặc biệt, các vật thể thường ở trạng thái hỏng, bị bẩn, nén vỡ, hoặc lẫn lộn với các loại rác khác — khác biệt căn bản so với ảnh studio trong TrashNet hay các bộ dữ liệu tổng hợp khác. Đây chính là đặc điểm làm cho RealWaste có giá trị thực tiễn cao hơn cho các ứng dụng phân loại rác thải tự động trong môi trường triển khai thực tế.

---

#### 2.3.1.3.1 Vấn đề Chất lượng Dữ liệu

Tập dữ liệu RealWaste không có giá trị thiếu — nhãn lớp đầy đủ cho toàn bộ 4.752 ảnh được xác định thông qua cấu trúc thư mục. Tuy nhiên, có một số đặc điểm chất lượng cần lưu ý trong quá trình xử lý và huấn luyện mô hình:

**Mất cân bằng lớp ở mức độ vừa phải.** Như đã trình bày ở Bảng 2.33, tỷ lệ chênh lệch giữa lớp *Plastic* (nhiều nhất, 921 ảnh) và *Textile Trash* (ít nhất, 318 ảnh) xấp xỉ 2,9:1. Mức độ này chưa đủ nghiêm trọng để đòi hỏi oversampling phức tạp, nhưng cần được giải quyết bằng *class_weight='balanced'* hoặc đánh giá theo Macro-F1 để tránh thiên lệch về các lớp có tần suất cao.

**Nhập nhằng giữa các lớp ranh giới.** Do đặc thù của môi trường thực địa, ranh giới giữa một số cặp lớp không rõ ràng về mặt thị giác. *Cardboard* ướt hoặc dẹt dễ nhầm lẫn với *Paper* và *Miscellaneous Trash*; *Vegetation* có ngoại quan tương đồng với *Food Organics* trong nhiều trường hợp; *Glass* vỡ và *Plastic* trong suốt cũng có thể bị nhầm lẫn. Điều này đặt ra thách thức phân loại ngay cả đối với các mô hình học sâu hiện đại.

**Đa đối tượng trong một ảnh (*multi-instance contamination*).** Vì ảnh được chụp tại điểm đổ rác thực tế, nhiều ảnh chứa đồng thời nhiều loại vật liệu khác nhau, trong khi nhãn chỉ phản ánh loại vật liệu được xem là chính. Đây là một nguồn nhiễu nhãn (*label noise*) vốn có của tập dữ liệu không thể loại bỏ hoàn toàn mà không có quá trình chú thích lại từ đầu.

---

#### 2.3.1.4 Tổng quan Nghiên cứu Liên quan

##### 2.3.1.4.1 Bài báo giới thiệu tập dữ liệu

Single và cộng sự (2023) [1] trình bày quá trình xây dựng và kiểm định bộ dữ liệu RealWaste trong bài báo *"RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning"*, đăng trên tạp chí **Information** (MDPI), tập 14, số 12, bài 633. Đây là bài báo mở (*open access*), xuất phát từ một khóa luận danh dự (*honors thesis*) nghiên cứu khả năng áp dụng mạng nơ-ron tích chập trong điều kiện chất thải thực địa.

**Mục tiêu và bối cảnh:** Mục tiêu cốt lõi của nghiên cứu là đánh giá liệu các mô hình CNN được huấn luyện trên ảnh chất thải ở dạng thuần túy, chưa qua sử dụng (*pure and unadulterated forms*) có duy trì hiệu suất tốt khi được đưa vào môi trường chất thải thực địa (*real waste items*) hay không. Câu hỏi nghiên cứu này phản ánh khoảng cách cốt lõi trong lĩnh vực — khoảng cách giữa dữ liệu huấn luyện lý tưởng và dữ liệu triển khai thực tế.

**Phương pháp và kết quả:** Bài báo áp dụng **năm mô hình học sâu** (*five deep learning models*) trên tập dữ liệu RealWaste, so sánh hiệu suất phân loại giữa các kiến trúc để xác định mô hình phù hợp nhất cho bài toán phân loại chất thải thực địa. *(Lưu ý: Tên cụ thể của các mô hình và kết quả số liệu chi tiết chưa được xác minh từ bài báo gốc tại thời điểm soạn thảo — xem mục 2.3.1.6.)*

**Ý nghĩa:** RealWaste là một trong số ít bộ dữ liệu ảnh phân loại chất thải được xây dựng trong điều kiện thực địa xác thực, góp phần thu hẹp khoảng cách giữa nghiên cứu học thuật và triển khai thực tế trong quản lý chất thải đô thị. Từ khi công bố vào cuối năm 2023, bộ dữ liệu đã được sử dụng trong nhiều nghiên cứu tiếp theo như Islam và cộng sự (2025) [2], GlobalWasteData (Ijaz et al., 2026) [3], và các nghiên cứu so sánh mô hình phân loại chất thải khác.

---

##### 2.3.1.4.2 Các Phương pháp Trích xuất Đặc trưng trong Nghiên cứu Liên quan

Bài toán phân loại chất thải từ ảnh đã trải qua một quá trình phát triển rõ ràng — từ các phương pháp trích xuất đặc trưng thủ công truyền thống sang các kiến trúc học sâu hiện đại — phản ánh xu hướng chung của lĩnh vực thị giác máy tính (*computer vision*).

**Phương pháp truyền thống — trích xuất đặc trưng thủ công.** Trước khi học sâu trở nên phổ biến, các hệ thống phân loại chất thải dựa trên các đặc trưng thị giác được thiết kế thủ công (*handcrafted features*), sau đó đưa vào bộ phân loại học máy cổ điển. Các đặc trưng phổ biến bao gồm biểu đồ màu sắc (*color histogram*) trong không gian màu RGB hoặc HSV — phản ánh thành phần màu của vật liệu; đặc trưng kết cấu bề mặt qua *Local Binary Pattern* (LBP) và *Gray-Level Co-occurrence Matrix* (GLCM); và đặc trưng hình dạng cạnh *Histogram of Oriented Gradients* (HOG). Các bộ phân loại đi kèm thường là *Support Vector Machine* (SVM) hoặc *k-Nearest Neighbors* (kNN). Nhược điểm căn bản của hướng tiếp cận này là đặc trưng thủ công phụ thuộc mạnh vào điều kiện chụp ảnh — ánh sáng, góc độ, tình trạng bề mặt — và không khái quát hóa tốt khi dữ liệu có biến động lớn, như trường hợp ảnh thực địa tại bãi thải.

**CNN căn bản và học chuyển tiếp (*Transfer Learning*).** Sự ra đời của tập dữ liệu TrashNet (Yang & Thung, 2016) đã thúc đẩy ứng dụng mạng nơ-ron tích chập vào bài toán phân loại chất thải. Tuy nhiên, do quy mô dữ liệu thường chỉ ở mức vài nghìn ảnh, các mô hình CNN huấn luyện từ đầu (*from scratch*) nhanh chóng được thay thế bằng chiến lược **học chuyển tiếp** — tận dụng các mô hình tiền huấn luyện (*pre-trained*) trên ImageNet và tinh chỉnh (*fine-tune*) cho bài toán phân loại chất thải. Wang (2020) ứng dụng VGG16 cho phân loại chất thải gia đình và đạt độ chính xác 75,6%, cho thấy kiến trúc VGG16, dù mạnh trong nhận dạng ảnh tổng quát, gặp khó khăn khi đặc trưng thị giác giữa các lớp chất thải có độ tương đồng cao. Rabano và cộng sự (2018) áp dụng MobileNet và đạt 87,2%, khai thác ưu điểm của kiến trúc nhẹ phù hợp cho thiết bị biên. Cao và Xiang (2020) sử dụng InceptionV3 với fine-tuning, đạt 93,2% test accuracy (99,3% train accuracy), tuy nhiên khoảng cách lớn giữa tập huấn luyện và kiểm tra phản ánh hiện tượng quá khớp (*overfitting*) do dữ liệu huấn luyện nhỏ. Meng và cộng sự (2020) so sánh SVM, CNN đơn giản, ResNet50 và mô hình kết hợp HOG+CNN trên 2.527 ảnh 6 danh mục, với ResNet50 đạt độ chính xác tốt nhất 95,35% — khẳng định ưu thế của kiến trúc residual trong trích xuất đặc trưng thứ bậc sâu từ ảnh chất thải [2].

**Cơ chế chú ý và kiến trúc song song.** Hướng nghiên cứu gần đây tập trung vào việc nâng cao khả năng chọn lọc đặc trưng thông qua các cơ chế chú ý (*attention mechanism*). Guo và cộng sự (2021) tích hợp cơ chế chú ý vào EfficientNet và đạt độ chính xác trung bình 93,47% (cao nhất 98,3% trên bộ dữ liệu cụ thể), cho thấy cơ chế chú ý giúp mô hình tập trung vào vùng đặc trưng phân biệt của chất thải trong ảnh, đặc biệt hiệu quả khi nền ảnh phức tạp và có nhiều đối tượng. Endah và Shiddiq (2020) so sánh VGG16, ResNet-50 và Xception trên TrashNet, với Xception đạt kết quả tốt nhất 88% nhờ cơ chế tích chập tách biệt theo chiều sâu (*depthwise separable convolution*) giúp trích xuất đặc trưng hiệu quả hơn trên ảnh với nền đơn giản [2].

Một bước tiến đáng chú ý đến từ Islam và cộng sự (2025) [2], khi đề xuất kiến trúc kết hợp **DenseNet201 với cơ chế *Squeeze-and-Excitation* (SE)** và **hai nhánh CNN song song** — một nhánh sử dụng max-pooling và nhánh kia sử dụng average-pooling. Cơ chế SE cho phép mạng học cách tái điều chỉnh trọng số theo kênh đặc trưng (*channel-wise feature recalibration*), nhấn mạnh các đặc trưng quan trọng và ức chế nhiễu. Kiến trúc song song đồng thời trích xuất đặc trưng từ hai góc độ bổ sung nhau, nâng cao khả năng biểu diễn đặc trưng đa tỉ lệ. Trên bộ dữ liệu RealWaste, kiến trúc này đạt độ chính xác **93,17%** (precision 93,26%, recall 93,26%, F1-score 93,25%).

Nhìn chung, xu hướng nghiên cứu trong lĩnh vực này cho thấy bốn bước chuyển biến rõ ràng: đặc trưng thủ công đã bị học sâu vượt qua về hiệu suất; học chuyển tiếp là chiến lược hiệu quả khi dữ liệu ở quy mô vừa phải; cơ chế chú ý và kiến trúc song song tiếp tục cải thiện khả năng phân biệt giữa các lớp có ngoại quan tương đồng; và đánh giá trên dữ liệu thực địa (như RealWaste) thường thấp hơn 5–10% so với dữ liệu studio, nhấn mạnh tầm quan trọng của tập dữ liệu có tính đại diện thực tế cao trong nghiên cứu phân loại chất thải.

---

#### 2.3.1.5 Chỉ số Đánh giá

Do bài toán gốc là **phân loại ảnh đa lớp** (*multi-class image classification*) với 9 danh mục và mức mất cân bằng lớp vừa phải, các chỉ số đánh giá phù hợp bao gồm:

**Bảng 2.34 — Chỉ số đánh giá cho bài toán phân loại ảnh chất thải**

| Chỉ số | Ký hiệu | Mô tả | Ghi chú |
|:-------|:-------:|:------|:--------|
| Độ chính xác tổng thể | *Accuracy* | Tỷ lệ ảnh được phân loại đúng trên toàn bộ tập kiểm tra | Cần đánh giá kèm với Macro-F1 do mất cân bằng lớp |
| F1-score trung bình macro | *Macro-F1* | Trung bình cộng F1 của từng lớp, không tính tỷ trọng tần suất | Chỉ số ưu tiên khi cần đánh giá bình đẳng mọi lớp vật liệu |
| F1-score trung bình có trọng số | *Weighted-F1* | Trung bình F1 có trọng số theo tần suất lớp | Phản ánh hiệu suất tổng thể theo phân phối thực tế |
| Độ chính xác dự đoán dương | *Precision* | Tỷ lệ dự đoán tích cực đúng trên tổng dự đoán tích cực | Quan trọng khi chi phí phân loại sai cao giữa các cặp vật liệu |
| Độ nhạy theo lớp | *Per-class Recall* | Tỷ lệ ảnh của lớp *k* được phân loại đúng | Đặc biệt quan trọng cho *Textile Trash* — lớp có ít ảnh nhất |
| Ma trận nhầm lẫn | *Confusion Matrix* | Bảng TP/FP/FN/TN cho từng cặp lớp | Xác định cặp lớp dễ nhầm: Cardboard↔Paper, Glass↔Plastic |

Trong khuôn khổ dự án CTH621, mô hình phân loại được đánh giá theo **Macro-F1** là chỉ số ưu tiên, nhằm đảm bảo hiệu suất nhất quán trên toàn bộ các lớp bao gồm cả những lớp ít mẫu nhất. *Accuracy* được báo cáo bổ sung. Kiểm định sử dụng **stratified train/test split** (80/20) để đảm bảo tỷ lệ lớp được duy trì trong cả tập huấn luyện lẫn tập kiểm tra.

---

#### 2.3.1.6 Ghi chú

> **[!] Thông tin chờ bổ sung từ bài báo gốc:** Bài báo Single et al. (2023) [DOI: 10.3390/info14120633] mô tả việc áp dụng **năm mô hình học sâu** trên RealWaste, tuy nhiên trang MDPI không thể truy cập trực tiếp tại thời điểm soạn thảo báo cáo này. Các thông tin sau cần được bổ sung sau khi có điều kiện truy cập bài báo: (1) tên cụ thể của 5 mô hình học sâu được so sánh; (2) kết quả accuracy / F1-score của từng mô hình; (3) phương pháp đánh giá (train/test split hay cross-validation); (4) tên trường đại học / đơn vị nghiên cứu của các tác giả. Người đọc có thể tham khảo trực tiếp bài báo tại [https://www.mdpi.com/2078-2489/14/12/633](https://www.mdpi.com/2078-2489/14/12/633).

---

#### Tài liệu tham khảo

[1] S. Single, S. Iranmanesh, và R. Raad, "RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning," *Information*, tập 14, số 12, tr. 633, 2023. DOI: [10.3390/info14120633](https://doi.org/10.3390/info14120633)

[2] M. M. Islam, S. M. M. Hasan, M. R. Hossain, M. P. Uddin, và M. A. Mamun, "Towards sustainable solutions: Effective waste classification framework via enhanced deep convolutional neural networks," *PLOS ONE*, tập 20, số 6, tr. e0324294, 2025. DOI: [10.1371/journal.pone.0324294](https://doi.org/10.1371/journal.pone.0324294)

[3] M. Ijaz, S. U. R. Khan, A. U. Rehman, T. Asif, S. Vollmer, A. Dengel, và M. N. Asim, "GlobalWasteData: A Large-Scale, Integrated Dataset for Robust Waste Classification and Environmental Monitoring," *arXiv preprint*, arXiv:2602.07463, 2026. DOI: [10.48550/arXiv.2602.07463](https://doi.org/10.48550/arXiv.2602.07463)

---

#### Trích dẫn LaTeX (BibTeX)

```bibtex
% [1] Bài báo giới thiệu tập dữ liệu RealWaste
@article{single2023realwaste,
  author    = {Single, Sam and Iranmanesh, Saeid and Raad, Raad},
  title     = {{RealWaste}: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning},
  journal   = {Information},
  volume    = {14},
  number    = {12},
  pages     = {633},
  year      = {2023},
  publisher = {MDPI},
  doi       = {10.3390/info14120633},
  url       = {https://www.mdpi.com/2078-2489/14/12/633}
}

% [1b] Trích dẫn dataset trên UCI (tuỳ định dạng yêu cầu)
@misc{single2023realwaste_uci,
  author       = {Single, Sam and Iranmanesh, Saeid and Raad, Raad},
  title        = {{RealWaste} {[Dataset]}},
  year         = {2023},
  howpublished = {UCI Machine Learning Repository},
  doi          = {10.24432/C5SS4G},
  url          = {https://archive.ics.uci.edu/dataset/908/realwaste},
  note         = {Truy cập ngày 07/06/2026}
}

% [2] Islam et al. (2025) — kiến trúc DenseNet201+SE+Parallel CNN, kết quả trên RealWaste
@article{islam2025sustainable,
  author  = {Islam, Md. Minhazul and Hasan, S. M. Mahedy and Hossain, Md. Rakib
             and Uddin, Md. Palash and Mamun, Md. Al},
  title   = {Towards sustainable solutions: Effective waste classification framework
             via enhanced deep convolutional neural networks},
  journal = {PLOS ONE},
  volume  = {20},
  number  = {6},
  pages   = {e0324294},
  year    = {2025},
  doi     = {10.1371/journal.pone.0324294},
  url     = {https://doi.org/10.1371/journal.pone.0324294}
}

% [3] GlobalWasteData (2026) — tích hợp RealWaste vào bộ dữ liệu lớn
@misc{ijaz2026globalwastedata,
  author        = {Ijaz, Misbah and Khan, Saif Ur Rehman and Rehman, Abd Ur
                   and Asif, Tayyaba and Vollmer, Sebastian and Dengel, Andreas
                   and Asim, Muhammad Nabeel},
  title         = {{GlobalWasteData}: A Large-Scale, Integrated Dataset for Robust
                   Waste Classification and Environmental Monitoring},
  year          = {2026},
  eprint        = {2602.07463},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2602.07463}
}
```
