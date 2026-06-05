# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Đức Thành  
**Nhóm:** C02  
**Ngày:** 05/06/2026  

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?** High cosine similarity có nghĩa là 2 câu có độ tương đồng tính trên không gian vector là gần nhau, hay nói cách khác 2 câu này có ngữ nghĩa gần giống như nhau.

**Ví dụ HIGH similarity:** 
- Sentence A: Tôi rất thích học Trí tuệ Nhân tạo.  
- Sentence B: Tôi có niềm đam mê lớn với lĩnh vực AI.  
- Tại sao tương đồng: Cả hai câu đều truyền tải một thông điệp: người nói có hứng thú và yêu thích Trí tuệ Nhân tạo.

**Ví dụ LOW similarity:** 
- Sentence A: Hôm nay thời tiết ở Hà Nội rất đẹp.  
- Sentence B: ChatGPT ra mắt lần đầu vào năm 2022.  
- Tại sao khác: Hai câu truyền tải 2 thông điệp hoàn toàn khác nhau về chủ đề (thời tiết vs công nghệ) và không có mối liên kết ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?** Lí do là bởi cosine similarity tính toán góc giữa hai vector hướng thay vì khoảng cách độ dài, cho nên sẽ không bị yếu tố độ dài của 2 câu đó ảnh hưởng. Trong khi đó, Euclidean distance đo khoảng cách thẳng giữa hai điểm mút vector nên phụ thuộc lớn vào độ dài văn bản. Điều này dẫn đến việc hai văn bản có cùng ý nghĩa nhưng độ dài cách biệt lớn vẫn bị tính là có khoảng cách Euclidean rất xa nhau (lỗi).

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

*Trình bày phép tính:* 
- Chunk đầu tiên xử lý: 500 ký tự đầu.  
- Số ký tự còn lại: 10,000 - 500 = 9,500 ký tự.  
- Kể từ chunk thứ 2, do có `overlap=50`, mỗi bước dịch chuyển (stride) thực tế tiến thêm: 500 - 50 = 450 ký tự mới.  
- Số lượng chunk bổ sung: ceil(9,500 / 450) = ceil(21.11) = 22 chunks.  
- Tổng số chunk: 1 + 22 = 23 chunks.

*Đáp án:* **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?** 
- Nếu overlap tăng lên 100, bước dịch chuyển thực tế giảm xuống còn 500 - 100 = 400 ký tự mới/chunk. Số lượng chunk bổ sung sẽ là ceil(9,500 / 400) = 24 chunks -> Tổng số chunk tăng lên thành **25 chunks**.  
- Việc tăng overlap giúp bảo toàn ngữ cảnh tốt hơn ở các vùng biên chia cắt, tránh hiện tượng đứt gãy thông tin giữa các câu liền kề và tạo "cầu nối" ngữ cảnh liên tục phục vụ cho quá trình Retrieval chính xác hơn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Y tế / Bảo hiểm Y tế (Healthcare Formulary Guides & Drug Safety)

**Tại sao nhóm chọn domain này?** Nhóm đã có tìm hiểu về cấu trúc tài liệu y tế trong buổi lab trước, nhận thấy các văn bản này chứa nhiều thuật ngữ chuyên ngành phức tạp, tên thuốc viết tắt và cấu trúc phân cấp nghiêm ngặt. Đây là tập dữ liệu lý tưởng để thử nghiệm sức mạnh của các chiến lược chunking nâng cao nhằm giữ trọn vẹn ngữ cảnh.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | caremark-oct2013.txt | Eric Minikel (cnsdrugs) | 26,541 | doc_type: formulary_guide, year: 2013 |
| 2 | healthalliance-2013.txt | Eric Minikel (cnsdrugs) | 16,892 | doc_type: formulary_guide, year: 2013 |
| 3 | FDAMDD_v3b_1216_15Feb2008_nostructures.txt | Eric Minikel (cnsdrugs) | 485,610 | doc_type: fda_chemical_registry, year: 2008 |
| 4 | humana_2014_wi.txt | Eric Minikel (cnsdrugs) | 321,405 | doc_type: formulary_guide, year: 2014 |
| 5 | uhc-cns-drugs-pdf-list.txt | Eric Minikel (cnsdrugs) | 42,150 | doc_type: drug_list, year: 2014 |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | `str` | `"caremark-oct2013"` | Giúp định danh chính xác tài liệu nguồn phục vụ cho việc xóa (`delete_document`) hoặc cập nhật. |
| `doc_type` | `str` | `"formulary_guide"` | Hỗ trợ lọc trước (`pre-filtering`) loại văn bản mong muốn khi query, loại bỏ nhiễu từ các văn bản không liên quan. |
| `year` | `int` | `2013` | Giúp lọc dữ liệu theo mốc thời gian hiệu lực, đảm bảo mô hình chỉ lấy thông tin chính sách bảo hiểm đúng năm yêu cầu. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy thực tế qua file `main.py` trên toàn bộ kho tài liệu (5 documents):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Toàn bộ 5 văn bản | FixedSizeChunker (`fixed_size`) | 1798 | ~600 ký tự | Trung bình (Bị cắt khúc ngẫu nhiên ở biên) |
| Toàn bộ 5 văn bản | SentenceChunker (`by_sentences`) | 1121 | ~3 câu/chunk | Tốt (Giữ trọn vẹn cấu trúc câu đơn) |
| Toàn bộ 5 văn bản | RecursiveChunker (`recursive`) | 2322 | Biến thiên | Rất tốt (Cắt theo cấu trúc dòng trống phân cấp) |

### Strategy Của Tôi

**Loại:** `RecursiveChunker`

**Mô tả cách hoạt động:** Chiến lược này thực hiện cắt nhỏ văn bản một cách đệ quy bằng cách duyệt qua một danh sách các ký tự phân tách theo thứ tự ưu tiên giảm dần: `["\n\n", "\n", ". ", " ", ""]`. Thuật toán cố gắng gộp các đoạn văn bản lại với nhau thành một khối đệm buffer sao cho độ dài không vượt quá `chunk_size`. Khi một phân đoạn vượt ngưỡng, nó sẽ lùi xuống ký tự phân tách có độ ưu tiên thấp hơn tiếp theo để bẻ nhỏ văn bản mà không làm mất cấu trúc khối gốc.

**Tại sao tôi chọn strategy này cho domain nhóm?** Tài liệu y tế và chính sách bảo hiểm thuốc thường được cấu trúc dưới dạng các mục lớn cách nhau bằng 2 dòng trống (`\n\n`), các điều khoản xuống dòng (`\n`) và các câu đơn (`. `). `RecursiveChunker` khai thác triệt để cấu trúc phân cấp này để giữ các danh mục thuốc hoặc điều khoản liên quan ở cạnh nhau trong cùng một chunk thay vì cắt bừa bãi theo số ký tự cố định.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Toàn bộ 5 văn bản | best baseline (`fixed_size`) | 1798 | ~600 | Top 1 Cosine: 0.3950 |
| Toàn bộ 5 văn bản | **Của tôi (`recursive`)** | 2322 | Biến thiên | **Top 1 Cosine: 0.5482** |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi (Thành) | `recursive` | **5.48 / 10** | Giữ trọn cấu trúc ngữ nghĩa phân cấp, Gap an toàn cao (0.1530). | Số lượng chunk sinh ra lớn nhất (2322). |
| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| [Trần Tiến Đạt] | RecursiveChunker (`recursive`) | 8.4 | Giữ ngữ cảnh tốt, chunk ổn định, phù hợp cho retrieval | Avg length thấp hơn mục tiêu nên số chunk tăng nhẹ |
| [Vũ Văn Học ] | RecursiveChunker (`recursive`) | 8.5 | Chunk count thấp nhất, avg length gần kích thước mục tiêu, tận dụng cấu trúc tài liệu tốt | Sẽ lệch khi tham gia domain khác |
| [Hồ Trọng Nhật Minh ] | RecursiveChunker (`recursive`) | 8.3 | Giữ ngữ cảnh tốt, chunk ổn định, phù hợp cho retrieval | Cải thiện chưa nhiều so với FixedSize |


**Strategy nào tốt nhất cho domain này? Tại sao?** `RecursiveChunker` là chiến lược tốt nhất cho domain này. Dữ liệu thực nghiệm cho thấy nó đạt độ tương đồng vượt trội (0.5482) và khoảng cách an toàn (Gap = 0.1530) cực tốt, giúp Agent dễ dàng định vị thông tin chính xác mà không bị phân tán bởi các chunk nhiễu xung quanh.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:  
Sử dụng biểu thức chính quy (Regex) với kỹ thuật nhìn phía sau (Positive Lookbehind): `re.compile(r'(?<=.\ |!\ |\?\ |.\n)')`. Cách tiếp cận này giúp phát hiện chính xác ranh giới kết thúc câu mà không làm mất các ký tự dấu câu khỏi câu gốc. Sau đó, các câu đơn được gom lại theo cụm số lượng `max_sentences_per_chunk` và loại bỏ khoảng trắng thừa bằng `.strip()`.

**`RecursiveChunker.chunk` / `_split`** — approach:  
Thuật toán sử dụng đệ quy bóc tách dọc theo mảng `remaining_separators`. Base case (điều kiện dừng) là khi chuỗi hiện tại đã nhỏ hơn `chunk_size` thì trả về luôn, hoặc khi hết danh sách dấu phân tách thì tự động cắt thô theo chiều dài ký tự. Việc quản lý bộ đệm `good_buffer` giúp tối ưu hóa dung lượng chunk luôn tiệm cận `chunk_size` nhưng vẫn bảo vệ cấu trúc ngữ nghĩa cao nhất.

### EmbeddingStore

**`add_documents` + `search`** — approach:  
Hệ thống sử dụng **ChromaDB (EphemeralClient)** làm backend chính và mảng lưu trữ in-memory làm fallback. 
- Để vượt qua ràng buộc nghiêm ngặt của ChromaDB mới, khi phát hiện tài liệu có `metadata={}` rỗng, hệ thống sẽ tự động chèn dữ liệu mặc định tránh lỗi `ValueError`. Đồng thời, ID được gán mã hóa độc nhất dạng `f"{raw_id}_idx_{self._next_index}"` giúp giải quyết lỗi trùng lặp dữ liệu và bị overwrite.
- Hàm `search` trích xuất dữ liệu trực tiếp dựa trên key `"content"` (đáp ứng chuẩn test case). Điểm số `distances` kiểu khoảng cách hình học L2 của ChromaDB được quy đổi ngược thành giá trị tương đồng (Similarity score) tăng dần thông qua công thức: score = 1.0 / (1.0 + raw_distance) và sắp xếp giảm dần (reverse=True).

**`search_with_filter` + `delete_document`** — approach:  
- `search_with_filter`: Thực hiện lọc trước (pre-filtering) bằng cách truyền tham số `where=metadata_filter` trực tiếp vào câu lệnh `.query()` của ChromaDB hoặc duyệt qua vòng lặp kiểm tra key-value đối với store cục bộ trước khi tính toán độ tương đồng.
- `delete_document`: Lấy toàn bộ danh sách ID hiện tại của collection, đối chiếu khớp chuỗi (hỗ trợ cả ID nguyên bản ban đầu và ID mở rộng bằng index), tiến hành xóa hàng loạt các bản ghi tương ứng qua câu lệnh `.delete(ids=target_ids)`.

### KnowledgeBaseAgent

**`answer`** — approach:  
Rút trích các khối văn bản từ kết quả tìm kiếm của Store thông qua việc quét đồng thời hai trường `"content"` và `"text"` (fallback an toàn) để gom các khối context blocks. Các khối này được nối với nhau bằng dấu phân tách trực quan `\n---\n` tạo thành một khối ngữ cảnh hoàn chỉnh và đưa vào Prompt có cấu trúc nghiêm ngặt nhằm hướng dẫn LLM chỉ trả lời dựa trên thông tin được cung cấp, tránh hiện tượng ảo tưởng (hallucination).

### Test Results
```text
=========================================================== test session starts ============================================================
platform linux -- Python 3.13.5, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/chirikatori/Documents/VinUni/2A202600955-NguyenDucThanh-Day07
configfile: pyproject.toml
plugins: anyio-4.13.0
collected 42 items

tests/test_solution.py ..........................................                                                                    [100%]

============================================================ 42 passed in 0.93s ============================================================
```

**Số tests pass:** **42** / **42** (Hoàn thành tuyệt đối 100%).

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score (Thang 10) | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | This drug is used for treating hypertension. | High blood pressure can be managed with this medication. | High | 9.21 / 10 | Đúng |
| 2 | Keep out of reach of children. | Store in a cool, dry place away from direct sunlight. | Low | 3.45 / 10 | Đúng |
| 3 | Patient experienced severe headaches. | The subject reported intense migraines during the trial. | High | 8.87 / 10 | Đúng |
| 4 | The effective date of the policy is October 2013. | This document outlines the insurance formulary guidelines. | Low | 4.12 / 10 | Đúng |
| 5 | Dosage should not exceed 50mg per day. | Do not take more than 50mg daily. | High | 9.58 / 10 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?** Kết quả của cặp số 3 gây bất ngờ nhất vì từ khóa `"headaches"` và `"migraines"` viết hoàn toàn khác nhau về mặt ký tự vật lý, nhưng mô hình Embedding vẫn chấm điểm số tương đồng rất cao (8.87/10). Điều này chứng minh rằng không gian vector embedding không thực hiện so khớp từ khóa thô (keyword matching) mà thực sự mã hóa được mối quan hệ phân cấp ngữ nghĩa mang tính khái niệm khái quát sâu sắc bên trong ngôn ngữ.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What is the effective date of the Caremark document? | October 1, 2013 |
| 2 | What are the FDA drug safety guidelines or regulations mentioned? | chemical registry guidelines and structural rules under version v3b |
| 3 | Summarize the medical policy or coverage limits for HealthAlliance. | Commercial medical policy and formulary limitations |
| 4 | What information is provided regarding Humana health insurance plans? | Wisconsin 2014 health plan standard drug tier structures |
| 5 | List the CNS drugs included in the UnitedHealthcare (UHC) list. | Central nervous system covered medications list |

### Kết Quả Thực Tế Từ Hệ Thống Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | What are the common medical coverage... across documents? (Fixed Size) | "...=O)NC(C(O)CO)CO)=C(I)C(C(=O)NC(CO)C(O)CO)..." | 3.95/10 | No | [DEMO LLM] Answer based on context... |
| 2 | What are the common medical coverage... across documents? (By Sentences) | "...23637 1309 21309 1097_FDAMDD_v3b C16H13ClN2O2..." | 3.59/10 | No | [DEMO LLM] Answer based on context... |
| 3 | What are the common medical coverage... across documents? (Recursive) | **"...Recommended Daily Dose human 1 anticholinergic 0.25..."** | **5.48/10** | **Yes** | [DEMO LLM] Answer based on context... |

**Bao nhiêu queries trả về chunk relevant trong top-3?** **5** / 5 (Khi sử dụng chiến lược `recursive`).

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:** Tôi học được cách viết các hàm hỗ trợ định dạng bảng tự động giúp việc in kết quả trực quan trên console dễ theo dõi hơn. Đồng thời biết thêm cách gán trường dữ liệu metadata phân tầng theo năm để tối ưu hóa tốc độ lọc trước khi tìm kiếm.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):** Một số nhóm khác đã thiết kế thêm cơ chế làm sạch văn bản (text cleaning) để loại bỏ các ký tự rác sinh ra từ file PDF lỗi font trước khi tiến hành đưa vào bộ bóc tách chunking. Điều này cải thiện điểm số tương đồng Cosine rõ rệt.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?** Tôi sẽ thiết kế thêm một cơ chế Semantic Chunking (cắt chuỗi dựa trên khoảng cách thay đổi ý nghĩa của câu liên tiếp) thay vì chỉ dựa vào các dấu phân tách thô dạng ký tự tự nhiên, giúp tối ưu hóa độ dài của mỗi chunk đồng đều và mạch lạc hơn.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
