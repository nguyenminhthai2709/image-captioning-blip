# BÀI TẬP LỚN COMPUTER VISION
## Đề tài: Image Captioning using Vision-Language Models with Fine-Tuning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-BLIP-yellow.svg)](https://huggingface.co/docs/transformers/model_doc/blip)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

---

## 1. Giới thiệu Đề tài (Project Overview)

Image Captioning (Sinh chú thích ảnh tự động) là bài toán giao thoa đa phương thức (Multimodal) giữa **Thị giác máy tính (Computer Vision)** và **Xử lý ngôn ngữ tự nhiên (NLP)**. Nhiệm vụ của hệ thống là tiếp nhận một bức ảnh đầu vào, hiểu được các thực thể, thuộc tính và mối quan hệ không gian, sau đó sinh ra một câu mô tả chính xác nội dung bằng ngôn ngữ tự nhiên.

Trong dự án này, hệ thống nghiên cứu và đối chiếu chuyên sâu **ba cấu hình mô hình Vision-Language**:
1. **BLIP Pretrained (`Salesforce/blip-image-captioning-base`)**: Zero-shot baseline kết hợp **Vision Transformer ViT-B/16 (384px)** và **Cross-Attention Text Decoder**.
2. **BLIP Fine-Tuned (`checkpoints/frozen_vision/best_model/`)**: Mô hình BLIP được tinh chỉnh bộ giải mã trên Flickr8k với chiến lược Frozen Vision Backbone.
3. **ViT-GPT2 Pretrained (`nlpconnect/vit-gpt2-image-captioning`)**: Kiến trúc lai ghép độc lập kết hợp **ViT (224px)** và **GPT-2 Causal LM**.

**Các đóng góp & tính năng chính:**
- Đã tải và tổ chức toàn bộ tập dữ liệu **Flickr8k thật** (8,091 ảnh JPEG, 40,460 chú thích thật kèm các file phân chia train/val/test chuẩn Karpathy).
- Xây dựng module suy luận độc lập [`src/inference.py`](file:///c:/Users/BOOK%20PRO/Downloads/btl%20computer%20vision/src/inference.py) và [`src/vit_gpt2_captioner.py`](file:///c:/Users/BOOK%20PRO/Downloads/btl%20computer%20vision/src/vit_gpt2_captioner.py) tự động nhận diện phần cứng (CUDA GPU -> MPS -> CPU fallback).
- Đánh giá định lượng toàn diện trên cùng 1,000 ảnh test chuẩn bằng: **BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, ROUGE-L** và đo lường thời gian suy luận thực tế.
- Trực quan hóa bản đồ chú ý **Cross-Attention Heatmap** (Visual Grounding) theo từng từ ngữ hoàn chỉnh.
- Triển khai ứng dụng web tương tác bằng **Streamlit** hỗ trợ so sánh đối đầu đồng thời 3 mô hình trên cùng một ảnh.

---

## 2. Phân Tích Chuyên Sâu Dưới Góc Nhìn Computer Vision

```mermaid
flowchart TD
    subgraph "1. Image Preprocessing & Tensorization"
        A["Image (PIL RGB: H x W x 3)"] --> B["Bicubic Resize (384 x 384)"]
        B --> C["Pixel Scaling [0, 1] & ImageNet Normalization"]
        C --> D["Vision Tensor: X ∈ ℝ^(1 x 3 x 384 x 384)"]
    end

    subgraph "2. Patch Extraction & Linear Embedding (ViT-B/16)"
        D --> E["Cắt lưới N=576 Patches (P = 16 x 16 px)"]
        E --> F["Flatten mỗi patch: x_p ∈ ℝ^(768)"]
        F --> G["Chiếu tuyến tính (Linear Projection E)"]
        G --> H["Bổ sung [CLS] Token & Position Embeddings E_pos"]
        H --> I["Chuỗi Visual Tokens z_0 ∈ ℝ^(1 x 577 x 768)"]
    end

    subgraph "3. Visual Feature Extraction (Vision Transformer Encoder)"
        I --> J["12 Khối Multi-Head Self-Attention (MSA) + MLP"]
        J --> K["Học quan hệ không gian cục bộ & toàn thể"]
        K --> L["Visual Representation H_vis ∈ ℝ^(1 x 577 x 768)"]
    end

    subgraph "4. Cross-Modal Fusion & Language Generation"
        L --> M["Visual Tokens đóng vai trò Keys (K) & Values (V)"]
        N["Text Prompt / Tokens trước w_<t"] --> O["Text Decoder (Queries Q)"]
        M & O --> P["Cross-Attention: Softmax(QK^T / √d) V"]
        P --> Q["Linear Head -> Softmax -> Token tiếp theo w_t"]
        Q --> R["Output Caption: 'a little girl standing in front of a chicken coop'"]
    end
```

### 2.1. Ảnh RGB được tiền xử lý như thế nào? (Image $\to$ Preprocessing)
* **Quy trình chuẩn hóa:** Tiền xử lý thị giác được thực hiện trực tiếp bởi `BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")` để đảm bảo độ tương thích tuyệt đối với pretrained checkpoint.
* **Spatial Resizing:** Ảnh thô (PIL RGB) được biến đổi về kích thước chuẩn $(384 \times 384)$ px bằng thuật toán **Bicubic Interpolation**.
* **Scaling:** Giá trị pixel từ số nguyên rời rạc $[0, 255]$ được chuẩn hóa về số thực trong khoảng $[0.0, 1.0]$.

### 2.2. Tensor biểu diễn ảnh như thế nào? (Preprocessing $\to$ Tensor)
* **Định dạng Chiều:** Chuyển đổi từ định dạng ảnh $\text{HWC}$ sang Tensor PyTorch $\text{BCHW}$:
  $$X \in \mathbb{R}^{B \times 3 \times 384 \times 384} \quad (\text{với } B=1 \text{ khi suy luận đơn ảnh})$$
* **Chuẩn hóa Phân phối Kênh màu (BLIP Processor Normalization - ImageNet-derived):**
  $$X_{\text{norm}}(c, i, j) = \frac{X(c, i, j) - \mu_c}{\sigma_c}$$
  với các tham số cấu hình của BLIP Processor:
  * $\mu = [0.48145466, 0.4578275, 0.40821073]$
  * $\sigma = [0.26862954, 0.26130258, 0.27577711]$
* **Ý nghĩa:** Giá trị pixel sau chuẩn hóa nằm trong đoạn $[-1.792, 2.146]$ với kỳ vọng $\text{mean} \approx 0$ và phương sai $\text{std} \approx 1$. Điều này giúp các hàm kích hoạt phi tuyến trong Transformer (như GeLU) tránh hiện tượng bão hòa gradient.

### 2.3. Vision Transformer xử lý Image Patches như thế nào? (Tensor $\to$ Vision Encoder)
* **Patch Partitioning:** Mạng $\text{ViT-B/16}$ chia ma trận ảnh $(3, 384, 384)$ thành lưới các mảnh vuông kích thước $P = 16 \times 16$ px:
  $$N = \left(\frac{384}{16}\right) \times \left(\frac{384}{16}\right) = 24 \times 24 = 576 \text{ patches}$$
* **Patch Flattening & Linear Projection:** Mỗi patch 2D $(16 \times 16 \times 3)$ được trải phẳng thành vector 1D ($D_{\text{raw}} = 768$) và nhân với ma trận chiếu học được $E \in \mathbb{R}^{768 \times 768}$.
* **Bổ sung Tọa độ Không gian:** Do cơ chế Self-Attention có tính bất biến theo vị trí (permutation-invariant), mô hình cộng thêm vector vị trí học được $E_{\text{pos}} \in \mathbb{R}^{577 \times 768}$ và chèn thêm 1 token toàn cục $x_{\text{class}}$ (`[CLS]`):
  $$z_0 = [x_{\text{class}}; x_p^1 E; x_p^2 E; \dots; x_p^{576} E] + E_{\text{pos}} \in \mathbb{R}^{1 \times 577 \times 768}$$

### 2.4. Ý nghĩa của Visual Representation (Vision Encoder $\to$ Visual Representation)
* Chuỗi $577$ visual tokens đi qua $12$ khối Transformer Encoder (Multi-Head Self-Attention + MLP).
* **Bản chất học được:** Từng patch trao đổi thông tin với toàn bộ các patch khác trên ảnh.
  * Các tầng thấp nhận diện đặc trưng cục bộ (màu sắc, hoa văn, cạnh góc).
  * Các tầng cao tổng hợp ngữ cảnh toàn thể và quan hệ không gian.
* **Đầu ra:** Ma trận biểu diễn thị giác hoàn chỉnh:
  $$H_{\text{vis}} \in \mathbb{R}^{1 \times 577 \times 768}$$

### 2.5. Visual Representation được sử dụng để sinh Caption như thế nào? (Visual Representation $\to$ Language Generation $\to$ Caption)
* **Cơ chế Cross-Attention Đa phương thức:**
  * Ma trận đặc trưng thị giác $H_{\text{vis}}$ được chiếu tuyến tính thành **Keys ($K$)** và **Values ($V$)**:
    $$K = H_{\text{vis}} W_K, \quad V = H_{\text{vis}} W_V$$
  * Các từ ngữ đã được sinh ra trước đó $w_{<t}$ được chiếu thành **Queries ($Q$)**:
    $$Q = H_{\text{text}} W_Q$$
  * Tính ma trận tương đồng Cross-Attention:
    $$\text{Cross-Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
  * **Ý nghĩa:** Khi sinh đến từ `"chicken"`, truy vấn $Q$ sẽ kích hoạt trọng số chú ý cao nhất tại các patch $K_j$ nằm ở khu vực chuồng gà trên ảnh.
* **Beam Search Decoding ($k=5$):** Duy trì top 5 chuỗi từ có xác suất đồng thời cao nhất để sinh ra câu văn tự nhiên:
  $$\hat{S} = \arg\max_{S} \sum_{t=1}^{T} \log P(w_t \mid w_{<t}, H_{\text{vis}})$$
* **Kết quả:** `"a little girl standing in front of a chicken coop"`.

### 2.6. Visual Grounding: Phân tích Cross-Attention Heatmap theo Word hoàn chỉnh (Word-Level Grounding)

> **Khái niệm Học thuật:**
> *"Cross-attention heatmap visualizes the relative attention weights assigned by the text decoder to visual patches when generating a specific token or word."*
>
> - Heatmap được sử dụng để **giải thích hành vi mô hình (Interpretability)** và phân tích tương quan thị giác - ngôn ngữ (Visual Grounding).
> - Heatmap **không phải** là ground-truth bounding box và **không phải** là kết quả Object Detection chính xác tuyệt đối. Vùng có trọng số attention cao phản ánh mức độ tập trung đặc trưng của Decoder lên các patch tương ứng, không đồng nghĩa chắc chắn mô hình đang localization đối tượng chuẩn xác nếu không có nhãn ground-truth localization để kiểm chứng.

**Pipeline Xử Lý:**
$$\text{Word} \longrightarrow \text{Subword tokens} \longrightarrow \text{Cross-Attention} \longrightarrow 576 \text{ visual patches} \longrightarrow 24 \times 24 \text{ spatial grid} \longrightarrow \text{Heatmap} \longrightarrow \text{Overlay}$$

#### A. Tại sao một Word có thể gồm nhiều Subword Tokens?
* BLIP sử dụng bộ từ vựng cố định (~30,522 tokens) dựa trên thuật toán **WordPiece Tokenization (BERT)**.
* Khi gặp các từ hiếm, từ ghép hoặc biến thể hình thái học (ví dụ: từ ngữ cảnh thời trang `"croche"`, từ ghép `"motorcycles"`), tokenizer chia nhỏ thành các **subwords**:
  $$\text{"croche"} \longrightarrow [13675: \text{"cr"}] + [23555: \text{"##oche"}]$$
  Dấu tiền tố `##` biểu thị subword này nối tiếp liền mạch với subword đứng trước để tạo thành 1 từ hoàn chỉnh.

#### B. Tại sao cần gộp Cross-Attention (Attention Aggregation)?
* Trong Text Decoder, mỗi subword token (ví dụ: `"cr"` và `"##oche"`) gửi truy vấn Query ($Q$) độc lập đến $576$ visual patches ($K, V$) của $\text{ViT-B/16}$.
* Để quan sát sự chú ý thị giác của **toàn bộ khái niệm ngữ nghĩa (semantic word)**, hệ thống tự động gộp các vector attention tương ứng bằng phép toán trung bình cộng:
  $$A_{\text{word}} = \frac{1}{|T_{\text{word}}|} \sum_{t \in T_{\text{word}}} A_t \in \mathbb{R}^{576}$$
  Với từ chỉ gồm 1 token duy nhất (như `"wearing"`, `"hat"`), vector attention được sử dụng trực tiếp ($|T|=1$).

#### C. Xử lý Trùng Lặp Từ (Repeated Words Disambiguation)
* Khi một từ xuất hiện nhiều lần ở các vị trí ngữ cảnh khác nhau (ví dụ: `"a man wearing a croche hat ... in a room"` có 3 từ `"a"`):
  * Hệ thống quản lý theo bộ chỉ số định danh: `(word_index, token_indices)`.
  * Vị trí `"a"` #1 ($t=1$), `"a"` #4 ($t=4$), `"a"` #10 ($t=11$) được phân tích hoàn toàn độc lập, kích hoạt các vùng patch thị giác riêng biệt (Patch #311, Patch #48, Patch #408).

#### D. Chuẩn hóa Khử Điểm Trũng Thị Giác ViT (Spatial Z-score Relevance)
* Trong Vision Transformer, một số patch nền/biên có chuẩn vector lớn (Attention Sink) thường nhận attention nền chung từ tất cả các token.
* Thuật toán áp dụng chuẩn hóa $Z$-score theo chuỗi câu để cô lập trọng số kích hoạt đặc thù của từ:
  $$R(t, j) = \text{ReLU}\left(\frac{A(t, j) - \mu_j}{\sigma_j}\right)$$
  Sau đó định hình thành ma trận không gian $24 \times 24$, nội suy song bậc ba (Bicubic) lên kích thước ảnh gốc và tạo bản đồ nhiệt (Heatmap Overlay).

---

## 3. Hướng dẫn Chạy Suy Luận (Pretrained BLIP Inference)

### 3.1. Chạy với ảnh bất kỳ qua Command Line
```bash
python src/inference.py --image data/flickr8k/Images/1000268201_693b08cb0e.jpg
```
*(Nếu không truyền tham số `--image`, script sẽ tự động lấy ảnh mẫu đầu tiên trong tập Flickr8k).*

### 3.2. Tùy chỉnh tham số giải mã
```bash
python src/inference.py --image path/to/your_image.jpg --num_beams 5 --max_length 32
```

### 3.3. Chạy Suy Luận ViT-GPT2 qua Command Line
```bash
# Suy luận đơn ảnh với ViT-GPT2 Pretrained
python src/vit_gpt2_captioner.py --image data/flickr8k/Images/1000268201_693b08cb0e.jpg

# Chạy benchmark đánh giá ViT-GPT2 trên 1,000 ảnh test chuẩn
python scripts/evaluate_vit_gpt2.py --num_samples 1000 --resume
```

### 3.4. Kết quả lưu tự động
Toàn bộ kết quả suy luận cùng thông số chi tiết của pipeline tiền xử lý thị giác được tự động lưu vào thư mục `results/predictions/`:
* File JSON: `results/predictions/<image_name>_prediction.json`
* File Text: `results/predictions/<image_name>_prediction.txt`

---

## 4. Huấn luyện Fine-Tuning Pipeline (Phần 6)

Mô hình hỗ trợ 2 chiến lược fine-tuning linh hoạt:
1. **Frozen Vision ViT (`frozen_vision`):** Đóng băng 86M tham số của Vision Transformer, chỉ huấn luyện 138M tham số của Text Decoder. Tiết kiệm VRAM, huấn luyện nhanh trên GPU nhỏ.
2. **Full Fine-Tuning (`full_finetune`):** Huấn luyện toàn bộ mô hình với **Differential Learning Rate** (ViT học với $\text{LR} = 5 \times 10^{-6}$, Decoder học với $\text{LR} = 5 \times 10^{-5}$) để tránh catastrophic forgetting các đặc trưng thị giác cốt lõi.

### 4.1. Lệnh chạy Huấn luyện (Command Line Interface)
```bash
# Huấn luyện chiến lược Frozen Vision (Khuyên dùng khi bắt đầu)
python src/train.py --strategy frozen_vision --epochs 5 --batch_size 16 --learning_rate 5e-5

# Huấn luyện Full Fine-Tuning (Differential LR cho ViT & Decoder)
python src/train.py --strategy full_finetune --epochs 5 --batch_size 16 --learning_rate 5e-5 --lr_vision 5e-6 --weight_decay 0.05
```

### 4.2. Các siêu tham số cho phép cấu hình (Hyperparameters)
| Tham số | Ý nghĩa | Mặc định |
| :--- | :--- | :--- |
| `--strategy` | Chiến lược huấn luyện (`frozen_vision` hoặc `full_finetune`) | `frozen_vision` |
| `--epochs` | Số lượng epoch huấn luyện | `5` |
| `--batch_size` | Kích thước batch | `16` |
| `--learning_rate` | Tốc độ học của Language Decoder | `5e-5` |
| `--lr_vision` | Tốc độ học của Vision Encoder (khi full fine-tune) | `5e-6` |
| `--max_length` | Độ dài tối đa của caption tokens | `32` |
| `--weight_decay` | Hệ số suy giảm trọng số (AdamW) | `0.05` |
| `--mixed_precision` | Kích hoạt tự động FP16 (AMP) trên GPU | `True` |

### 4.3. Logging và Quản lý Checkpoint
- **Training Logs:** Quá trình huấn luyện tự động ghi lại loss theo từng step và epoch vào:
  * `logs/training_log_<strategy>.json`
  * `logs/training_log_<strategy>.csv`
- **Best Checkpoint:** Sau mỗi epoch validation, nếu validation loss đạt kỷ lục thấp nhất mới, mô hình sẽ tự động lưu trọng số tốt nhất vào:
  * `checkpoints/<strategy>/best_model/`

---

## 5. Kết Quả Đánh Giá & So Sánh Thực Nghiệm (3-Model Benchmark)

Kết quả đánh giá chính thức được thực hiện đối đầu trực tiếp giữa **3 cấu hình mô hình**:
1. **Pretrained BLIP (`Salesforce/blip-image-captioning-base`)**
2. **Fine-Tuned BLIP (`checkpoints/frozen_vision/best_model/`)**
3. **ViT-GPT2 Pretrained (`nlpconnect/vit-gpt2-image-captioning`)**

### 5.1. Giao thức Đánh giá (Evaluation Protocol)
* **Tập dữ liệu kiểm thử:** Toàn bộ $1,000$ ảnh test chuẩn từ `data/flickr8k/splits/Flickr_8k.testImages.txt` (hoàn toàn cô lập khỏi tập train/validation).
* **Ground-Truth References:** $5,000$ câu chú thích chuẩn người dùng từ `data/flickr8k/captions.txt` (đúng $5$ reference captions cho mỗi ảnh test).
* **Cấu hình Sinh chuỗi (Generation Settings):** Cả 3 mô hình sử dụng cùng cấu hình Beam Search:
  * `method = "beam_search"`
  * `num_beams = 3`
  * `max_length = 32`
* **Chỉ số Đánh giá (Metrics):**
  * **BLEU-1, BLEU-2, BLEU-3, BLEU-4:** NLTK `corpus_bleu` kết hợp Chen & Cherry Smoothing Method 1.
  * **METEOR:** NLTK `meteor_score` đối chiếu với 5 references của mỗi ảnh, lấy trung bình toàn tập.
  * **ROUGE-L:** Longest Common Subsequence lớn nhất với 5 references qua `rouge-score` (use_stemmer=True), lấy trung bình toàn tập.
* **Độ toàn vẹn dữ liệu:** $1,000$ ảnh xử lý hoàn chỉnh cho từng model, $0$ ảnh lỗi/thiếu, $0$ giá trị NaN.
* **Môi trường thực thi:** CPU (`PyTorch 2.13.0+cpu`).
* **Dữ liệu xuất chính thức:**
  * BLIP Metrics: [`outputs/full_test_1000.json`](file:///c:/Users/BOOK%20PRO/Downloads/btl%20computer%20vision/outputs/full_test_1000.json) | Predictions: [`outputs/predictions/full_test_1000.json`](file:///c:/Users/BOOK%20PRO/Downloads/btl%20computer%20vision/outputs/predictions/full_test_1000.json)
  * ViT-GPT2 Metrics: [`outputs/vit_gpt2_test_1000.json`](file:///c:/Users/BOOK%20PRO/Downloads/btl%20computer%20vision/outputs/vit_gpt2_test_1000.json) | Predictions: [`outputs/predictions/vit_gpt2_test_1000.json`](file:///c:/Users/BOOK%20PRO/Downloads/btl%20computer%20vision/outputs/predictions/vit_gpt2_test_1000.json)
  * Báo cáo chuyên sâu: [`results/model_comparison.md`](file:///c:/Users/BOOK%20PRO/Downloads/btl%20computer%20vision/results/model_comparison.md)

### 5.2. Bảng Kết Quả Đánh Giá Chính Thức (Official Quantitative Results - 1,000 Test Images)

| Chỉ số (Metric) | 1. Pretrained BLIP (Zero-Shot) | 2. Fine-Tuned BLIP (Flickr8k) | 3. ViT-GPT2 Pretrained | BLIP Fine-Tuned vs ViT-GPT2 | BLIP Pretrained vs ViT-GPT2 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BLEU-1** | 58.99% | **71.15%** | 61.22% | **+9.93 pp** | -2.23 pp |
| **BLEU-2** | 45.08% | **54.86%** | 41.82% | **+13.04 pp** | **+3.26 pp** |
| **BLEU-3** | 33.37% | **40.65%** | 27.01% | **+13.64 pp** | **+6.36 pp** |
| **BLEU-4** | 24.50% | **29.33%** | 16.87% | **+12.46 pp** | **+7.63 pp** |
| **METEOR** | 37.19% | **44.40%** | 36.71% | **+7.69 pp** | **+0.48 pp** |
| **ROUGE-L** | 49.40% | **53.25%** | 45.47% | **+7.78 pp** | **+3.93 pp** |
| **Độ trễ trung bình/ảnh** | ~2,871 ms | ~2,871 ms | **1,061.5 ms** | ViT-GPT2 nhanh hơn ~2.7x | ViT-GPT2 nhanh hơn ~2.7x |
| **Tổng runtime 1,000 ảnh** | 5,742.63 s (~95.7 min) | 5,742.63 s (~95.7 min) | **1,056.10 s (~17.6 min)** | ViT-GPT2 nhanh hơn ~5.4x | ViT-GPT2 nhanh hơn ~5.4x |

*Ghi chú:* pp = percentage points (điểm phần trăm tuyệt đối).

![Biểu đồ so sánh 3 mô hình](results/figures/model_comparison.png)
![Biểu đồ phân rã BLEU 1-4](results/figures/bleu_comparison.png)

### 5.3. Mẫu So Sánh Định Tính Trực Quan (Qualitative Samples)
*Mẫu trích xuất thực tế từ tập test chính thức của cả 3 mô hình:*

| Ảnh (Image) | Ground Truth (5 References) | Pretrained BLIP | Fine-Tuned BLIP | ViT-GPT2 Pretrained | Nhận xét Chuyên Môn |
| :---: | :--- | :--- | :--- | :--- | :--- |
| `3482062809_3b694322c4.jpg` | • A group of tourists stand around as a lady puts her hand near the mouth of a statue .<br>• Woman gets her hand kissed by living statue street artist . | `a man on a skateboard` | `a group of people sitting on a bench in front of a statue of a soldier` | `a man standing on top of a skateboard next to a group of people` | Cả 2 model pretrained đều nhầm lẫn ván trượt; BLIP Fine-Tuned nhận diện chuẩn xác tượng và băng ghế |
| `280706862_14c30d734a.jpg` | • A black dog emerges from the water onto the sand , holding a white object in its mouth .<br>• A black dog on a beach carrying a ball in its mouth . | `the dog is black` | `a black dog with a white ball in it ' s mouth` | `a black dog with a frisbee in its mouth` | BLIP Pretrained quá sơ sài; ViT-GPT2 nhận diện được hành vi ngậm nhưng nhầm bóng thành frisbee; BLIP FT chính xác 100% |
| `3385593926_d3e9c21170.jpg` | • Two brown dogs playfully fight in the snow .<br>• Two dogs playing in the snow . | `two dogs playing in the snow` | `a couple of dogs playing in the snow` | `a brown and black dog playing with a brown horse` | ViT-GPT2 bị ảo giác nhầm chú chó nâu thành con ngựa ("horse"); cả 2 mô hình BLIP nhận diện chuẩn xác |
| `311146855_0b65fdb169.jpg` | • A man and a woman in festive costumes dancing .<br>• one performer wearing a feathered headdress dancing ... | `a man in a costume` | `a man in a yellow and green costume` | `a man in a black shirt and a woman in a white shirt` | BLIP FT miêu tả chi tiết màu sắc lễ hội; ViT-GPT2 miêu tả theo khuôn mẫu trang phục thường ngày |

### 5.4. Phân Tích & Nhận Xét Chuyên Môn
1. **Unigram vs. n-gram:** ViT-GPT2 đạt BLEU-1 cao (61.22% vs. 58.99% của Pretrained BLIP) nhờ vốn từ BPE phong phú của GPT-2, nhưng giảm sút mạnh ở $n$-gram dài (BLEU-4: 16.87% vs. 24.50%). BLIP có khả năng nắm bắt trật tự cú pháp và ngữ cảnh cụm từ chặt chẽ hơn.
2. **Độ phân giải thị giác:** BLIP xử lý ở độ phân giải $384 \times 384$ px (576 visual patches) so với $224 \times 224$ px (196 patches) của ViT-GPT2, giúp giảm thiểu đáng kể hiện tượng ảo giác (hallucination).
3. **Hiệu năng Fine-tuning:** BLIP Fine-Tuned dẫn đầu toàn diện (+9.93 pp BLEU-1, +12.46 pp BLEU-4, +7.69 pp METEOR so với ViT-GPT2).
4. **Tốc độ suy luận:** ViT-GPT2 đạt tốc độ suy luận nhanh hơn ~2.7 lần trên CPU (~1,061 ms/ảnh vs. ~2,870 ms/ảnh).

---

## 6. Vẽ Biểu đồ Hội tụ & Visual Grounding (Visualizer)

```bash
# Vẽ đường cong Training Loss & Validation Loss từ log
python src/visualizer.py --plot_loss logs/training_log_frozen_vision.json

# Tạo bản đồ chú ý thị giác Cross-Attention Heatmaps
python src/visualizer.py --image data/flickr8k/Images/1000268201_693b08cb0e.jpg
```

---

## 7. Cấu trúc Thư mục Dự án

```text
btl-computer-vision/
├── app/
│   └── app.py                      # Giao diện Web tương tác Streamlit (hỗ trợ 3 mô hình)
├── data/
│   ├── flickr8k/                   # Thư mục dữ liệu Flickr8k thật (8,091 ảnh)
│   │   ├── Images/                 # 8,091 ảnh JPEG thật
│   │   ├── captions.txt            # 40,460 captions chuẩn hóa
│   │   ├── Flickr8k.token.txt      # File raw token gốc
│   │   └── splits/                 # Flickr_8k.trainImages.txt, devImages.txt, testImages.txt
│   ├── download_and_extract_flickr8k.py
│   ├── fast_parallel_download.py   # Script tải song song 8 luồng tự động retry
│   └── finalize_dataset.py         # Script chuẩn hóa dữ liệu
├── src/
│   ├── __init__.py
│   ├── config.py                   # Cấu hình hyperparameters & đường dẫn
│   ├── dataset_pipeline.py         # Pipeline xử lý dữ liệu, kiểm tra lỗi & thống kê
│   ├── dataset.py                  # PyTorch Dataset & DataLoader
│   ├── model.py                    # Wrapper BLIP: Freeze/Unfreeze ViT, Differential LR
│   ├── train.py                    # Training loop, LR scheduling, AMP, checkpointing
│   ├── evaluate.py                 # Đánh giá BLIP (BLEU 1-4, METEOR, ROUGE-L)
│   ├── inference.py                # Pipeline suy luận Pretrained BLIP chuẩn hóa
│   ├── vit_gpt2_captioner.py       # Wrapper suy luận và tiền xử lý ViT-GPT2
│   ├── visualizer.py               # Trực quan hóa Attention Heatmap & biểu đồ hội tụ
│   └── utils.py                    # Tiện ích tái lập (seed), logger, I/O
├── scripts/
│   └── evaluate_vit_gpt2.py        # Benchmark ViT-GPT2 trên Flickr8k (hỗ trợ --resume)
├── results/
│   ├── figures/                    # Biểu đồ phân tích EDA, training loss và so sánh 3 mô hình
│   │   ├── model_comparison.png    # Biểu đồ radar/bar tổng quan 3 mô hình
│   │   └── bleu_comparison.png     # Biểu đồ phân rã BLEU-1 đến BLEU-4
│   ├── predictions/                # Kết quả caption dự đoán (JSON + TXT)
│   └── model_comparison.md         # Báo cáo phân tích chuyên sâu đối đầu 3 mô hình
├── outputs/
│   ├── full_test_1000.json         # Metrics đánh giá chính thức BLIP (1,000 test images)
│   ├── vit_gpt2_test_1000.json     # Metrics đánh giá chính thức ViT-GPT2 (1,000 test images)
│   └── predictions/
│       ├── full_test_1000.json     # Chi tiết 1,000 predictions & 5,000 references của BLIP
│       └── vit_gpt2_test_1000.json # Chi tiết 1,000 predictions & 5,000 references của ViT-GPT2
├── compare_experiments.py          # Pipeline so sánh thực nghiệm Pretrained vs Fine-Tuned
├── dataset_analysis.py             # Script phân tích EDA Flickr8k
├── checkpoints/                    # Lưu model weights sau khi train (local)
├── requirements.txt                # Thư viện phụ thuộc
└── README.md                       # Báo cáo học thuật chi tiết
```

---

## 8. Khởi chạy Ứng dụng Demo Streamlit

```bash
streamlit run app/app.py
```
Giao diện cung cấp:
- **Tab 1: Demo Trực Tiếp & Phân Tích Chú Ý Thị Giác:**
  - Chọn ảnh trực tiếp từ kho $8,091$ ảnh Flickr8k hoặc tải ảnh tùy ý.
  - Tùy chọn 3 mô hình suy luận: **BLIP Fine-Tuned**, **BLIP Pretrained**, hoặc **ViT-GPT2 Pretrained**.
  - Nút **"So sánh đồng thời 3 mô hình"**: Thực thi suy luận song song đối đầu trên cùng bức ảnh, hiển thị trực quan caption và thời gian suy luận (latency) của từng mô hình.
  - Phân tích **Visual Attention Heatmap (Visual Grounding)** tương tác theo từng từ khóa để kiểm tra vùng thị giác mà Vision Encoder tập trung.
- **Tab 2: So Sánh Thực Nghiệm (Official Benchmark):**
  - Hiển thị bảng kết quả định lượng đầy đủ (BLEU-1..4, METEOR, ROUGE-L, Runtime, Latency) của cả 3 mô hình trên 1,000 ảnh test chuẩn.
  - Tích hợp biểu đồ trực quan hóa đa chiều so sánh hiệu năng và phân rã BLEU.
