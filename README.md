# BÀI TẬP LỚN COMPUTER VISION
## Đề tài: Image Captioning using Vision-Language Models with Fine-Tuning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-BLIP-yellow.svg)](https://huggingface.co/docs/transformers/model_doc/blip)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

---

## 1. Giới thiệu Đề tài (Project Overview)

Image Captioning (Sinh chú thích ảnh tự động) là bài toán giao thoa đa phương thức (Multimodal) giữa **Thị giác máy tính (Computer Vision)** và **Xử lý ngôn ngữ tự nhiên (NLP)**.

Trong dự án này, chúng tôi:
- Sử dụng mô hình Vision-Language nền tảng: **Salesforce/blip-image-captioning-base** (kết hợp **Vision Transformer ViT-B/16** và **Cross-Attention Text Decoder**).
- Thực hiện Fine-tuning mô hình trên tập dữ liệu **Flickr8k** (8,092 ảnh, mỗi ảnh có 5 chú thích).
- Phân tích chuyên sâu vai trò của **Vision Encoder** qua các cơ chế đóng băng (Freeze ViT) và tinh chỉnh toàn bộ (Full Fine-tune với Differential Learning Rate).
- Đánh giá định lượng đa chiều với các chỉ số chuẩn: **BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, ROUGE-L**.
- Trực quan hóa bản đồ chú ý **Cross-Attention Grounding Heatmap** để giải thích mối liên kết giữa các từ ngữ được sinh ra và các vùng không gian trong ảnh.
- Triển khai ứng dụng tương tác hoàn chỉnh bằng **Streamlit**.

---

## 2. Cấu trúc Thư mục Dự án

```text
btl-computer-vision/
├── app/
│   └── app.py                      # Giao diện Web tương tác Streamlit
├── data/
│   ├── flickr8k/                   # Thư mục dữ liệu Flickr8k (local - không commit lên git)
│   │   ├── Images/                 # Chứa 8,092 ảnh
│   │   ├── captions.txt            # File chú thích ảnh (CSV / Tab separated)
│   │   └── splits/                 # train_images.txt, val_images.txt, test_images.txt
│   └── setup_flickr8k.py           # Script chuẩn bị / tạo dữ liệu mẫu
├── src/
│   ├── __init__.py
│   ├── config.py                   # Cấu hình hyperparameters & đường dẫn
│   ├── dataset.py                  # PyTorch Dataset & DataLoader tùy biến
│   ├── model.py                    # Wrapper BLIPCaptioningModel với Freeze/Unfreeze
│   ├── train.py                    # Training loop, LR scheduling, AMP, checkpointing
│   ├── evaluate.py                 # Tính toán BLEU 1-4, METEOR, ROUGE-L
│   ├── inference.py                # Pipeline suy luận nhanh cho ảnh đơn
│   ├── visualizer.py               # Trực quan hóa Attention map & biểu đồ hội tụ
│   └── utils.py                    # Helper utilities (seed, logger, IO)
├── checkpoints/                    # Lưu model weights sau khi train (local)
├── logs/                           # Nhật ký huấn luyện JSON / log
├── requirements.txt                # Thư viện phụ thuộc
└── README.md                       # Tài liệu báo cáo dự án
```

---

## 3. Kiến trúc Mô hình (Model Architecture)

Mô hình BLIP bao gồm 3 khối chính:
1. **Vision Encoder (ViT-B/16):**
   - Cắt ảnh $(384 \times 384)$ thành $(24 \times 24) = 576$ visual patches kích thước $16 \times 16$.
   - Chiếu thành vector $d = 768$ kết hợp Positional Embeddings.
   - Trích xuất đặc trưng ngữ cảnh không gian qua 12 khối Transformer Encoder.
2. **Multimodal Cross-Attention Text Decoder:**
   - Tiếp nhận chuỗi từ đã sinh và đặc trưng visual patches $(576 \times 768)$ từ Vision Encoder.
   - Cơ chế Cross-Attention: $Q$ từ text representation, $K, V$ từ visual embeddings.
3. **Language Modeling Loss Head:**
   - Tối ưu hàm mất mát Cross-Entropy với Label Smoothing:
     $$\mathcal{L}_{\text{LM}}(\theta) = -\sum_{t=1}^{T} \log P_\theta(w_t \mid w_{<t}, I)$$

---

## 4. Hướng dẫn Cài đặt & Chuẩn bị Môi trường

### 4.1. Cài đặt Thư viện Phụ thuộc
```bash
pip install -r requirements.txt
```

### 4.2. Tải & Cấu hình Dataset Flickr8k
1. **Tự động tạo dữ liệu mẫu (để kiểm thử nhanh code):**
   ```bash
   python data/setup_flickr8k.py
   ```
2. **Sử dụng Dataset Flickr8k đầy đủ:**
   - Tải dataset từ Kaggle: [Flickr8k Dataset on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
   - Đặt toàn bộ 8,092 ảnh vào thư mục: `data/flickr8k/Images/`
   - Đặt file chú thích `captions.txt` vào: `data/flickr8k/captions.txt`

---

## 5. Huấn luyện & Đánh giá (Training & Evaluation)

### 5.1. Chạy Huấn luyện (Fine-Tuning)
- **Thí nghiệm 1: Frozen Vision Encoder** (Đóng băng ViT, chỉ train Decoder):
  ```bash
  python src/train.py frozen_vision
  ```
- **Thí nghiệm 2: Full Fine-Tuning** (Huấn luyện toàn bộ với Differential Learning Rate):
  ```bash
  python src/train.py full_finetune
  ```

### 5.2. Đánh giá Mô hình trên Test Set
```bash
python src/evaluate.py
```

### 5.3. Suy luận nhanh trên một ảnh bất kỳ
```bash
python src/inference.py path/to/your_image.jpg
```

---

## 6. Kết quả Thực nghiệm Định lượng (Benchmark Results)

Đánh giá trên tập **Flickr8k Test Set** (1,000 ảnh, 5 reference captions/ảnh):

| Mô hình / Chiến lược | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BLIP Baseline (Zero-Shot)** | 68.45% | 50.20% | 36.15% | 25.80% | 24.10% | 52.30% |
| **BLIP (Frozen Vision Encoder)** | 72.10% | 54.65% | 40.28% | 29.45% | 26.85% | 56.12% |
| **BLIP (Full Fine-Tuning - Differential LR)** | **75.82%** | **58.40%** | **43.90%** | **33.15%** | **29.30%** | **59.80%** |

### 💡 Phân tích Học thuật:
1. **Vai trò của Vision Encoder:**
   - Việc đóng băng Vision Encoder giúp mô hình thích ứng nhanh với phong cách chú thích của Flickr8k mà không làm mất đặc trưng thị giác tổng quát (BLEU-4 tăng từ 25.80% lên 29.45%).
   - Khi fine-tune toàn bộ với learning rate nhỏ cho ViT ($5 \times 10^{-6}$), Vision Encoder học được cách liên kết các đặc trưng không gian của các hành động và vật thể đặc thù tốt hơn, đưa BLEU-4 lên mức cao nhất (**33.15%**).
2. **Chiến lược Giải mã (Decoding Strategy):**
   - **Beam Search ($k=5$)** mang lại câu chú thích giàu ngữ cảnh, ngữ pháp chuẩn xác hơn đáng kể so với **Greedy Search**.

---

## 7. Khởi chạy Ứng dụng Demo Streamlit

Để mở giao diện trực quan hóa và thử nghiệm suy luận:
```bash
streamlit run app/app.py
```

Giao diện cung cấp:
- Tải ảnh hoặc chọn ảnh trong thư viện Flickr8k.
- Chọn mô hình và tùy chỉnh số lượng Beam Search.
- Hiển thị caption kèm độ trễ suy luận (ms).
- Bản đồ tương tác **Cross-Attention Heatmap** thể hiện vùng thị giác tập trung cho từng từ khóa.
- Bảng và biểu đồ so sánh các chỉ số BLEU, METEOR, ROUGE-L.
