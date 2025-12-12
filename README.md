# Video Keyframe Retrieval System

Hệ thống tìm kiếm keyframe video đa phương thức sử dụng CLIP và Sentence Transformers, hỗ trợ tìm kiếm bằng hình ảnh, văn bản CLIP, transcription và description.

---

## 🚀 Tính năng

### 4 Phương thức Tìm kiếm
1. **CLIP Text Search** - Tìm keyframe bằng mô tả văn bản (sử dụng CLIP model)
2. **CLIP Image Search** - Tìm keyframe tương tự bằng upload ảnh (sử dụng CLIP model)
3. **Transcription Search** - Tìm keyframe qua nội dung transcription (sử dụng Sentence Transformer)
4. **Description Search** - Tìm keyframe qua mô tả video (sử dụng Sentence Transformer)

### Công nghệ
- **Dual Model Architecture**: CLIP (RN50) + Sentence Transformers
- **FastAPI Backend**: RESTful API với CORS support
- **Modern Frontend**: Giao diện web responsive với animations
- **Pre-computed Embeddings**: Tìm kiếm nhanh với embeddings đã tính sẵn
- **Temporal Mapping**: Mapping keyframe với thông tin temporal (frame_idx, pts_time)

---

## 📁 Cấu trúc dự án

```
rag_langchain_/
├── backend/                        # FastAPI backend
│   ├── main.py                    # API endpoints và khởi tạo models
│   ├── config.py                  # Cấu hình đường dẫn và data models
│   ├── models.py                  # Load CLIP và Sentence Transformer models
│   ├── dataset.py                 # Dataset class quản lý data
│   ├── retrieval.py               # Logic tìm kiếm (ClipRetrieval, TextRetrieval)
│   ├── utils.py                   # Helper functions (load embeddings, mapping)
│   └── test.ipynb                 # Notebook để test
│
├── data/                          # Dữ liệu và embeddings
│   ├── embs/                      # Pre-computed embeddings
│   │   ├── clip/                  # CLIP embeddings (.npy files)
│   │   ├── transcription/         # Transcription embeddings
│   │   └── description/           # Description embeddings
│   ├── info/                      # Metadata
│   │   ├── media/                 # Video info (title, watch_url)
│   │   ├── transcription/         # Transcription text
│   │   └── description/           # Description text
│   ├── keyframes/                 # Keyframe images (organized by video)
│   └── map-keyframes/             # Temporal mapping (frame_idx, pts_time, fps)
│
├── frontend/                      # Web UI
│   ├── index.html                # Main HTML
│   ├── style.css                 # Styling
│   ├── app.js                    # JavaScript logic
│   └── README.md                 # Frontend docs
│
├── .env                          # Environment variables (HF_TOKEN, model IDs)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🛠️ Cài đặt

### 1. Clone repository
```bash
git clone <repo-url>
cd rag_langchain_
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

**Dependencies chính:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `torch` + `torchvision` - Deep learning
- `clip` - OpenAI CLIP model
- `sentence-transformers` - Text embeddings
- `pillow` - Image processing
- `python-dotenv` - Environment variables

### 3. Cấu hình environment variables

Tạo file `.env` trong thư mục root:
```env
HF_TOKEN=your_huggingface_token
TEXT_MODEL_ID=your_text_embedding_model_id
CLIP_MODEL_ID=your_clip_model_id
```

### 4. Chuẩn bị dữ liệu

Đảm bảo cấu trúc thư mục `data/` đầy đủ:
- Keyframes trong `data/keyframes/`
- Embeddings trong `data/embs/`
- Metadata trong `data/info/`
- Temporal mapping trong `data/map-keyframes/`

---

## 🎯 Sử dụng

### 1. Chạy API Server

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

API sẽ chạy tại: `http://localhost:8000`

### 2. Mở Web Demo

Mở `frontend/index.html` trong trình duyệt hoặc dùng Live Server.

---

## 📡 API Endpoints

### Health Check
```bash
GET /health
```
**Response:**
```json
{
  "status": "ok",
  "model": "your_clip_model_id",
  "num_images": 12345
}
```

### 1. CLIP Text Search
```bash
POST /search/clip_text
Content-Type: multipart/form-data

query: "a person playing guitar"
top_k: 100
```

### 2. CLIP Image Search
```bash
POST /search/clip_image
Content-Type: multipart/form-data

file: <image_file>
top_k: 100
```

### 3. Transcription Search
```bash
POST /search/transcription
Content-Type: multipart/form-data

query: "machine learning tutorial"
top_k: 100
```

### 4. Description Search
```bash
POST /search/description
Content-Type: multipart/form-data

query: "cooking recipe video"
top_k: 100
```

### Response Format
Tất cả endpoints trả về danh sách `SearchResult`:
```json
[
  {
    "video_name": "L21_V001",
    "title": "60 Giây Sáng - Ngày 01082024 - HTV Tin Tức Mới Nhất 2024",
    "watch_url": "https://youtube.com/embed/Rzpw5WR7nAY",
    "keyframes": [
      {
        "path": "keyframes/L21_V001/170.jpg",
        "frame_idx": 20702,
        "pts_time": 690.067,
        "similarity": 0.5121440857687809
      },
      {
        "path": "keyframes/L22_V023/232.jpg",
        "frame_idx": 23490,
        "pts_time": 939.6,
        "similarity": 0.4272028442152153
      }
    ]
  }
]
```

### Static Files
```bash
GET /static/keyframes/{video_folder}/{frame_number}.jpg
```

---

## 🏗️ Kiến trúc hệ thống

### Models

#### 1. CLIP Model (`models.py`)
```python
load_clip_model(device) -> (model, preprocess)
```
- Model: your_clip_model_id
- Sử dụng cho: CLIP text/image search
- Output: 1024-dim embeddings (depends on model)

#### 2. Text Embedding Model (`models.py`)
```python
load_text_embedding_model(device) -> model
```
- Model: your_text_embedding_model_id
- Sử dụng cho: Transcription/Description search
- Output: 384-dim embeddings (depends on model)

### Dataset (`dataset.py`)

Class `Dataset` quản lý toàn bộ data:
```python
dataset = Dataset()
dataset.clip_embs              # CLIP embeddings
dataset.transcription_embs     # Transcription embeddings
dataset.description_embs       # Description embeddings
dataset.media_info            # Video metadata
dataset.transcription_info    # Transcription text + temporal mapping
dataset.description_info      # Description text + temporal mapping
dataset.keyframes             # Keyframe paths
dataset.map_keyframes         # Temporal info (frame_idx, pts_time, fps)
```

### Retrieval Classes (`retrieval.py`)

#### 1. ClipRetrieval
```python
clip_retriever = ClipRetrieval(model, preprocess, device)
clip_retriever.search_text(query, dataset, top_k)
clip_retriever.search_image(image, dataset, top_k)
results = clip_retriever.collect_results(dataset)
```

#### 2. TextRetrieval
```python
text_retriever = TextRetrieval(model, support_model, device)
text_retriever.search_text(query, dataset, "transcription", top_k)
results = text_retriever.collect_results(dataset, "transcription", top_k)
```

**Đặc biệt:** TextRetrieval sử dụng CLIP model như support model để chọn keyframe tốt nhất từ các keyframe có cùng transcription/description.

---

## 🎨 Frontend Features

- **4 Search Tabs**: CLIP Text, CLIP Image, Transcription, Description
- **Drag & Drop**: Upload ảnh dễ dàng
- **Real-time Preview**: Xem trước ảnh upload
- **Video Results**: Hiển thị kết quả theo video với YouTube embed
- **Keyframe Gallery**: Xem tất cả keyframe tìm được với similarity scores
- **Responsive Design**: Tương thích mobile/desktop
- **Modern UI**: Animations, glassmorphism, gradient backgrounds

---

## 📊 Performance

- **CLIP Model**: your_clip_model_id 
- **Text Model**: your_text_embedding_model_id 
- **Search Speed**: ~10-100ms per query (depends on dataset size)
- **Embedding Dimensions**: 
  - CLIP: 1024
  - Text: 384 (default)
- **Similarity Metric**: Cosine similarity

---

## 🔧 Customization

### Thay đổi CLIP model
Trong `.env`:
```env
CLIP_MODEL_ID=your_clip_model_id
```

### Thay đổi Text Embedding model
Trong `.env`:
```env
TEXT_MODEL_ID=your_text_embedding_model_id
```

### Điều chỉnh temporal expansion
Trong `dataset.py`:
```python
# Mở rộng temporal window cho description search
self.description_info = mapping_temporal_keyframe(
    self.description_info, 
    self.map_keyframes, 
    expand_temporal=4  # ±4 keyframes
)
```

---

## 🐛 Troubleshooting

### CORS Error
Đảm bảo API server đang chạy và CORS middleware đã được cấu hình trong `main.py`.

### Model Download Failed
- Kiểm tra kết nối internet
- Đảm bảo `HF_TOKEN` hợp lệ trong `.env`
- CLIP model sẽ tự động download lần đầu (~350MB)

### Keyframe Not Found
- Kiểm tra đường dẫn trong `config.py`
- Đảm bảo structure `data/keyframes/{video_name}/{frame}.jpg`

### Out of Memory
- Giảm batch size khi tính embeddings
- Sử dụng CPU thay vì GPU: `device = "cpu"`
- Giảm `top_k` trong search

---

## 📚 References

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [OpenAI CLIP GitHub](https://github.com/openai/CLIP)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## 📝 Notes

- Embeddings được pre-compute để tăng tốc độ search
- CLIP hỗ trợ zero-shot learning, không cần training
- Temporal mapping giúp tìm đúng thời điểm trong video
- Support model (CLIP) trong TextRetrieval giúp chọn keyframe tốt nhất khi có nhiều keyframe match cùng text
