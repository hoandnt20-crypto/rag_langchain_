# CLIP Image Retrieval - Baseline

Hệ thống tìm kiếm ảnh sử dụng CLIP model với khả năng tìm kiếm bằng văn bản (text-to-image) và ảnh (image-to-image).

## 🚀 Tính năng

- **Text-to-Image Search**: Tìm ảnh bằng mô tả văn bản
- **Image-to-Image Search**: Tìm ảnh tương tự bằng cách upload ảnh
- **CLIP Model**: Sử dụng OpenAI CLIP (RN50) để encode
- **Fast API**: RESTful API với FastAPI
- **Beautiful UI**: Giao diện web hiện đại với animations

## 📁 Cấu trúc thư mục data

```
rag_langchain_/
├── data/
│   ├── clip_embs/            # Pre-computed embeddings
│   |── keyframes/            # Image
│   |── map-keyframes/
│   └── media-info

## 🛠️ Cài đặt

### 1. Cài đặt dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Kiểm tra dữ liệu

Đảm bảo bạn có: 
- Ảnh trong `data/images/`
- Embeddings trong `data/clip_embs/clip_image_embeddings.npz`

## 🎯 Sử dụng

### 1. Chạy API Server

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API sẽ chạy tại: `http://localhost:8000`

### 2. Mở Web Demo

Mở file `frontend/index.html` trong trình duyệt hoặc dùng Live Server.

### 3. API Endpoints

#### Health Check
```bash
GET http://localhost:8000/health
```

#### Text Search
```bash
POST http://localhost:8000/search/text
Form Data:
  - query: "a cute dog"
  - top_k: 5
```

#### Image Search
```bash
POST http://localhost:8000/search/image
Form Data:
  - file: <image file>
  - top_k: 5
```

#### Get Image
```bash
GET http://localhost:8000/images/{image_name}
```

## 📊 Đánh giá Baseline

Chạy script đánh giá để xem metrics:

```bash
cd backend
python evaluate.py
```

Kết quả bao gồm:
- **Recall@K**: Độ chính xác tìm kiếm
- **Search Time**: Thời gian tìm kiếm trung bình
- **Sample Results**: Kết quả mẫu cho các query

## 🧪 Test thủ công

### Test với Python

```python
from clip_retrieval import CLIPRetrieval
from PIL import Image

# Initialize
retriever = CLIPRetrieval()

# Text search
results = retriever.search_by_text("a cat", top_k=5)
for img_name, score in results:
    print(f"{img_name}: {score:.4f}")

# Image search
image = Image.open("path/to/image.jpg")
results = retriever.search_by_image(image, top_k=5)
```

### Test với cURL

```bash
# Text search
curl -X POST "http://localhost:8000/search/text" \
  -F "query=a cute dog" \
  -F "top_k=5"

# Image search
curl -X POST "http://localhost:8000/search/image" \
  -F "file=@path/to/image.jpg" \
  -F "top_k=5"
```

## 🎨 Web Demo Features

- **Dual Search Modes**: Tab switching giữa text và image search
- **Drag & Drop**: Kéo thả ảnh để upload
- **Real-time Preview**: Xem trước ảnh upload
- **Beautiful Results**: Hiển thị kết quả với similarity scores
- **Responsive Design**: Tương thích mọi thiết bị

## 📈 Performance

- **Model**: CLIP RN50 (~38M parameters)
- **Search Speed**: ~10-50ms per query (CPU)
- **Embedding Dim**: 1024
- **Similarity**: Cosine similarity

## 🔧 Tùy chỉnh

### Thay đổi CLIP model

Trong `clip_retrieval.py`:
```python
retriever = CLIPRetrieval(model_name="ViT-B/32")  # hoặc RN101, ViT-L/14
```

### Thay đổi số lượng kết quả

Trong API call hoặc web demo, điều chỉnh `top_k` parameter.

## 📝 Notes

- Embeddings được pre-compute để tăng tốc độ search
- CLIP hỗ trợ zero-shot learning, không cần training
- Cosine similarity được dùng để đo độ tương đồng

## 🐛 Troubleshooting

### CORS Error
Đảm bảo API server đang chạy và CORS middleware được cấu hình đúng.

### Image not found
Kiểm tra đường dẫn trong `config.py` và đảm bảo ảnh tồn tại trong `data/images/`.

### CLIP model download
Lần đầu chạy sẽ tải CLIP model (~350MB), cần kết nối internet.

## 📚 References

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [OpenAI CLIP GitHub](https://github.com/openai/CLIP)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
