import torch

from PIL import Image
from io import BytesIO
from pathlib import Path
from retrieval import ClipRetrieval, TextRetrieval
from dataset import Dataset
from models import load_text_embedding_model, load_clip_model

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles



app = FastAPI(title="Image Retrieval API")


# Initialize 
dataset   = Dataset()
device    = "cuda" if torch.cuda.is_available() else "cpu"


# Load models
text_model = load_text_embedding_model(device)
clip_model, clip_preprocess = load_clip_model(device)


# Init retrievers
text_retriever = TextRetrieval(model=text_model, support_model=clip_model, device=device)
clip_retriever = ClipRetrieval(model=clip_model, preprocess=clip_preprocess, device=device)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving keyframes
# Get the absolute path to the data directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
app.mount("/static", StaticFiles(directory=str(DATA_DIR)), name="static")

@app.get("/health")
def health():
    return {"status": "ok", "num_images": len(dataset.keyframes)}


@app.post("/search/clip_text")
async def search_by_text(query: str = Form(...), top_k: int = Form(100)):
    try:
        clip_retriever.search_text(query, dataset, top_k=top_k)
        return clip_retriever.collect_results(dataset) 
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/search/clip_image")
async def search_by_image(file: UploadFile = File(...), top_k: int = Form(100)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        clip_retriever.search_image(image, dataset, top_k=top_k)
        return clip_retriever.collect_results(dataset)
     
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/search/transcription")
async def search_by_transcription(query: str = Form(...), top_k: int = Form(100)):
    try:
        text_retriever.search_text(query, dataset, "transcription", top_k=top_k)
        return text_retriever.collect_results(dataset, "transcription", top_k)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/search/description")
async def search_by_transcription(query: str = Form(...), top_k: int = Form(100)):
    try:
        text_retriever.search_text(query, dataset, "description", top_k=top_k)
        return text_retriever.collect_results(dataset, "description", top_k)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )