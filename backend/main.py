from PIL import Image
from io import BytesIO
from pathlib import Path
from retrieval import ClipRetrieval
from dataset import Dataset

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


app = FastAPI(title="CLIP Image Retrieval API")

# Initialize Data, CLIP retrieval system
dataset = Dataset()
retriever = ClipRetrieval(dataset)



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
    return {"status": "ok", "model": "CLIP-RN50", "num_images": len(dataset.keyframes)}


@app.post("/search/text")
async def search_by_text(query: str = Form(...), top_k: int = Form(100)):
    try:
        retriever.search_text(query, top_k=top_k)
        return retriever.collect_results() 
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...), top_k: int = Form(100)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        retriever.search_image(image, top_k=top_k)
        return retriever.collect_results()
     
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
