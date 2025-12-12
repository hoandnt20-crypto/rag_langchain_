import clip
import os
import logging

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import logging as hf_logging


# Turn off sentence_transformers log
logging.getLogger("sentence_transformers").setLevel(logging.CRITICAL)
hf_logging.set_verbosity_error()


# Init logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


def load_text_embedding_model(device):
    logger.info("Loading text embedding model ...")
    model = SentenceTransformer(os.getenv("TEXT_MODEL_ID"), token=os.getenv("HF_TOKEN"), device=device)
    logger.info("Text embedding model loaded successfully")
    return model.eval()


def load_clip_model(device):
    logger.info("Loading clip model ...")
    model, preprocess = clip.load(os.getenv("CLIP_MODEL_ID"), device=device)    
    logger.info("Clip model loaded successfully")
    return model.eval(), preprocess
