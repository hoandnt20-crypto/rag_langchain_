import torch
import numpy as np
import clip

from typing import List
from dataset import Dataset
from utils import norm_vectors
from config import TOP_K, SearchResult

import logging
logging.basicConfig(level=logging.INFO)

class Retrieval:

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.search_result = None

    def search(self, query_emb: np.ndarray, embs: np.ndarray, top_k: int = TOP_K):
        query_emb = norm_vectors(query_emb)
        sims = (embs @ query_emb.T).squeeze()
        idxs = np.argsort(-sims)[:top_k]
        self.search_result = {
            "indexes": idxs.tolist(), 
            "similarity": [s.item() for s in sims[idxs]]
        }

    def collect_results(self) -> List[SearchResult]:
        if self.search_result is None:
            raise ValueError("No search has been performed yet.")
        
        results = {}
        samples = self.dataset.get_items(self.search_result["indexes"])
        for sim, sample in zip(self.search_result["similarity"], samples):
            video_name = sample.media_info.name
            # Clean the keyframe path - remove '../data/' prefix and convert backslashes to forward slashes
            keyframe = sample.keyframe.replace("../data/", "").replace("\\", "/")
            watch_url = sample.media_info.watch_url.replace("watch?v=", "embed/")
            frame_idx = int(sample.map_keyframe.frame_idx.item())
            pts_time = float(sample.map_keyframe.pts_time.item())

            if video_name not in results:
                results[video_name] = SearchResult(
                    video_name=video_name,
                    watch_url=watch_url,
                    keyframe=[keyframe],
                    frame_idx=[frame_idx],
                    pts_time=[pts_time],
                    similarity=[sim]
                )
            else:
                results[video_name].keyframe.append(keyframe)
                results[video_name].frame_idx.append(frame_idx)
                results[video_name].pts_time.append(pts_time)
                results[video_name].similarity.append(sim)
        return list(results.values())


class ClipRetrieval(Retrieval):

    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self.embs = self.dataset.clip_embs

        logging.info(" Loading CLIP model...")
        self.model, self.preprocess = clip.load("RN50", device=self.device)
        logging.info(" CLIP model loaded.")

    def search_text(self, text: str, top_k: int = TOP_K):
        tokenized_text = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_embedding = self.model.encode_text(tokenized_text)
        text_embedding = norm_vectors(text_embedding.cpu().numpy())
        self.search(text_embedding, self.embs, top_k=top_k)
    
    def search_image(self, image: torch.Tensor, top_k: int = TOP_K):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embedding = self.model.encode_image(image)
        image_embedding = norm_vectors(image_embedding.cpu().numpy())
        self.search(image_embedding, self.embs, top_k=top_k)