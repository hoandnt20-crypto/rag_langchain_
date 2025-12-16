import torch
import numpy as np
import clip
import pandas as pd

from PIL import Image
from typing import List, Literal
from dataset import Dataset
from utils import norm_vectors, translate_vi_to_en, truncate_text
from config import SearchResult, Sample, Keyframe



class JointEmbeddingRetrieval:

    def __init__(self,  model=None, preprocess=None, device: str = "cpu"):
        self.device = device
        self.model = model
        self.preprocess = preprocess
        self.search_results = None


    def search(self, query_emb: np.ndarray, embs: np.ndarray, top_k: int = 100):
        query_emb = norm_vectors(query_emb)
        sims = (embs @ query_emb.T).squeeze()
        idxs = np.argsort(-sims)[:top_k]
        self.search_results = {
            "indexes": idxs.tolist(), 
            "similarity": [s.item() for s in sims[idxs]]
        }


    def collect_results(self, dataset: Dataset) -> List[SearchResult]:

        if self.search_results is None:
            raise ValueError("No search has been performed yet.")
        
        results = {}
        samples = dataset.get_items(self.search_results["indexes"])
        for sim, sample in zip(self.search_results["similarity"], samples):
            video_name = sample.media_info.name
            title      = sample.media_info.title
            keyframe   = sample.keyframe.replace("../data/", "").replace("\\", "/")      # eg: https://youtube.com/embed/p6h043fMCUA
            watch_url  = sample.media_info.watch_url.replace("watch?v=", "embed/")       # eg: keyframes/L22_V025/203.jpg
            frame_idx  = int(sample.map_keyframe.frame_idx.item())
            pts_time   = float(sample.map_keyframe.pts_time.item())

            kf = Keyframe(
                path       = keyframe,
                frame_idx  = frame_idx,
                pts_time   = pts_time,
                similarity = sim
            )

            if video_name not in results:
                results[video_name] = SearchResult(
                    video_name = video_name,
                    watch_url  = watch_url,
                    title      = title,
                    keyframes  = [kf],
                )
            else:
                results[video_name].keyframes.append(kf)

        return list(results.values())


class ClipRetrieval(JointEmbeddingRetrieval):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def search_text(self, text: str, datatset: Dataset, top_k: int = 100):
        tokenized_text = clip.tokenize([translate_vi_to_en(text, max_length=60)]).to(self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=torch.float16):
            text_embedding = self.model.encode_text(tokenized_text)
        text_embedding = norm_vectors(text_embedding.detach().cpu().numpy())
        self.search(text_embedding, datatset.clip_embs, top_k)
    

    def search_image(self, image: torch.Tensor, dataset: Dataset, top_k: int = 100):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=torch.float16):
            image_embedding = self.model.encode_image(image)
        image_embedding = norm_vectors(image_embedding.detach().cpu().numpy())
        self.search(image_embedding, dataset.clip_embs, top_k)



class TextRetrieval:


    def __init__(self, model, support_model=None, device: str = "cpu"):
        self.device = device
        self.model = model
        self.support_model = support_model
        self.search_results = None
        self.support_embedding = None


    def search(self, query_emb: np.ndarray, embs: np.ndarray, top_k: int = 100):
        query_emb = norm_vectors(query_emb)
        sims = (embs @ query_emb.T).squeeze()
        idxs = np.argsort(-sims)[:top_k]
        self.search_results = {
            "indexes": idxs.tolist(),
            "similarity": [s.item() for s in sims[idxs]]
        }


    def search_text(
        self,
        text: str,
        dataset: Dataset,
        info_search: Literal["transcription", "description"],
        top_k: int = 100
    ):
        tokenized_text = clip.tokenize([translate_vi_to_en(text, max_length=60)]).to(self.device)   # for support select keyframe
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=torch.float16):
            support_embedding = self.support_model.encode_text(tokenized_text)
            text_embedding = self.model.encode(
                [truncate_text(text, max_length=110)], 
                prompt_name="document", 
                show_progress_bar=False
            )
        
        text_embedding = norm_vectors(text_embedding)   # TODO: check device
        self.support_embedding = norm_vectors(support_embedding.detach().cpu().numpy())
        
        self.search(
            query_emb=text_embedding, 
            embs=dataset.transcription_embs if info_search == "transcription" else dataset.description_embs, 
            top_k=top_k
        )


    def _select_sample(self, dataset: Dataset, info: pd.DataFrame, idx: int) -> Sample:
        '''
        select best keyframe from samples
        '''
        selected_info = info.loc[idx]
        selected_embeddings = dataset.clip_embs[selected_info.agg_index]
        sims = (selected_embeddings @ self.support_embedding.T).squeeze()
        return dataset[selected_info.agg_index[np.argmax(sims).item()]]


    def collect_results(
        self,
        dataset: Dataset,
        info_search: Literal["transcription", "description"],
    ) -> List[SearchResult]:

        if self.search_results is None:
            raise ValueError("No search has been performed yet.")


        results = {}
        seen = []
        info = dataset.transcription_info if info_search == "transcription" else dataset.description_info
        for idx, sim in zip(self.search_results["indexes"], self.search_results["similarity"]):
            sample  = self._select_sample(dataset, info, idx)

            video_name = sample.media_info.name
            title      = sample.media_info.title
            keyframe   = sample.keyframe.replace("../data/", "").replace("\\", "/")      # eg: keyframes/L22_V025/203.jpg
            watch_url  = sample.media_info.watch_url.replace("watch?v=", "embed/")       # eg: https://youtube.com/embed/p6h043fMCUA
            frame_idx  = int(sample.map_keyframe.frame_idx.item())
            pts_time   = float(sample.map_keyframe.pts_time.item())

            kf = Keyframe(
                path       = keyframe,
                frame_idx  = frame_idx,
                pts_time   = pts_time,
                similarity = sim
            )

            if video_name not in results:
                results[video_name] = SearchResult(
                    video_name = video_name,
                    watch_url  = watch_url,
                    title      = title,
                    keyframes  = [kf],
                )
            else:
                results[video_name].keyframes.append(kf)

        return list(results.values())