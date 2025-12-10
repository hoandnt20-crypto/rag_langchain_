import os
import torch
import numpy as np
import clip

from PIL import Image
from typing import List, Literal, Optional
from dataset import Dataset
from utils import norm_vectors
from config import SearchResult, Sample



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

            if video_name not in results:
                results[video_name] = SearchResult(
                    video_name = video_name,
                    watch_url  = watch_url,
                    title      = title,
                    keyframe   = [keyframe],
                    frame_idx  = [frame_idx],
                    pts_time   = [pts_time],
                    similarity = [sim]
                )
            else:
                results[video_name].keyframe.append(keyframe)
                results[video_name].frame_idx.append(frame_idx)
                results[video_name].pts_time.append(pts_time)
                results[video_name].similarity.append(sim)

        return list(results.values())


class ClipRetrieval(JointEmbeddingRetrieval):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def search_text(self, text: str, datatset: Dataset, top_k: int = 100):
        tokenized_text = clip.tokenize([text]).to(self.device)
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


    def __init__(self, model, support_model=None, support_preprocess=None, device: str = "cpu"):
        self.device = device
        self.model = model
        self.support_model = support_model
        self.support_preprocess = support_preprocess
        self.search_results = None
        self.text_query = None


    def search(self, query_emb: np.ndarray, embs: np.ndarray, top_k: int = 100):
        query_emb = norm_vectors(query_emb)
        sims = (embs @ query_emb.T).squeeze()
        idxs = np.argsort(-sims)[:top_k]
        self.search_result = {
            "indexes": idxs.tolist(),
            "similarity": [s.item() for s in sims[idxs]]
        }

    def search_text(self, text: str, dataset: Dataset, top_k: int = 100):
        self.text_query = [text]
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=torch.float16):
            text_embedding = self.model.encode([text])
        text_embedding = norm_vectors(text_embedding.detach().cpu().numpy())
        self.search(text_embedding, dataset.transcript_embs, top_k)

    def _select_sample(self, samples: List[Sample]) -> Sample:
        '''
        select best keyframe from samples
        '''
        images = [Image.open("../data/" + sample.keyframe).convert("RGB") for sample in samples]
        inputs = torch.stack([self.support_preprocess(image).to(self.device) for image in images], axis=0)
        tokenized_text = clip.tokenize(self.text_query).to(self.device) # Todo: dynamic tokenize

        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=torch.float16):
            image_embs = self.support_model.encode_image(inputs)
            text_embs = self.support_model.encode_text(tokenized_text)

        image_embs = norm_vectors(image_embs.detach().cpu().numpy())
        text_embs = norm_vectors(text_embs.detach().cpu().numpy())
        sims = (image_embs @ text_embs.T).squeeze()
        idxs = np.argsort(-sims)
        return samples[idxs[0]]

    def collect_results(
        self,
        dataset: Dataset,
        info_search: Optional[Literal["transcript", "description"]]= "transcript",
        top_k: int = 100
    ) -> List[SearchResult]:

        if self.search_results is None:
            raise ValueError("No search has been performed yet.")


        results = {}
        info = dataset.transcript_info if info_search == "transcript" else dataset.media_info
        for idx, sim in zip(self.search_results["indexes"], self.search_results["similarity"]):
            selected_info = info.loc[idx]          # content, start, end, video_name, agg_index
            samples = dataset.get_items(transcript_info.agg_index)
            sample  = self._select_sample(samples)

            video_name = sample.media_info.name
            title      = sample.media_info.title
            keyframe   = sample.keyframe.replace("../data/", "").replace("\\", "/")      # eg: keyframes/L22_V025/203.jpg
            watch_url  = sample.media_info.watch_url.replace("watch?v=", "embed/")       # eg: https://youtube.com/embed/p6h043fMCUA
            frame_idx  = int(sample.map_keyframe.frame_idx.item())
            pts_time   = float(sample.map_keyframe.pts_time.item())

            if video_name not in results:
                results[video_name] = SearchResult(
                    video_name = video_name,
                    watch_url  = watch_url,
                    title      = title,
                    keyframe   = [keyframe],
                    frame_idx  = [frame_idx],
                    pts_time   = [pts_time],
                    similarity = [sim]
                )
            else:
                results[video_name].keyframe.append(keyframe)
                results[video_name].frame_idx.append(frame_idx)
                results[video_name].pts_time.append(pts_time)
                results[video_name].similarity.append(sim)

        return list(results.values())



class TranscriptRetrieval(TextRetrieval):


    def __init__(self):
        super.__init__()