import os
import tqdm
import json
import numpy as np
import pandas as pd

from typing import List
from config import (
    CLIP_EMBS,
    KEYFRAMES,      
    MAP_KEYFRAMES,
    MEDIA_INFO
)


def get_video_name(img_path: str) -> str:
    i = img_path.find("_V")
    return img_path[i-3:i+5]

def norm_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim == 1:
        norms = np.linalg.norm(vectors)
        return vectors / norms
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / norms

def load_clip_embs(keyframe_list: List[str]) -> np.ndarray:
    '''
    Load and normalize clip embeddings for the given keyframe folder.
    '''
    embeddings = []
    for keyframe in tqdm.tqdm(keyframe_list, desc="Loading clip embeddings", ncols=100):
        embs = np.load(os.path.join(CLIP_EMBS, keyframe + ".npy"))
        embeddings.extend(norm_vectors(embs))
    embeddings = np.array(embeddings)
    return embeddings

def load_keyframes(keyframe_list: List[str]) -> List[str]:
    '''
    Load image paths for the given keyframe folder.
    '''
    image_paths = []
    for keyframe in tqdm.tqdm(keyframe_list, desc="Loading image paths", ncols=100):
        kf_folder = os.path.join(KEYFRAMES, keyframe)
        img_files = os.listdir(kf_folder)
        img_files.sort()
        img_paths = [os.path.join(kf_folder, img_file) for img_file in img_files]
        image_paths.extend(img_paths)
    return image_paths

def load_media_info(keyframe_list: List[str]) -> pd.DataFrame:
    '''
    Load media information from JSON files for the given keyframe folder.
    '''
    media_infos = []
    for keyframe in tqdm.tqdm(keyframe_list, desc="Loading media info", ncols=100):
        json_path = os.path.join(MEDIA_INFO, keyframe + ".json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        select_keys = ["title", "watch_url"]
        select_data = {key: data[key] for key in select_keys if key in data.keys()}
        media_infos.append(select_data)
    return pd.DataFrame(media_infos, index=keyframe_list)

def load_map_keyframes(keyframe_list: List[str]) -> pd.DataFrame:
    '''
    Load mapping of keyframes from CSV files for the given keyframe folder.
    '''
    map_keyframes = []
    for keyframe in tqdm.tqdm(keyframe_list, desc="Loading map keyframes", ncols=100):
        csv_path = os.path.join(MAP_KEYFRAMES, keyframe + ".csv")
        df = pd.read_csv(csv_path)
        map_keyframes.append(df)
    return pd.concat(map_keyframes, ignore_index=True)