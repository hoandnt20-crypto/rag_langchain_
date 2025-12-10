import os
import tqdm
import json
import numpy as np
import pandas as pd

from typing import List
from config import (
    CLIP_EMBS,
    TRANSCRIPTION_EMBS,
    DESCRIPTION_EMBS,
    MEDIA_INFO,
    TRANSCRIPTION_INFO,
    DESCRIPTION_INFO,
    KEYFRAMES,      
    MAP_KEYFRAMES,
    N_cols
)


def get_video_name(img_path: str) -> str:
    i = img_path.find("_V")
    return img_path[i-3:i+5]


def extract_timestamp(timestamp: str):
    def convert_to_seconds(time_str: str) -> float:
        minutes, seconds = map(int, time_str.split(":"))
        return minutes * 60 + seconds
    start, end = timestamp.split(" - ")
    return convert_to_seconds(start), convert_to_seconds(end)


def norm_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim == 1:
        norms = np.linalg.norm(vectors)
        return vectors / norms
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / norms


def mapping_temporal_keyframe(info: pd.DataFrame, map_keyframe: pd.DataFrame, expand_temporal: int = 0):
    '''
    Mapping temporal keyframe to transcript
    '''
    df = info.merge(map_keyframe, how="left", on="video_name")
    df_matched = df[(df["start"]<= df["pts_time"]+expand_temporal) & (df["pts_time"]-expand_temporal <= df["end"])]
    grouped = df_matched.groupby(["video_name", "start", "end"]).agg({"index": list}).reset_index()
    info = info.merge(grouped, how="left", on=["video_name", "start", "end"])
    info["agg_index"] = info["index"].apply(lambda x: x if isinstance(x, list) else [])
    info.drop(columns=["index"], inplace=True)
    return info


def load_clip_embs(keyframe_list: List[str]) -> np.ndarray:
    '''
    Load and normalize clip embeddings for the given keyframe folder.
    '''
    embeddings = []
    for keyframe in tqdm.tqdm(keyframe_list, desc="Loading clip embeddings", ncols=N_cols):
        embs = np.load(os.path.join(CLIP_EMBS, keyframe + ".npy"))
        embeddings.extend(norm_vectors(embs))
    embeddings = np.array(embeddings)
    return embeddings


def load_transcription_embs(keyframe_list: List[str]) -> np.ndarray:
    '''
    Load and normalize transcription embeddings for the given keyframe folder.
    '''
    embeddings = []
    for keyframe in tqdm.tqdm(keyframe_list, desc="Loading transcription embeddings", ncols=N_cols):
        embs = np.load(os.path.join(TRANSCRIPTION_EMBS, keyframe + ".npy"))
        embeddings.extend(norm_vectors(embs))
    embeddings = np.array(embeddings)
    return embeddings


def load_description_embs(keyframe_list: List[str]) -> np.ndarray:
    '''
    Load and normalize description embeddings for the given keyframe folder.
    '''
    embeddings = []
    for keyframe in tqdm.tqdm(keyframe_list, desc="Loading description embeddings", ncols=N_cols):
        embs = np.load(os.path.join(DESCRIPTION_EMBS, keyframe + ".npy"))
        embeddings.extend(norm_vectors(embs))
    embeddings = np.array(embeddings)
    return embeddings


def load_media_info(keyframe_list: List[str]) -> pd.DataFrame:
    '''
    Load media information from JSON files for the given keyframe folder.
    '''
    media_infos = []
    for keyframe in tqdm.tqdm(keyframe_list, desc="Loading media info", ncols=N_cols):
        json_path = os.path.join(MEDIA_INFO, keyframe + ".json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        select_keys = ["title", "watch_url"]
        select_data = {key: data[key] for key in select_keys if key in data.keys()}
        media_infos.append(select_data)
    return pd.DataFrame(media_infos, index=keyframe_list)


def load_transcription_info(keyframe_list: List[str]):
    '''
    Load media information from JSON files for the given keyframe folder.
    '''
    transcript_infos = []
    for keyframe in tqdm.tqdm(keyframe_list, desc="Loading transcription info", ncols=N_cols):
        json_path = os.path.join(TRANSCRIPTION_INFO, keyframe + ".json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for dt in data:
            start, end = dt["timestamp"].split(":")
            transcript_infos.append({
                "video_name": keyframe,
                "content": dt["content"],
                "start": float(start),
                "end": float(end)
            })
    return pd.DataFrame(transcript_infos)


def load_description_info(keyframe_list: List[str]):
    '''
    Load description information from JSON files for the given keyframe folder.
    '''
    description_infos = []
    for keyframe in tqdm.tqdm(keyframe_list, desc="Loading description info", ncols=N_cols):
        json_path = os.path.join(DESCRIPTION_INFO, keyframe + ".json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for dt in data:
            start, end = extract_timestamp(dt["timestamp"])
            description_infos.append({
                "video_name": keyframe,
                "content": dt["content"],
                "start": float(start),
                "end": float(end)
            })
    return pd.DataFrame(description_infos)


def load_keyframes(keyframe_list: List[str]) -> List[str]:
    '''
    Load image paths for the given keyframe folder.
    '''
    image_paths = []
    for keyframe in tqdm.tqdm(keyframe_list, desc="Loading image paths", ncols=N_cols):
        kf_folder = os.path.join(KEYFRAMES, keyframe)
        img_files = os.listdir(kf_folder)
        img_files.sort()
        img_paths = [os.path.join(kf_folder, img_file) for img_file in img_files]
        image_paths.extend(img_paths)
    return image_paths


def load_map_keyframes(keyframe_list: List[str]) -> pd.DataFrame:
    '''
    Load mapping of keyframes from CSV files for the given keyframe folder.
    '''
    map_keyframes = []
    for keyframe in tqdm.tqdm(keyframe_list, desc="Loading map keyframes", ncols=N_cols):
        csv_path = os.path.join(MAP_KEYFRAMES, keyframe + ".csv")
        df = pd.read_csv(csv_path)
        df["video_name"] = [keyframe] * df.shape[0]
        map_keyframes.append(df)
    combine_df = pd.concat(map_keyframes, ignore_index=True)
    combine_df["index"] = combine_df.index
    return combine_df