CLIP_EMBS     = r"..\data\clip_embs".replace("\\", "/")
KEYFRAMES     = r"..\data\keyframes".replace("\\", "/")
MAP_KEYFRAMES = r"..\data\map-keyframes".replace("\\", "/")
MEDIA_INFO    = r"..\data\media-info".replace("\\", "/")


import dataclasses
from typing import List
from pydantic import BaseModel


@dataclasses.dataclass
class Sample:
    keyframe    : str       # image path
    media_info  : dict      # name, title, watch_url
    map_keyframe: dict      # pts_time, fps, frame_idx


class SearchResult(BaseModel):
    video_name : str
    watch_url  : str
    keyframe   : List[str]
    frame_idx  : List[int]
    pts_time   : List[float]
    similarity : List[float]