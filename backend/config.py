# Embs
CLIP_EMBS          = r"..\data\embs\clip".replace("\\", "/")
TRANSCRIPTION_EMBS = r"..\data\embs\transcription".replace("\\", "/")
DESCRIPTION_EMBS   = r"..\data\embs\description".replace("\\", "/")

# Info
MEDIA_INFO         = r"..\data\info\media".replace("\\", "/")
DESCRIPTION_INFO   = r"..\data\info\description".replace("\\", "/")
TRANSCRIPTION_INFO = r"..\data\info\transcription".replace("\\", "/")

KEYFRAMES          = r"..\data\keyframes".replace("\\", "/")
MAP_KEYFRAMES      = r"..\data\map-keyframes".replace("\\", "/")

N_cols = 100  # use for progress bar


import dataclasses
from typing import List
from pydantic import BaseModel


@dataclasses.dataclass
class Sample:
    keyframe     : str       # image path
    media_info   : dict      # name, title, watch_url
    map_keyframe : dict      # pts_time, fps, frame_idx

class Keyframe(BaseModel):  # 1 sample keyframe
    path       : str
    frame_idx  : int
    pts_time   : float
    similarity : float

class SearchResult(BaseModel):
    video_name : str
    title      : str 
    watch_url  : str
    keyframes  : List[Keyframe]