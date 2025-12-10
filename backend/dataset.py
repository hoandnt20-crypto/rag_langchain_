import os
import logging
from typing import List
from config import KEYFRAMES, Sample
from utils import (
    load_clip_embs,
    load_transcript_embs,
    load_media_info,
    load_transcript_info,
    load_keyframes,
    load_map_keyframes,
    get_video_name,
    mapping_temporal_keyframe,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dataset:

    def __init__(self):
        # Get list of keyframe folders
        self.keyframe_list =  os.listdir(KEYFRAMES)
        self.keyframe_list.sort()
        
        # Load dataset components
        self.clip_embs       = load_clip_embs(self.keyframe_list)
        self.transcript_embs = load_transcript_embs(self.keyframe_list)
        self.media_info      = load_media_info(self.keyframe_list)
        self.transcript_info = load_transcript_info(self.keyframe_list)
        self.keyframes       = load_keyframes(self.keyframe_list)
        self.map_keyframes   = load_map_keyframes(self.keyframe_list)
        
        # Mapping temporal keyframe 
        logger.info("Mapping temporal keyframe ...")
        mapping_temporal_keyframe(self.transcript_info, self.map_keyframes)
        logger.info("Dataset loaded successfully")

    def __len__(self):
        return len(self.keyframes)
    
    def __getitem__(self, idx) -> Sample:
        video_name = get_video_name(self.keyframes[idx])
        return Sample(
            keyframe=self.keyframes[idx],
            media_info=self.media_info.loc[video_name],
            map_keyframe=self.map_keyframes.loc[idx]
        )
    
    def get_items(self, idxs) -> List[Sample]:
        return [self[idx] for idx in idxs]