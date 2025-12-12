import os
import logging
from typing import List
from config import KEYFRAMES, Sample
from utils import (
    load_clip_embs,
    load_transcription_embs,
    load_description_embs,
    load_media_info,
    load_transcription_info,
    load_description_info,
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
        self.clip_embs          = load_clip_embs(self.keyframe_list)
        self.transcription_embs = load_transcription_embs(self.keyframe_list)
        self.description_embs   = load_description_embs(self.keyframe_list)

        self.media_info         = load_media_info(self.keyframe_list)
        self.transcription_info = load_transcription_info(self.keyframe_list)
        self.description_info   = load_description_info(self.keyframe_list)
        
        self.keyframes          = load_keyframes(self.keyframe_list)
        self.map_keyframes      = load_map_keyframes(self.keyframe_list)
        
        # Mapping temporal keyframe 
        logger.info("Mapping temporal keyframe ...")
        self.transcription_info = mapping_temporal_keyframe(self.transcription_info, self.map_keyframes)
        self.description_info   = mapping_temporal_keyframe(self.description_info, self.map_keyframes, expand_temporal=4)
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