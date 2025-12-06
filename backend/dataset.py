import os
from typing import List
from config import KEYFRAMES, Sample
from utils import ( 
    load_clip_embs, 
    load_keyframes,     
    load_media_info,
    load_map_keyframes,
    get_video_name,
)


class Dataset:

    def __init__(self):
        # Get list of keyframe folders
        self.keyframe_list =  os.listdir(KEYFRAMES)
        self.keyframe_list.sort()
        
        # Load dataset components
        self.clip_embs     = load_clip_embs(self.keyframe_list)
        self.keyframes     = load_keyframes(self.keyframe_list)
        self.media_info    = load_media_info(self.keyframe_list)
        self.map_keyframes = load_map_keyframes(self.keyframe_list)

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