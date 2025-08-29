from typing import Optional

import numpy as np


class FrameElement:
    def __init__(self,
                 path: Optional[str] = None,
                 frame: Optional[np.ndarray] = None,
                 timestamp: Optional[float] = None,
                 frame_number: Optional[int] = None,
                 seconds: Optional[int] = None,
                 ) -> None:
        self.video_path = path
        video_id = self.video_path.split('\\')[-1]
        self.video_id = video_id[:video_id.find('.mkv')]
        self.video_seconds = seconds
        self.frame = frame
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.box_xy = None
        self.box_confidence_scores = None
        self.box_class_ids = None
        self.box_tracker_id = None
        self.box_names = None
        self.clip_output = None
        self.clip_image_embedding = None

    def __repr__(self,
                 show_frame: bool = False,
                 show_embedding: bool = False) -> str:
        representation = f'''
            Video ID: {self.video_id},
            Source video path: {self.video_path},
            The length of the video (in seconds): {self.video_seconds},
            Timestamp: {self.timestamp},
            Frame Number: {self.frame_number},
            Box XY: {self.box_xy},
            Box confidence scores: {self.box_confidence_scores},
            Box class IDs: {self.box_class_ids},
            Box tracker ID: {self.box_tracker_id},
            Box names: {self.box_names},
            CLIP detected classes: {self.clip_output}
        '''

        if show_frame:
            representation = representation + f",\n Frame: {self.frame}"
        if show_embedding:
            representation = representation + f",\n Box embedding: {self.clip_output}"

        return representation
