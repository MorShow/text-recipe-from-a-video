from pathlib import Path
from typing import Optional

import cv2


class VideoElement:
    def __init__(self, video_url: str | Path) -> None:
        self.video_path = video_url
        video_id = str(self.video_path).split('\\')[-1]
        self.video_id = video_id[:video_id.find('.mkv')]
        self.frame_count = None
        self.frame_ratio = None
        self.video_seconds = None
        self.matrix = None
        self.frames_dict = {}
        self.probabilities_dict = {}

    def initiate(self, frame_ratio: Optional[int] = None) -> None:
        cap = cv2.VideoCapture(self.video_path)
        self.frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_ratio = frame_ratio if frame_ratio is not None else cap.get(cv2.CAP_PROP_FPS)
        self.video_seconds = int(self.frame_count / self.frame_ratio)
        cap.release()

    def return_dict(self) -> dict:
        return {
            'video_path': str(self.video_path),
            'video_id': self.video_id,
            'frame_count': self.frame_count,
            'frame_ratio': self.frame_ratio,
            'video_seconds': self.video_seconds,
            'matrix': self.matrix
        }
