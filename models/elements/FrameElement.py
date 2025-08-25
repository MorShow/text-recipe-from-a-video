from typing import Optional

import numpy as np


class FrameElement:
    def __init__(self,
                 path: Optional[str] = None,
                 frame: Optional[np.ndarray] = None,
                 timestamp: Optional[float] = None,
                 frame_number: Optional[int] = None
                 ) -> None:
        self.path = path
        self.frame = frame
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.box_xy = None
        self.box_confidence_scores = None
        self.box_class_ids = None
        self.box_tracker_id = None
        self.box_names = None
