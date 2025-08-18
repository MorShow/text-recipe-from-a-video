from typing import Optional

import numpy as np


class FrameElement:
    def __init__(self,
                 path: Optional[str] = None,
                 frame: Optional[np.ndarray] = None,
                 timestamp: Optional[float] = None,
                 frame_number: Optional[int] = None
                 ):
        self.path = path
        self.frame = frame
        self.timestamp = timestamp
        self.frame_number = frame_number
