from models.nodes import VideoReader

import os
from pathlib import Path

import cv2


class Main:
    def __init__(self, config: dict):
        self.config = config
        self.video_reader = VideoReader(self.config)

    def process(self):
        for frame in self.video_reader.process():
            pass
            # print(frame.frame_number)
            # print(frame.timestamp)
