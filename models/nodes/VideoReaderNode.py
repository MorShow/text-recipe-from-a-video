import os
import json
import time
import logging
from typing import Generator
import cv2

from models.elements import FrameElement

logger = logging.getLogger(__name__)


class VideoReader:
    """Node for reading video frames from a video file."""

    def __init__(self, config: dict) -> None:
        self.video_pth = config["src"]
        assert (
                os.path.isfile(self.video_pth)
                or "://" in self.video_pth
        ), f"VideoReader| The file ({self.video_pth}) was not found"

        self.stream = cv2.VideoCapture(self.video_pth)

        self.skip_secs = config["skip_secs"]  # how many seconds you want to skip (the end of the video)
        self.frames_ratio = config["frames_ratio"]
        self.last_frame_timestamp = -1
        self.first_timestamp = 0

        # self.break_element_sent = False

        # Read JSON config (???)
        # with open(config["roads_info"], "r") as file:
        #     data_json = json.load(file)

    def process(self) -> Generator[FrameElement, None, None]:
        frame_number = 0

        while True:
            self.stream.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.stream.read()
            if not ret:
                logger.warning("Can't receive frame. Exiting ...")
                break

            if "://" in self.video_pth:
                if frame_number == 0:
                    self.first_timestamp = time.time()

                timestamp = time.time() - self.first_timestamp
            else:
                timestamp = self.stream.get(cv2.CAP_PROP_POS_MSEC) / 1000

                timestamp = (
                    timestamp
                    if timestamp > self.last_frame_timestamp
                    else self.last_frame_timestamp + 0.1
                )

            if abs(self.last_frame_timestamp - timestamp) < self.skip_secs:
                continue

            self.last_frame_timestamp = timestamp

            frame_number += self.frames_ratio

            yield FrameElement(self.video_pth, frame, timestamp, frame_number)
