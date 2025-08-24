import cv2

from models.elements import FrameElement

import yaml
import numpy as np
import torch
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, config) -> None:
        config_yolo = config["detection_node"]
        self.model = YOLO(config_yolo["weight_pth"], task='detect')
        self.classes = self.model.names
        self.conf = config_yolo["confidence"]
        self.iou = config_yolo["iou"]
        self.imgsz = config_yolo["imgsz"]

    def process(self, source):
        result = self.model.track(
            source=source,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            persist=True
        )

        return result
