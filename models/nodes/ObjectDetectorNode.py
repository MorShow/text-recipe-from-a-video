from models.elements import FrameElement

import os
from pathlib import Path
from typing import Optional

import yaml
from ultralytics import YOLO

project_root = Path(__file__).parent
WEIGHTS_DIR = project_root / 'models' / 'weights'


class ObjectDetector:
    def __init__(self, config) -> None:
        with open(config["detection_node"], "r") as file:
            config_yolo = yaml.load(file, Loader=yaml.FullLoader)["detection_node"]
        self.model = YOLO(os.path.join(WEIGHTS_DIR, config_yolo["weight_pth"]), task='detect')
        self.classes = self.model.names
        self.conf = config_yolo["confidence"]
        self.iou = config_yolo["iou"]
        self.imgsz = config_yolo["imgsz"]

    def process(self, source: FrameElement, save_path: Optional[str | Path] = None):
        results = self.model.track(
            source=source.frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            persist=True
        )

        source.box_xy = results[0].boxes.xyxy
        source.box_confidence_scores = results[0].boxes.conf
        source.box_class_ids = results[0].boxes.cls
        source.box_tracker_id = results[0].boxes.id
        source.box_names = results[0].names

        if isinstance(save_path, str) or isinstance(save_path, Path):
            results[0].save(save_path if isinstance(save_path, str) else save_path.as_posix())

        return results
