from models.nodes import VideoReader, ObjectDetector
from source.config import CONFIGS_DIR

import os
from pathlib import Path


class Main:
    def __init__(self, config: dict):
        self.config = config
        self.video_reader = VideoReader(self.config)
        self.object_reader = ObjectDetector(self.config)

    def process(self):
        for frame in self.video_reader.process():
            frame = self.object_reader.process(frame)


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent

    config0 = {
        'src': os.path.join(project_root, 'data', 'raw', 'youcookII', 'training', '101', '0O4bxhpFX9o.mkv'),
        'skip_secs': 0,
        'frames_ratio': 24,
        'confidence': 0.5,
        'iou': 0.5,
        'imgsz': (640, 480),
        'detection_node': os.path.join(CONFIGS_DIR, "yolo-object-detector-config.yaml")
    }

    app = Main(config0)
    app.process()
