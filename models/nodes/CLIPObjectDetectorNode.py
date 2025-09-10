from typing import List

from source.config import RAW_DATA, CONFIGS_DIR, CLASS_NAMES
from models.nodes import VideoReader
from models.elements import FrameElement

import os
import cv2
import clip
import torch
from PIL import Image


class CLIPObjectDetector:
    _class_names = CLASS_NAMES

    def __init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, self._preprocess = clip.load("ViT-B/32", device=self._device)

    @property
    def preprocess(self):
        return self._preprocess

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @model.setter
    def model(self, value: torch.nn.Module) -> None:
        self._model = value

    @property
    def device(self) -> str:
        return self._device

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def process(self, source: FrameElement) -> FrameElement:
        frame_rgb = cv2.cvtColor(source.frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        image = self._preprocess(pil_image).unsqueeze(0).to(self._device)
        text = clip.tokenize(self._class_names).to(self._device)

        with torch.no_grad():
            image_features = self._model.encode_image(image)
            text_features = self._model.encode_text(text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        probs = similarities[0].cpu().numpy()
        output_dict = dict()
        for cls, p in zip(self._class_names, probs):
            output_dict[cls] = round(float(p), 4)
        source.clip_output = output_dict
        source.clip_image_embedding = image_features

        return source


if __name__ == "__main__":
    config = {}
    config['src'] = RAW_DATA / "youcookII" / "training" / "101" / "0O4bxhpFX9o.mkv"
    config['frames_ratio'] = 24
    config['skip_secs'] = 0
    config['confidence'] = 0.5
    config['iou'] = 0.5
    config['imgsz'] = (640, 480)
    config['detection_node'] = os.path.join(CONFIGS_DIR, "yolo-object-detector-config.yaml")

    reader = VideoReader(config)
    detector = CLIPObjectDetector()

    for frame in reader.process():
        obj = detector.process(frame)
        print(obj)
