from models.nodes import VideoReader
from models.elements import VideoElement
from source.config import KINETICS_400_URL, RAW_DATA

import pandas as pd
import numpy as np
import torch
from pytorchvideo.models.hub import slowfast_r50
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor


class ActionRecognizer:
    def __init__(self):
        self.classes = pd.read_csv(KINETICS_400_URL, index_col=0)
        self.model = slowfast_r50(pretrained=True)
        self.model.eval()

        self.transform = Compose([
            ToPILImage(),
            Resize((224, 224)),
            ToTensor()
        ])

    def frames_to_tensor(self,
                         source: VideoElement,
                         second_start: float,
                         second_end: float):
        frames = [frame for sec, frame in source.frames_dict.items() if second_start <= sec < second_end]
        frames = torch.stack([self.transform(f) for f in frames])

        frames_tensor = (torch
                         .tensor(np.array(frames), dtype=torch.float32)
                         .permute(1, 0, 2, 3)
                         .unsqueeze(0)
                         / 255.0)

        return frames_tensor

    def process(self,
                slow_source: VideoElement,
                fast_source: VideoElement,
                second_start: float,
                second_end: float,
                top_k: int = 5):
        frames_tensor_slow = self.frames_to_tensor(slow_source, second_start, second_end)
        frames_tensor_fast = self.frames_to_tensor(fast_source, second_start, second_end)
        # print(frames_tensor_slow.shape)
        # print(frames_tensor_fast.shape)

        inputs = [frames_tensor_slow, frames_tensor_fast]

        with torch.no_grad():
            outputs = self.model(inputs)

        probs = torch.softmax(outputs, dim=1)
        top_probs = torch.topk(probs, k=top_k)

        slow_source.probabilities_dict[(second_start, second_end)] = probs
        fast_source.probabilities_dict[(second_start, second_end)] = probs

        for i in range(top_k):
            print(self.classes.iloc[top_probs.indices[0][i].item()]['name'],
                  round(top_probs.values[0][i].item(), 2))


if __name__ == '__main__':
    config_slow = {}
    config_slow['src'] = RAW_DATA / 'youcookII' / 'training' / '101' / '0O4bxhpFX9o.mkv'
    config_slow['frames_ratio'] = 24
    config_slow['skip_secs'] = 0
    config_slow['confidence'] = 0.5

    config_fast = {}
    config_fast['src'] = RAW_DATA / 'youcookII' / 'training' / '101' / '0O4bxhpFX9o.mkv'
    config_fast['frames_ratio'] = 6
    config_fast['skip_secs'] = 0
    config_fast['confidence'] = 0.5

    video_reader_slow = VideoReader(config_slow)
    video_reader_fast = VideoReader(config_fast)

    for frame in video_reader_slow.process():
        pass

    for frame in video_reader_fast.process():
        pass

    video_frame_slow = video_reader_slow.video_element
    video_frame_fast = video_reader_fast.video_element

    recognizer = ActionRecognizer()
    recognizer.process(video_frame_slow, video_frame_fast, second_start=1., second_end=9.)
