from source.config import RAW_DATA, PROCESSED_DATA
from models.nodes import VideoReader
from source.clip_datasets_util import CLIPDatasetsUtil

import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


class CLIPImageTextDataset(Dataset):
    def __init__(self, json_data: str | Path, video_data: str | Path | list):
        self._util = CLIPDatasetsUtil(json_data)
        self._video_data = str(video_data) if isinstance(video_data, Path) else video_data
        self._length = 0
        self.images = {}
        self.captions = {}
        self.initiate()

    def initiate(self):
        if isinstance(self._video_data, list):
            # Format: [<path_to_matrices_dataframe>, <path_to_videos_dict>]
            result_df, videos_dict = self._video_data
            result_df = pd.read_pickle(result_df)
            with open(videos_dict, 'r') as f:
                videos_dict = json.load(f)
        else:
            result_df, videos_dict = self._util.create_dataset(self._video_data)

        for video_id, matrix_row in result_df.iterrows():
            self.images[video_id] = []
            self.captions[video_id] = []

            matrix = matrix_row.values[0]

            config = {
                'src': videos_dict[video_id]['video_path'],
                'frames_ratio': videos_dict[video_id]['frame_ratio'],
                'skip_secs': 0
            }

            index = 0
            reader = VideoReader(config)

            for frame in reader.process():
                if index >= matrix.shape[0]:
                    break

                self.images[video_id].append(frame.frame)
                self.captions[video_id].append(matrix[index])
                index += 1
                self._length += 1

    def __len__(self):
        return self._length

    def __getitem__(self, idx: str | int):
        """
        :param idx: format (if string) - "<video_id>_<frame_number>"
        :return:
        """
        video_idx, frame_number_idx = None, None

        if isinstance(idx, str):
            video_idx, frame_number_idx = idx.split('_')
            frame_number_idx = int(frame_number_idx)
        elif isinstance(idx, int):
            for item_index, item in self.captions.items():
                if idx < len(item):
                    video_idx = item_index
                    frame_number_idx = idx
                else:
                    idx -= len(item)

        image = self.images[video_idx][frame_number_idx]
        caption = self.captions[video_idx][frame_number_idx]

        return image, caption


if __name__ == '__main__':
    clip_df = CLIPImageTextDataset(
        PROCESSED_DATA / 'youcookii_annotations_small_processed.jsonl',
        [PROCESSED_DATA / 'clip_training' / 'small_training.pkl',
         PROCESSED_DATA / 'clip_training' / 'small_videos.json']
    )

    clip_df.initiate()
    print(len(clip_df))
    print(clip_df['GLd3aX16zBg_93'][0])
    print(clip_df[93][0])
