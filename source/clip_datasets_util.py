from source.config import CLASS_NAMES, PROCESSED_DATA, RAW_DATA, CONFIGS_DIR
from models.elements import VideoElement
from models.nodes import VideoReader, ObjectDetector, load_videos

import os
import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import cv2
from numpy import ndarray, dtype, float64


class CLIPDatasetsUtil:
    vector_dim = len(CLASS_NAMES)

    def __init__(self, input_json: str | Path):
        with open(input_json, 'r') as file:
            self._json_dir = json.loads(file.read())

    def create_matrix(self, source: VideoElement) -> ndarray:
        seconds = source.video_seconds
        matrix = np.zeros((seconds, self.vector_dim))
        list_to_check = [d for d in self._json_dir if d['video_id'] == source.video_id]

        for d in list_to_check:
            start, end = d['segment_time']
            vector = np.zeros(self.vector_dim)

            for annotation in d['annotations']:
                noun = annotation['noun']
                target = annotation['target']

                try:
                    noun_index = CLASS_NAMES.index(noun)
                except ValueError:
                    noun_index = -1

                try:
                    target_index = CLASS_NAMES.index(target)
                except ValueError:
                    target_index = -1

                vector[noun_index] = 1 if vector[noun_index] == 0 and noun_index != -1 else 1
                vector[target_index] = 1 if vector[target_index] == 0 and target_index != -1 else 1

            matrix[start:end] = vector

        return matrix

    def create_dataset(self, input_dir: str | Path) -> tuple[pd.DataFrame, dict]:
        """Input format:
            <input_directory>
            ---- <batch_1>
            -------- <video_1>
            -------- <video_2>
            -------- ...
            ---- <batch_2>
            -------- <video_1>
            -------- ...
            ---- ...
        """

        result_df = pd.DataFrame(columns=['matrix'])
        result_dict = load_videos(input_dir)

        for index, video in result_dict.items():
            matrix = self.create_matrix(video)
            result_df.loc[index] = {'matrix': matrix}

        return result_df, result_dict


if __name__ == '__main__':
    util = CLIPDatasetsUtil(PROCESSED_DATA / 'youcookii_annotations_small_processed.jsonl')
    df = util.create_dataset(RAW_DATA / 'youcookII' / 'training')
    df.to_csv(PROCESSED_DATA / 'clip_training' / 'small_training.csv')
