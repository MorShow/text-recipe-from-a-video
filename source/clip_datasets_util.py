from source.config import CLASS_NAMES, PROCESSED_DATA, RAW_DATA, CONFIGS_DIR
from models.elements import VideoElement
from models.nodes import VideoReader, ObjectDetector, load_videos

import os
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import cv2
from numpy import ndarray, dtype, float64


class CLIPDatasetsUtil:
    vector_dim = len(CLASS_NAMES)

    def __init__(self, input_json: str | Path):
        with open(input_json, 'r') as file:
            self._json_dir = json.loads(file.read())
        self._ids = set()
        for d in self._json_dir:
            self._ids.add(d['video_id'])

    def create_matrix(self, source: dict) -> ndarray:
        seconds = source['video_seconds']
        matrix = np.zeros((seconds, self.vector_dim))
        list_to_check = [d for d in self._json_dir if d['video_id'] == source['video_id']]

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

    def create_dataset(self,
                       input_dir: str | Path,
                       save_df: Optional[str] = None,
                       save_dict: Optional[str] = None) -> tuple[pd.DataFrame, dict]:
        """
        - Input directory structue:
            <input_directory>
            ---- <batch_1>
            -------- <video_1>
            -------- <video_2>
            -------- ...
            ---- <batch_2>
            -------- <video_1>
            -------- ...
            ---- ...
        - save_df format:
            <filename>.pkl
        - save_dict format:
            <filename>.json
        """

        result_df = pd.DataFrame(columns=['matrix'])
        result_dict = load_videos(input_dir)

        for index, video in result_dict.items():
            if index in self._ids:
                matrix = self.create_matrix(video)
                result_df.loc[index] = {'matrix': matrix}

        if save_df:
            result_df.to_pickle(PROCESSED_DATA / 'clip_training' / save_df)
        if save_dict:
            with open(PROCESSED_DATA / 'clip_training' / save_dict, 'w') as file:
                json.dump(result_dict, file, indent=4)
        return result_df, result_dict


if __name__ == '__main__':
    util = CLIPDatasetsUtil(PROCESSED_DATA / 'youcookii_annotations_small_processed.jsonl')
    df = util.create_dataset(RAW_DATA / 'youcookII' / 'training',
                             'small_training.pkl',
                             'small_videos.json')
