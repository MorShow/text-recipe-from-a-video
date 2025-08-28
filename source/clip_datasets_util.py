from numpy import ndarray, dtype, float64

from source.config import CLASS_NAMES, PROCESSED_DATA, RAW_DATA

import json
from pathlib import Path
from typing import Any

import numpy as np
import cv2


class CLIPDatasetsUtil:
    vector_dim = len(CLASS_NAMES)

    def __init__(self, input_json: str | Path):
        with open(input_json, 'r') as file:
            self._json_dir = json.loads(file.read())

    def create_matrix(self, video_id: str) -> tuple[str, ndarray]:
        matrix = np.zeros((seconds, self.vector_dim))
        list_to_check = [d for d in self._json_dir if d['video_id'] == video_id]

        for d in list_to_check:
            start, end = d['segment_time']
            vector = np.zeros(self.vector_dim)
            print(d)

            for annotation in d['annotations']:
                noun = annotation['noun']
                target = annotation['target']
                print(noun, target)

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

        return video_id, matrix


if __name__ == '__main__':
    util = CLIPDatasetsUtil(PROCESSED_DATA / 'youcookii_annotations_small_processed.jsonl')
    matrix = util.create_matrix(RAW_DATA / 'youcookII' / 'training' / '113' / 'GLd3aX16zBg.mkv')
    print(matrix[90:120])
