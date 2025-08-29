from source.config import RAW_DATA, PROCESSED_DATA
from models.nodes import VideoReader
from source.clip_datasets_util import CLIPDatasetsUtil

from pathlib import Path

from torch.utils.data import Dataset


class CLIPImageTextDataset(Dataset):
    def __init__(self, json_data: str | Path, video_data: str | Path):
        self._util = CLIPDatasetsUtil(json_data)
        self._video_data = str(video_data)
        self.images = []
        self.captions = []

    def initiate(self):
        result_df, videos_dict = self._util.create_dataset(self._video_data)

        for video_id, matrix_row in result_df.iterrows():
            matrix = matrix_row.values[0]

            config = {
                'src': videos_dict[video_id].video_path,
                'frames_ratio': videos_dict[video_id].frame_ratio,
                'skip_secs': 0
            }

            index = 0
            reader = VideoReader(config)

            for frame in reader.process():
                self.images.append(frame.frame)
                self.captions.append(matrix[index])
                index += 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: str):
        # image = self.images[]
        pass


if __name__ == '__main__':
    clip_df = CLIPImageTextDataset(
        PROCESSED_DATA / 'youcookii_annotations_small_processed.jsonl',
        RAW_DATA / 'youcookII' / 'training'
    )

    clip_df.initiate()
