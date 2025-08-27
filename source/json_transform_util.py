import json
from pathlib import Path
from source.config import RAW_DATA, PROCESSED_DATA

project_root = Path(__file__).resolve().parents[1]


def json_transform(input_path: str | Path, output_filename: str):
    json_input = json.load(open(input_path, 'r'))
    json_dict = json_input['database']
    result = []

    for recipe_id, video_dict in json_dict.items():
        for annotation in video_dict['annotations']:
            segment_id = annotation['id']
            segment_time = annotation['segment']
            sentence = annotation['sentence']

            final_dict = {
                "video_id": recipe_id,
                "segment_id": segment_id,
                "segment_time": segment_time,
                "sentence": sentence,
                "annotations": {"action": None, "noun": None, "target": None}
            }

            result.append(final_dict)

    json.dump(result, open(PROCESSED_DATA / output_filename, 'w'))


if __name__ == '__main__':
    json_file = RAW_DATA / 'youcookII' / 'annotations' / 'youcookii_annotations_trainval.json'
    json_transform(json_file, 'youcookii_annotations_extracted.jsonl')
