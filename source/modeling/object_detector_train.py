from models.nodes import ObjectDetector, VideoReader

import os
import json
from source.config import RAW_DATA, PROCESSED_DATA, CONFIGS_DIR

if __name__ == '__main__':
    annotations_dict = json.load(open(PROCESSED_DATA / 'youcookii_annotations_small_processed.jsonl', 'r'))
    list_segments = []

    for annotation in annotations_dict:
        if annotation['video_id'] == 'GLd3aX16zBg':
            list_segments.append(annotation)

    config = {}
    config['src'] = RAW_DATA / "youcookII" / "training" / "113" / "GLd3aX16zBg.mkv"
    config['frames_ratio'] = 24
    config['skip_secs'] = 0
    config['confidence'] = 0.5
    config['iou'] = 0.5
    config['imgsz'] = (640, 480)
    config['detection_node'] = os.path.join(CONFIGS_DIR, "yolo-object-detector-config.yaml")

    vr = VideoReader(config)
    od = ObjectDetector(config)
    print(f'Frames ratio: {vr.frames_ratio}')

    for frame in vr.process():
        frame = od.process(frame)
        print(f'Frame number: {frame.frame_number}')
        second = frame.frame_number / vr.frames_ratio
        print(f'Second: {second}')
        for segment in list_segments:
            time_segment = segment['segment_time']
            if second > time_segment[0] and second < time_segment[1]:
                print(f'Detected objects: {frame.box_names}')
                print(f'Ground truth: ', end='')
                for annotation in segment['annotations']:
                    print(f'{annotation["noun"]}, ', end='')
                    print(f'{annotation["target"]}, ', end='')
                print()
