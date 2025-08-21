import os
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

project_root = Path(__file__).resolve().parents[1]

# Video: data/raw/youcookII/training/101/0O4bxhpFX9o.mkv
video = cv2.VideoCapture(os.path.join(project_root, 'data/raw/youcookII/training/101/0O4bxhpFX9o.mkv'))


def basic_stats(input_video: cv2.VideoCapture, name: str = 'Undefined') -> Tuple:
    frame_count = input_video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = input_video.get(cv2.CAP_PROP_FPS)
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Name: {name}\nFrame count: {frame_count}\nFPS: {fps}\nWidth: {width}\nHeight: {height}')
    return frame_count, fps, width, height


def retrieve_frames(input_video: cv2.VideoCapture, name: str = 'Undefined') -> List:
    frames = []
    frame_count, sample_rate, _, _ = basic_stats(input_video, name)

    for i in range(0, int(frame_count), int(sample_rate)):
        input_video.set(cv2.CAP_PROP_POS_FRAMES, i)
        status, frame = input_video.read()
        if status:
            frames.append(frame)
    input_video.release()
    return frames


def video_intensity_distribution(input_video: cv2.VideoCapture, name: str = 'Undefined') -> Tuple:
    intensity_values = []
    video_frames = retrieve_frames(input_video, name)

    for frame in video_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_mean = round(float(np.mean(gray)), 1)
        intensity_values.append(gray_mean)

    mean_intensity = round(np.mean(intensity_values), 1)
    std_intensity = round(np.std(intensity_values), 1)
    print('Mean: {}'.format(mean_intensity))
    print('Std: {}'.format(std_intensity))
    return intensity_values, mean_intensity, std_intensity


values, mean, std = video_intensity_distribution(video, '0O4bxhpFX9o.mkv')
print(f'Values: {values}')

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.imshow(img, interpolation='bicubic', cmap='gray')
# plt.show()
