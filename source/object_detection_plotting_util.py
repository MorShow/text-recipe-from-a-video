from source.config import REPORTS_FIGURES_DIR

import cv2
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


def visualize_frame_with_probs(frame: torch.Tensor,
                               probs: torch.Tensor,
                               class_names: list,
                               top_k: int = 5,
                               show: bool = False,
                               save_filename: str = None,
                               text_scale: float = 1.0,
                               text_color: tuple = (0, 255, 0),
                               text_thickness: int = 2):
    frame = frame.permute(1, 2, 0) if frame.shape[0] == 3 else frame
    frame = frame.cpu().numpy()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    top_probs, top_idxs = torch.topk(probs, top_k)
    top_probs = top_probs.cpu().numpy()
    top_idxs = top_idxs.cpu().numpy()

    y0, dy = 30, 30
    x = 10

    for index, (idx, prob) in enumerate(zip(top_idxs, top_probs)):
        text = f"{class_names[idx]}: {prob:.2f}"
        y = y0 + index * dy
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, text_thickness)

    if show:
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.show()
    if save_filename is not None:
        cv2.imwrite(REPORTS_FIGURES_DIR / 'clip_detector_imgs' / save_filename, frame)


def visualize_frame_with_bars(frame: torch.Tensor,
                              probs: torch.Tensor,
                              class_names: list,
                              top_k: int = 5,
                              show: bool = False,
                              save_filename: str = None,
                              text_scale: float = 1.0,
                              text_color: tuple = (0, 255, 0),
                              text_thickness: int = 2,
                              bar_color: tuple = (0, 200, 0),
                              bar_thickness: int = -1):
    frame = frame.permute(1, 2, 0) if frame.shape[0] == 3 else frame
    frame = frame.cpu().numpy()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    top_probs, top_idxs = torch.topk(probs, top_k)
    top_probs = top_probs.cpu().numpy()
    top_idxs = top_idxs.cpu().numpy()

    h, w, _ = frame.shape
    bar_width = 300
    canvas = np.ones((h, w + bar_width, 3), dtype=np.uint8) * 255
    canvas[:, :w] = frame
    y0, dy = 40, 60

    for i, (idx, prob) in enumerate(zip(top_idxs, top_probs)):
        bar_length = int(prob * (bar_width - 100))
        y = y0 + i * dy

        cv2.rectangle(canvas, (w + 10, y), (w + 10 + bar_length, y + 30), bar_color, bar_thickness)

        text = f"{class_names[idx]}: {prob:.2f}"
        cv2.putText(canvas, text, (w + 10, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, text_thickness)

    if show:
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.show()
    if save_filename is not None:
        cv2.imwrite(REPORTS_FIGURES_DIR / 'clip_detector_imgs' / save_filename, canvas)
