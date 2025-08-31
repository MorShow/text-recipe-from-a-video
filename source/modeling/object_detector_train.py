from models.nodes import CLIPObjectDetector, VideoReader
from source.config import RAW_DATA, PROCESSED_DATA, CONFIGS_DIR
from source.modeling.CLIPImageTextDataset import CLIPImageTextDataset

import os
import json

import cv2
import torch
import clip
from torch.utils.data import DataLoader
from PIL import Image


class ObjectDetectorT:
    def __init__(self):
        pass


if __name__ == '__main__':
    detector = CLIPObjectDetector()
    detector_preprocess = detector.preprocess
    detector_model = detector.model
    dataset = CLIPImageTextDataset(PROCESSED_DATA / 'youcookii_annotations_small_processed.jsonl',
                                   [PROCESSED_DATA / 'clip_training' / 'small_training.pkl',
                                    PROCESSED_DATA / 'clip_training' / 'small_videos.json'])

    optimizer = torch.optim.AdamW(detector_model.parameters(), lr=1e-15)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    with open(PROCESSED_DATA / 'clip_training' / 'small_videos.json', 'r') as file:
        dictionary = json.load(file)

    for image, text in dataloader:
        text = text.to(detector.device)
        image = image[0].permute(1, 2, 0).cpu().numpy().astype("uint8")
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        image = detector_preprocess(pil_image).unsqueeze(0).to(detector.device)
        text_processed = clip.tokenize(detector.class_names).to(detector.device)

        image_embedding = detector_model.encode_image(image)
        text_embedding = detector_model.encode_text(text_processed)

        image_embedding_norm = image_embedding / image_embedding.norm(dim=-1, keepdim=True).float()
        text_embedding_norm = text_embedding / text_embedding.norm(dim=-1, keepdim=True).float()

        logits = image_embedding_norm @ text_embedding_norm.t()
        logits = logits.squeeze(0).to(torch.float32)
        text = text.squeeze(0).to(torch.float32)
        print(logits)
        print(text)
        loss = loss_fn(logits, text)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f'Loss: {loss}')
