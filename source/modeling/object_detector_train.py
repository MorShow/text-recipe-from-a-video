from models.nodes import CLIPObjectDetector
from source.config import NUM_CLASSES, RAW_DATA, PROCESSED_DATA, SAVED_MODELS
from source.modeling.CLIPImageTextDataset import CLIPImageTextDataset

import os
import random
from pathlib import Path

import cv2
import torch
import clip
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image


def focal_bce(logits, targets, gamma=2):
    l = logits.reshape(-1)
    t = targets.reshape(-1)

    p = torch.sigmoid(l)
    p = torch.where(t >= 0.5, p, 1-p)

    logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
    loss = logp * ((1 - p) ** gamma)
    loss = NUM_CLASSES * loss.mean()

    return loss


class CLIPMultiLabelLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-6

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor, targets: torch.Tensor):
        image_emb = F.normalize(image_emb, dim=-1)  # [batch, dim]
        text_emb = F.normalize(text_emb, dim=-1)  # [num_classes, dim]

        logits = (image_emb @ text_emb.T) / self.temperature

        pos_count = targets.squeeze().sum(dim=0)
        neg_count = targets.squeeze().size(0) - pos_count
        pos_weight = (neg_count + self.eps) / (pos_count + self.eps)

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fn(logits, targets.float())

        return loss, logits


class ObjectDetectorT:
    def __init__(self,
                 epochs: int,
                 json_data: str | Path,
                 video_data: str | Path | list[str | Path],
                 batch_size: int,
                 shuffle: bool,
                 output_path: str | Path,
                 loss_fn: nn.Module = CLIPMultiLabelLoss()
                 ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.detector = CLIPObjectDetector()
        self.detector_preprocess = self.detector.preprocess
        self.detector_model = self.detector.model

        self.json_data = json_data
        self.video_data = video_data
        self.dataset = CLIPImageTextDataset(self.json_data, self.video_data)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)

        self.optimizer = torch.optim.Adadelta(self.detector_model.parameters(), lr=1e-3)
        self.loss_fn = loss_fn

        self.output_path = output_path

    def train(self):
        for epoch in range(self.epochs):
            c = 0

            print('=' * 30 + f' EPOCH: {epoch + 1} ' + '=' * 30)

            for image, text in self.dataloader:
                if text.sum() == 0 and random.random() > .33:
                    continue

                text = text.to(self.detector.device).float()
                text_processed = clip.tokenize(self.detector.class_names).to(self.detector.device)
                text_embedding = self.detector_model.encode_text(text_processed).float()

                images_processed = []
                for i in range(image.shape[0]):
                    frame_rgb = cv2.cvtColor(image[i].permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    images_processed.append(self.detector_preprocess(pil_image))
                image = torch.stack(images_processed).to(self.detector.device)
                image_embedding = self.detector_model.encode_image(image).float()

                loss, logits = self.loss_fn(image_embedding, text_embedding, text)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if c % 5 == 0:
                    # print(f'Logits: {logits}, Shape: {logits.shape}')
                    print(f'Probs: {torch.sigmoid(logits)}')
                    print(f'Text: {text}, Shape: {text.shape}')
                    print(f'Loss: {loss}')

                c += 1

        self.save_model()

    def save_model(self) -> None:
        with open(os.path.join(SAVED_MODELS, self.output_path), 'wb') as f:
            torch.save(self.detector_model.state_dict(), f)


if __name__ == '__main__':
    process = ObjectDetectorT(
        epochs=20,
        json_data=PROCESSED_DATA / 'youcookii_annotations_small_processed.jsonl',
        video_data=[PROCESSED_DATA / 'clip_training' / 'small_training.pkl',
                    PROCESSED_DATA / 'clip_training' / 'small_videos.json'],
        batch_size=1,
        shuffle=True,
        output_path='first_model.pth'
    )
    process.train()
