from models.nodes import CLIPObjectDetector, VideoReader
from source.config import RAW_DATA, PROCESSED_DATA, CONFIGS_DIR, NUM_CLASSES
from source.modeling.CLIPImageTextDataset import CLIPImageTextDataset

import os
import json

import cv2
import torch
import clip
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
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
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor, targets: torch.Tensor):
        image_emb = F.normalize(image_emb, dim=-1)  # [batch, dim]
        text_emb = F.normalize(text_emb, dim=-1)  # [num_classes, dim]

        logits = (image_emb @ text_emb.T) / self.temperature
        loss = self.loss_fn(logits, targets.float())

        return loss, logits


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

    # optimizer = torch.optim.AdamW(detector_model.parameters(), lr=1e-5)
    optimizer = torch.optim.Adadelta(detector_model.parameters(), lr=1e-5)
    # loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = focal_bce()
    loss_fn = CLIPMultiLabelLoss()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    with open(PROCESSED_DATA / 'clip_training' / 'small_videos.json', 'r') as file:
        dictionary = json.load(file)

    for image, text in dataloader:
        text = text.to(detector.device)
        # image = image[0].permute(1, 2, 0).cpu().numpy().astype("uint8")
        # image = image.permute(0, 3, 1, 2).cpu().numpy().astype("uint8")
        # print(image.shape)
        images_processed = []
        for i in range(image.shape[0]):
            frame_rgb = cv2.cvtColor(image[i].permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            images_processed.append(detector_preprocess(pil_image))
        images_tensor = torch.stack(images_processed).to(detector.device)
        image = images_tensor
        # image = detector_preprocess(pil_image).unsqueeze(0).to(detector.device)
        # print(image.shape)
        text_processed = clip.tokenize(detector.class_names).to(detector.device)

        classifier = nn.Linear(512, NUM_CLASSES).to(detector.device)

        image_embedding = detector_model.encode_image(image).float()
        text_embedding = detector_model.encode_text(text_processed).float()
        # text_embedding = text.float()
        # print(text_processed.shape)
        # print(text_embedding.shape)
        # print(image_embedding.shape)

        image_embedding_norm = image_embedding / (image_embedding.norm(dim=-1, keepdim=True) + 1e-6)
        text_embedding_norm = text_embedding / (text_embedding.norm(dim=-1, keepdim=True) + 1e-6)

        # logits = image_embedding_norm @ text_embedding_norm.t()
        logits = classifier(image_embedding_norm).float()
        print(logits.shape)
        # logits = logits.squeeze(0).float()
        # text = text.squeeze(0).float()
        text = text.float()
        print(text.shape)
        print(f'Logits: {logits}')
        print(f'Text: {text}')
        # loss = sigmoid_focal_loss(
        #     inputs=logits,
        #     targets=text,
        #     alpha=-1,
        #     gamma=4,
        #     reduction='mean'
        # )
        loss, logits = loss_fn(image_embedding, text_embedding, text)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f'Loss: {loss}')
