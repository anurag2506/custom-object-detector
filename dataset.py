import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random


class StreetDataset(Dataset):
    """Dataset for street object detection"""

    def __init__(self, root, split, classes, augment=False):
        self.root = root
        self.split = split
        self.classes = classes
        self.augment = augment
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.img_dir = os.path.join(root, 'images', split)
        self.anno_file = os.path.join(root, 'annotations', f'{split}_annotations.csv')

        self.annotations = {}
        self._load_annotations()

        self.img_ids = [k for k in self.annotations.keys()
                        if os.path.exists(os.path.join(self.img_dir, f"{k}.jpg"))]

        print(f"Loaded {split}: {len(self.img_ids)} images")

    def _load_annotations(self):
        with open(self.anno_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row['ImageID']
                cls = row['ClassName']
                if cls not in self.class_to_idx:
                    continue
                bbox = [float(row['XMin']), float(row['YMin']),
                        float(row['XMax']), float(row['YMax'])]

                if img_id not in self.annotations:
                    self.annotations[img_id] = []
                self.annotations[img_id].append({'class': cls, 'bbox': bbox})

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        boxes = []
        labels = []
        for ann in self.annotations[img_id]:
            x1 = ann['bbox'][0] * w
            y1 = ann['bbox'][1] * h
            x2 = ann['bbox'][2] * w
            y2 = ann['bbox'][3] * h
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_to_idx[ann['class']])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # augmentation
        if self.augment:
            img, boxes = self._augment(img, boxes)

        # to tensor and normalize
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return img, {'boxes': boxes, 'labels': labels, 'image_id': img_id}

    def _augment(self, img, boxes):
        # horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            w = img.size[0]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        # color jitter
        img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
        img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
        img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))

        return img, boxes


def collate_fn(batch):
    """Custom collate for variable sized targets"""
    imgs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return imgs, targets
