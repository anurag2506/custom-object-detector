"""
Prepare dataset from archive folder
Creates detection format with pseudo bounding boxes
"""

import os
import shutil
import csv
import random
from PIL import Image

# source and destination
ARCHIVE_DIR = "../archive"
OUTPUT_DIR = "./data/street_objects"

# classes we want (must match folder names in archive)
CLASS_MAP = {
    'person': 'Person',
    'car': 'Car',
    'truck': 'Truck',
    'bicycle': 'Bicycle',
    'trafficlight': 'Traffic light'
}

IMAGES_PER_CLASS = 400
TRAIN_RATIO = 0.8


def create_pseudo_bbox():
    """Generate pseudo bbox (centered, 70-90% coverage)"""
    size = random.uniform(0.7, 0.9)
    margin = (1 - size) / 2
    return [margin, margin, 1 - margin, 1 - margin]


def main():
    print("Preparing dataset...")

    # create directories
    for split in ['train', 'validation']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'annotations'), exist_ok=True)

    train_annos = []
    val_annos = []

    for folder_name, class_name in CLASS_MAP.items():
        src_dir = os.path.join(ARCHIVE_DIR, folder_name)
        if not os.path.exists(src_dir):
            print(f"  Warning: {src_dir} not found, skipping")
            continue

        # get images
        images = [f for f in os.listdir(src_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        images = images[:IMAGES_PER_CLASS]

        n_train = int(len(images) * TRAIN_RATIO)

        for i, img_name in enumerate(images):
            split = 'train' if i < n_train else 'validation'
            img_id = f"{class_name.lower().replace(' ', '_')}_{i:04d}"

            # copy image
            src_path = os.path.join(src_dir, img_name)
            dst_path = os.path.join(OUTPUT_DIR, 'images', split, f"{img_id}.jpg")

            img = Image.open(src_path).convert('RGB')
            img.save(dst_path, 'JPEG')

            # create annotation
            bbox = create_pseudo_bbox()
            anno = {
                'ImageID': img_id,
                'ClassName': class_name,
                'XMin': bbox[0],
                'YMin': bbox[1],
                'XMax': bbox[2],
                'YMax': bbox[3]
            }

            if split == 'train':
                train_annos.append(anno)
            else:
                val_annos.append(anno)

        print(f"  {class_name}: {len(images)} images")

    # save annotations
    for split, annos in [('train', train_annos), ('validation', val_annos)]:
        path = os.path.join(OUTPUT_DIR, 'annotations', f'{split}_annotations.csv')
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['ImageID', 'ClassName', 'XMin', 'YMin', 'XMax', 'YMax'])
            writer.writeheader()
            writer.writerows(annos)
        print(f"Saved {len(annos)} {split} annotations")

    print("\nDone! Dataset ready at:", OUTPUT_DIR)


if __name__ == '__main__':
    random.seed(42)
    main()
