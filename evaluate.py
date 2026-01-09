import os
import time
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src import FasterRCNN, box_iou
from dataset import StreetDataset, collate_fn


def compute_ap(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    """Compute average precision for one class"""
    if len(gt_boxes) == 0:
        return 0.0
    if len(pred_boxes) == 0:
        return 0.0

    # sort by score
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    matched = np.zeros(len(gt_boxes), dtype=bool)

    for i, pb in enumerate(pred_boxes):
        if len(gt_boxes) == 0:
            fp[i] = 1
            continue

        ious = compute_iou_np(pb, gt_boxes)
        best_idx = np.argmax(ious)
        best_iou = ious[best_idx]

        if best_iou >= iou_thresh and not matched[best_idx]:
            tp[i] = 1
            matched[best_idx] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / len(gt_boxes)
    precision = tp_cum / (tp_cum + fp_cum)

    # AP
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])

    idx = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[idx+1] - recall[idx]) * precision[idx+1])

    return ap


def compute_iou_np(box, boxes):
    """Compute IoU between one box and array of boxes"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return inter / (area1 + area2 - inter + 1e-7)


def evaluate(checkpoint_path):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')

    # load model
    model = FasterRCNN(config.NUM_CLASSES, config.BACKBONE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # data
    val_data = StreetDataset(config.DATA_ROOT, 'validation', config.CLASSES)
    val_loader = DataLoader(val_data, batch_size=1, collate_fn=collate_fn)

    # collect predictions and ground truth per class
    class_preds = {c: {'boxes': [], 'scores': []} for c in range(1, config.NUM_CLASSES)}
    class_gt = {c: [] for c in range(1, config.NUM_CLASSES)}

    times = []

    print("Running evaluation...")
    with torch.no_grad():
        for imgs, targets in tqdm(val_loader):
            imgs = [img.to(device) for img in imgs]

            start = time.time()
            outputs = model(imgs)
            times.append(time.time() - start)

            # collect
            for out, tgt in zip(outputs, targets):
                pred_boxes = out['boxes'].cpu().numpy()
                pred_labels = out['labels'].cpu().numpy()
                pred_scores = out['scores'].cpu().numpy()

                gt_boxes = tgt['boxes'].numpy()
                gt_labels = tgt['labels'].numpy()

                for c in range(1, config.NUM_CLASSES):
                    # predictions for class c
                    mask = pred_labels == c
                    if mask.sum() > 0:
                        class_preds[c]['boxes'].append(pred_boxes[mask])
                        class_preds[c]['scores'].append(pred_scores[mask])

                    # gt for class c
                    mask = gt_labels == c
                    if mask.sum() > 0:
                        class_gt[c].append(gt_boxes[mask])

    # compute mAP
    aps = {}
    for c in range(1, config.NUM_CLASSES):
        if len(class_preds[c]['boxes']) == 0 or len(class_gt[c]) == 0:
            aps[config.CLASSES[c]] = 0.0
            continue

        all_boxes = np.vstack(class_preds[c]['boxes'])
        all_scores = np.concatenate(class_preds[c]['scores'])
        all_gt = np.vstack(class_gt[c])

        ap = compute_ap(all_boxes, all_scores, all_gt)
        aps[config.CLASSES[c]] = ap

    mAP = np.mean(list(aps.values()))

    # speed
    avg_time = np.mean(times) * 1000
    fps = 1000 / avg_time

    # print results
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"\nmAP@0.5: {mAP:.4f}")
    print("\nPer-class AP:")
    for cls, ap in aps.items():
        print(f"  {cls}: {ap:.4f}")
    print(f"\nSpeed: {avg_time:.1f}ms ({fps:.1f} FPS)")

    # model size
    size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024
    print(f"Model size: {size_mb:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='output/best.pth')
    args = parser.parse_args()
    evaluate(args.checkpoint)
