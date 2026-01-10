import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import config
from src import FasterRCNN
from src.utils import box_iou
from dataset import StreetDataset, collate_fn


def compute_ap(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    if len(pred_boxes) == 0:
        return 0.0 if len(gt_boxes) > 0 else 1.0
    if len(gt_boxes) == 0:
        return 0.0

    sorted_idx = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_idx]

    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)
    tp = torch.zeros(len(pred_boxes))
    fp = torch.zeros(len(pred_boxes))

    for i, pred_box in enumerate(pred_boxes):
        ious = box_iou(pred_box.unsqueeze(0), gt_boxes)[0]
        max_iou, max_idx = ious.max(0) if len(ious) > 0 else (torch.tensor(0.0), torch.tensor(0))

        if max_iou >= iou_thresh and not gt_matched[max_idx]:
            tp[i] = 1
            gt_matched[max_idx] = True
        else:
            fp[i] = 1

    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recall = tp_cumsum / (len(gt_boxes) + 1e-8)

    precision = torch.cat([torch.tensor([1.0]), precision])
    recall = torch.cat([torch.tensor([0.0]), recall])

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    ap = torch.sum((recall[1:] - recall[:-1]) * precision[1:])
    return ap.item()


@torch.no_grad()
def evaluate(model, val_loader, device, num_classes):
    model.eval()
    all_aps = {c: [] for c in range(1, num_classes)}

    for imgs, targets in tqdm(val_loader, desc="Evaluating", leave=False):
        imgs = [img.to(device) for img in imgs]
        outputs = model(imgs)

        for output, target in zip(outputs, targets):
            gt_boxes = target['boxes'].to(device)
            gt_labels = target['labels'].to(device)
            pred_boxes = output['boxes']
            pred_labels = output['labels']
            pred_scores = output['scores']

            for c in range(1, num_classes):
                gt_mask = gt_labels == c
                pred_mask = pred_labels == c

                gt_c = gt_boxes[gt_mask]
                pred_c = pred_boxes[pred_mask]
                scores_c = pred_scores[pred_mask]

                if len(gt_c) == 0 and len(pred_c) == 0:
                    continue

                ap = compute_ap(pred_c, scores_c, gt_c)
                all_aps[c].append(ap)

    class_aps = {}
    for c in range(1, num_classes):
        if len(all_aps[c]) > 0:
            class_aps[c] = sum(all_aps[c]) / len(all_aps[c])
        else:
            class_aps[c] = 0.0

    mAP = sum(class_aps.values()) / len(class_aps) if class_aps else 0.0
    return mAP, class_aps


def train():
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.benchmark = True
    device = config.DEVICE
    print(f"Using: {device}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    train_data = StreetDataset(config.DATA_ROOT, "train", config.CLASSES, augment=True)
    val_data = StreetDataset(config.DATA_ROOT, "validation", config.CLASSES, augment=False)

    train_loader = DataLoader(
        train_data,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = FasterRCNN(config.NUM_CLASSES, config.BACKBONE)
    model.to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, config.LR_STEPS, config.LR_GAMMA
    )
    scaler = GradScaler('cuda')

    best_mAP = 0.0

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        for imgs, targets in pbar:
            imgs = [img.to(device, non_blocking=True) for img in imgs]
            targets = [
                {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            if epoch == 0:
                warmup_lr(optimizer, pbar.n, len(train_loader), config.LEARNING_RATE)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                losses = model(imgs, targets)
                loss = sum(losses.values())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)

        mAP, class_aps = evaluate(model, val_loader, device, config.NUM_CLASSES)

        print(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, Val mAP: {mAP:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        for c, ap in class_aps.items():
            print(f"  {config.CLASSES[c]}: AP={ap:.4f}")

        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, "best.pth"))
            print(f"  Saved best model (mAP: {mAP:.4f})")

        if (epoch + 1) % 4 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "mAP": mAP,
                },
                os.path.join(config.OUTPUT_DIR, f"ckpt_epoch{epoch + 1}.pth"),
            )

    print(f"\nTraining done! Best mAP: {best_mAP:.4f}")


def warmup_lr(optimizer, step, total_steps, target_lr):
    lr = target_lr * (step + 1) / total_steps
    for pg in optimizer.param_groups:
        pg["lr"] = lr


if __name__ == "__main__":
    train()
