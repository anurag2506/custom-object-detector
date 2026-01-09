import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import config
from src import FasterRCNN
from dataset import StreetDataset, collate_fn


def train():
    # setup
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.benchmark = True
    device = config.DEVICE
    print(f"Using: {device}")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # data
    train_data = StreetDataset(config.DATA_ROOT, "train", config.CLASSES, augment=True)
    val_data = StreetDataset(
        config.DATA_ROOT, "validation", config.CLASSES, augment=False
    )

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

    # model
    model = FasterRCNN(config.NUM_CLASSES, config.BACKBONE)
    model.to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, config.LR_STEPS, config.LR_GAMMA
    )
    scaler = GradScaler()

    best_loss = float("inf")

    # training loop
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

            # warmup lr for first epoch
            if epoch == 0:
                warmup_lr(optimizer, pbar.n, len(train_loader), config.LEARNING_RATE)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
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
        print(
            f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, "best.pth"))
            print("  Saved best model")

        # save checkpoint
        if (epoch + 1) % 4 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(config.OUTPUT_DIR, f"ckpt_epoch{epoch + 1}.pth"),
            )

    print(f"\nTraining done! Best loss: {best_loss:.4f}")


def warmup_lr(optimizer, step, total_steps, target_lr):
    """Linear warmup"""
    lr = target_lr * (step + 1) / total_steps
    for pg in optimizer.param_groups:
        pg["lr"] = lr


if __name__ == "__main__":
    train()
