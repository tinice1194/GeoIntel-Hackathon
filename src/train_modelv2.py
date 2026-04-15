import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

from torchvision.models.segmentation import deeplabv3_resnet50

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(r"G:\GIS_AI_PROJECT")
PATCHES_DIR = PROJECT_ROOT / "data" / "patches"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 11
BATCH_SIZE = 8
EPOCHS = 80
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SegmentationDataset(Dataset):
    def __init__(self, split="train"):
        self.img_dir = PATCHES_DIR / split / "images"
        self.mask_dir = PATCHES_DIR / split / "masks"
        self.files = sorted(self.img_dir.glob("*.npy"))

        if not self.files:
            raise RuntimeError(f"No .npy images found in {self.img_dir}")

        print(f"Loaded {len(self.files)} {split} patches")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        mask_path = self.mask_dir / img_path.name

        image = np.load(img_path)        
        mask = np.load(mask_path)       


        image = image / 255.0


        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        if np.random.rand() > 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()


        image = np.transpose(image, (2, 0, 1))

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        return image, mask


def get_loaders():
    train_ds = SegmentationDataset("train")
    val_ds = SegmentationDataset("val")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader


def pixel_accuracy(preds, targets):
    preds = torch.argmax(preds, dim=1)
    correct = (preds == targets).float().sum()
    total = torch.numel(targets)
    return (correct / total).item()


def train_one_epoch(model, loader, optimizer, scaler, criterion, epoch):
    model.train()

    running_loss = 0.0
    running_acc = 0.0

    for images, masks in tqdm(loader, desc=f"Train {epoch}", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            outputs = model(images)["out"]              
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        running_acc += pixel_accuracy(outputs.detach(), masks) * images.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def evaluate(model, loader, criterion, epoch):
    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc=f"Val {epoch}", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                outputs = model(images)["out"]
                loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)
            running_acc += pixel_accuracy(outputs, masks) * images.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def main():
    train_loader, val_loader = get_loaders()

    model = deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.5
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_val_loss = float("inf")

    print(f"\nTraining on: {DEVICE}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, epoch
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, epoch
        )

        scheduler.step(val_loss)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = CHECKPOINT_DIR / "deeplab_best.pth"
            torch.save(model.state_dict(), path)
            print(f"Saved best model → {path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
