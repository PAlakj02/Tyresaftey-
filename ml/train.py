"""Fine-tune EfficientNet-B0 on good vs defective tyres.

Usage: python -m ml.train [--epochs 20] [--batch-size 32] [--lr 3e-4]
Saves the best checkpoint (by validation loss) to artifacts/model.pt.
"""
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ml.config import ARTIFACTS_DIR, CHECKPOINT_PATH, CLASSES, IMG_SIZE, PROCESSED_DIR, SEED
from ml.model import build_model
from ml.preprocessing import MEAN, STD

# Augmentations approximate real phone captures: odd angles, lighting, focus, framing.
TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2, hue=0.02),
    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN.tolist(), STD.tolist()),
])
# Val/test are cached at IMG_SIZE short side, so this is identical to ml.preprocessing.to_model_input.
EVAL_TF = transforms.Compose([
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN.tolist(), STD.tolist()),
])


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split(split: str, tf, batch_size: int, shuffle: bool) -> DataLoader:
    # ImageFolder sorts folders alphabetically (defective=0); remap so index 1 = defective as in CLASSES.
    ds = datasets.ImageFolder(PROCESSED_DIR / split, transform=tf)
    ds.samples = [(p, CLASSES.index(ds.classes[t])) for p, t in ds.samples]
    ds.targets = [t for _, t in ds.samples]
    ds.classes, ds.class_to_idx = CLASSES, {c: i for i, c in enumerate(CLASSES)}
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, persistent_workers=True)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = get_device()
    print(f"device: {device}")

    train_loader = load_split("train", TRAIN_TF, args.batch_size, shuffle=True)
    val_loader = load_split("val", EVAL_TF, args.batch_size, shuffle=False)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "epoch": epoch}, CHECKPOINT_PATH)
            marker = "  * saved"
        print(f"epoch {epoch:2d}/{args.epochs}  train loss {train_loss:.4f} acc {train_acc:.3f}  "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}  ({time.time() - t0:.0f}s){marker}")

    print(f"best val loss {best_val_loss:.4f} -> {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
