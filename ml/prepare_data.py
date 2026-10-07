"""Stratified train/val/test split + resized image cache.

The raw photos are ~4000x1800, so decoding them every epoch would dominate training time.
Train images are cached at a slightly larger size (room for random crops); val/test are cached
already resized to the model's short side so evaluation matches the API preprocessing.
Exact duplicate files are removed before splitting.

Usage: python -m ml.prepare_data
"""
import hashlib
import random
import shutil

from ml.config import CLASSES, IMG_SIZE, PROCESSED_DIR, RAW_DIR, SEED, SPLIT, TRAIN_CACHE_SHORT
from ml.preprocessing import load_image, resize_short_side


def main() -> None:
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    rng = random.Random(SEED)
    seen: set[str] = set()
    for cls in CLASSES:
        files = []
        # Drop byte-identical duplicates so the same photo can't land in both train and test.
        for p in sorted(p for p in (RAW_DIR / cls).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
            digest = hashlib.md5(p.read_bytes()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                files.append(p)
        rng.shuffle(files)
        n_train = int(len(files) * SPLIT["train"])
        n_val = int(len(files) * SPLIT["val"])
        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, split_files in splits.items():
            out_dir = PROCESSED_DIR / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            short = TRAIN_CACHE_SHORT if split == "train" else IMG_SIZE
            for f in split_files:
                resize_short_side(load_image(str(f)), short).save(out_dir / f"{f.stem}.jpg", quality=95)
            print(f"{split:5s} {cls:9s} {len(split_files)}")


if __name__ == "__main__":
    main()
