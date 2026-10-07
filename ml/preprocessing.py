"""Image preprocessing shared by evaluation and the API.

Deliberately torch-free so the serving container only needs numpy + Pillow + onnxruntime.
Keeping one implementation guarantees the API sees images exactly as the model did at eval time.
"""
import io

import numpy as np
from PIL import Image, ImageOps

from ml.config import IMG_SIZE

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(data: bytes | str) -> Image.Image:
    """Open from bytes or path, applying phone EXIF rotation."""
    img = Image.open(io.BytesIO(data) if isinstance(data, bytes) else data)
    return ImageOps.exif_transpose(img).convert("RGB")


def resize_short_side(img: Image.Image, short: int) -> Image.Image:
    w, h = img.size
    scale = short / min(w, h)
    return img.resize((round(w * scale), round(h * scale)), Image.BILINEAR, reducing_gap=3.0)


def center_crop(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def to_model_input(img: Image.Image) -> np.ndarray:
    """PIL image -> normalised NCHW float32 array of shape (1, 3, IMG_SIZE, IMG_SIZE)."""
    img = center_crop(resize_short_side(img, IMG_SIZE))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)[None]
