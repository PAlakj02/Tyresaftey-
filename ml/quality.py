"""Image quality gate run before the model: reject photos too blurry, dark, bright or small to judge.

Thresholds were set from the training photos (see README) so that genuine dataset images pass.
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image

ANALYSIS_SHORT_SIDE = 512  # measure at a fixed scale so blur scores don't depend on camera resolution

MIN_SHORT_SIDE = 224
MIN_SHARPNESS = 8.0      # variance of Laplacian at ANALYSIS_SHORT_SIDE
MIN_BRIGHTNESS = 25.0    # mean grey level, 0-255
MAX_BRIGHTNESS = 235.0


@dataclass
class QualityResult:
    ok: bool
    reason: str | None
    sharpness: float
    brightness: float
    width: int
    height: int

    def to_dict(self) -> dict:
        return asdict(self)


def check_quality(img: Image.Image) -> QualityResult:
    w, h = img.size
    grey = np.asarray(img.convert("L"))
    scale = ANALYSIS_SHORT_SIDE / min(w, h)
    if scale < 1:
        grey = cv2.resize(grey, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    brightness = float(grey.mean())

    reason = None
    if min(w, h) < MIN_SHORT_SIDE:
        reason = "Image resolution is too low. Move closer or use a higher camera resolution."
    elif brightness < MIN_BRIGHTNESS:
        reason = "Image is too dark. Turn on the flash or move to better light."
    elif brightness > MAX_BRIGHTNESS:
        reason = "Image is overexposed. Avoid direct glare on the tyre."
    elif sharpness < MIN_SHARPNESS:
        reason = "Image is blurry. Hold the phone steady and tap to focus on the tyre."

    return QualityResult(reason is None, reason, round(sharpness, 1), round(brightness, 1), w, h)
