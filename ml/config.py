"""Shared paths and constants for the TireGuard ML pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "Digital images of defective and good condition tyres"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"

CHECKPOINT_PATH = ARTIFACTS_DIR / "model.pt"
ONNX_PATH = ARTIFACTS_DIR / "model.onnx"
THRESHOLDS_PATH = ARTIFACTS_DIR / "thresholds.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# Index order matters: the model's output 1 is P(defective).
CLASSES = ["good", "defective"]

IMG_SIZE = 256          # model input (square)
TRAIN_CACHE_SHORT = 320  # train images cached a bit larger so random crops have room

SEED = 42
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

# Used if thresholds.json has not been produced by evaluate.py yet.
DEFAULT_THRESHOLDS = {"good_below": 0.35, "defective_above": 0.65}
