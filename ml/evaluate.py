"""Evaluate the trained model and calibrate the Good / Okay / Defective thresholds.

The dataset only has two labels, so the three app-facing levels come from P(defective):
    p < good_below        -> good
    p > defective_above   -> defective
    otherwise             -> okay  ("not clearly fine, get it checked")

Thresholds are picked on the validation set with safety first:
  * good_below:      at most MAX_DEFECTIVE_AS_GOOD of defective val tyres may score below it
  * defective_above: at most MAX_GOOD_AS_DEFECTIVE of good val tyres may score above it
Both are capped at the defaults so there is always an "okay" band. The test set is only used
for reporting.

Usage: python -m ml.evaluate
"""
import json

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

from ml.config import CHECKPOINT_PATH, DEFAULT_THRESHOLDS, METRICS_PATH, THRESHOLDS_PATH
from ml.model import build_model
from ml.train import EVAL_TF, get_device, load_split

MAX_DEFECTIVE_AS_GOOD = 0.02
MAX_GOOD_AS_DEFECTIVE = 0.05


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), 1)[:, 1].cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def calibrate(p: np.ndarray, y: np.ndarray) -> dict:
    good_below = float(np.quantile(p[y == 1], MAX_DEFECTIVE_AS_GOOD))
    defective_above = float(np.quantile(p[y == 0], 1 - MAX_GOOD_AS_DEFECTIVE))
    return {
        "good_below": round(min(good_below, DEFAULT_THRESHOLDS["good_below"]), 4),
        "defective_above": round(max(defective_above, DEFAULT_THRESHOLDS["defective_above"]), 4),
    }


def three_level(p: np.ndarray, t: dict) -> np.ndarray:
    return np.where(p < t["good_below"], "good", np.where(p > t["defective_above"], "defective", "okay"))


def report(name: str, p: np.ndarray, y: np.ndarray, t: dict) -> dict:
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    levels = three_level(p, t)
    breakdown = {
        true: {lvl: int(((y == idx) & (levels == lvl)).sum()) for lvl in ("good", "okay", "defective")}
        for idx, true in enumerate(["good", "defective"])
    }
    m = {
        "n": int(len(y)),
        "accuracy@0.5": round(float((pred == y).mean()), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "defective_recall@0.5": round(float(tp / (tp + fn)), 4),
        "good_recall@0.5": round(float(tn / (tn + fp)), 4),
        "confusion@0.5 [[tn, fp], [fn, tp]]": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "three_level_breakdown": breakdown,
        "defective_called_good": breakdown["defective"]["good"],
    }
    print(f"\n== {name} ==")
    for k, v in m.items():
        print(f"  {k}: {v}")
    return m


def main() -> None:
    device = get_device()
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device)["state_dict"])

    p_val, y_val = predict(model, load_split("val", EVAL_TF, 32, shuffle=False), device)
    p_test, y_test = predict(model, load_split("test", EVAL_TF, 32, shuffle=False), device)

    thresholds = calibrate(p_val, y_val)
    print(f"thresholds: {thresholds}")
    THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2))

    metrics = {"thresholds": thresholds, "val": report("val", p_val, y_val, thresholds),
               "test": report("test", p_test, y_test, thresholds)}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {THRESHOLDS_PATH} and {METRICS_PATH}")


if __name__ == "__main__":
    main()
