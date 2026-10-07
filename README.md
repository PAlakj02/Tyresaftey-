# TireGuard — ML layer

Tyre condition screening model and inference API for the TireGuard app.
The app sends a tyre photo to `POST /predict`; the API returns **good / okay / defective**
(or asks for a retake if the photo is unusable).

```
App camera ──photo──▶ POST /predict ──▶ quality gate ──▶ EfficientNet-B0 (ONNX) ──▶ P(defective) ──▶ good / okay / defective
                                           │
                                           └─ too blurry / dark / small ──▶ status: "retake"
```

## Quick start

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements-train.txt

# 1. Get the data (needs a Kaggle API token) -> data/Digital images of defective and good condition tyres/
kaggle datasets download warcoder/tyre-quality-classification -p data --unzip

# 2. Pipeline
python -m ml.prepare_data   # dedupe, stratified 70/15/15 split, resized cache -> data/processed/
python -m ml.train          # fine-tune EfficientNet-B0 -> artifacts/model.pt
python -m ml.evaluate       # test metrics + calibrate thresholds -> artifacts/{thresholds,metrics}.json
python -m ml.export_onnx    # -> artifacts/model.onnx (+ parity check vs PyTorch)

# 3. Serve
uvicorn ml.api.main:app --host 0.0.0.0 --port 8000
```

Interactive docs: http://localhost:8000/docs

Serving only needs `requirements-api.txt` plus `artifacts/model.onnx` and `artifacts/thresholds.json` — no PyTorch.

## API contract (for the app team)

### `POST /predict`
`multipart/form-data` with one field `image` (JPEG/PNG, ≤ 15 MB).

```bash
curl -F "image=@tyre.jpg" http://localhost:8000/predict
```

**Result**
```json
{
  "status": "ok",
  "condition": "defective",
  "confidence": 0.97,
  "probabilities": {"good": 0.03, "defective": 0.97},
  "message": "Visible defect detected (e.g. cracks, bulges or heavy wear). Get this tyre inspected before your next long drive.",
  "disclaimer": "TireGuard is a visual screening aid, not a professional inspection. ...",
  "quality": {"ok": true, "reason": null, "sharpness": 312.4, "brightness": 88.1, "width": 4000, "height": 1800},
  "model_version": "effnet-b0-v1"
}
```

**Unusable photo** (HTTP 200, the app should prompt to re-capture):
```json
{"status": "retake", "message": "Image is blurry. Hold the phone steady and tap to focus on the tyre.", "quality": {...}, "model_version": "effnet-b0-v1"}
```

HTTP `400` = not an image, `413` = file too large.

| `condition` | Meaning shown to the user |
|---|---|
| `good` | No visible defects detected |
| `okay` | Possible wear/damage — recommend professional inspection soon |
| `defective` | Visible defect — get it inspected before driving far |

### `GET /health`
Returns model version and the active thresholds.

### Photo guidance for the capture screen
The model was trained on **close-up photos of the tread or sidewall** filling most of the frame.
The framing guide should ask for that (not the whole car/wheel). The centre square of the photo is
what the model looks at.

## Current results (`effnet-b0-v1`)

Held-out test set (275 photos, never seen in training or threshold tuning):

| Metric | Value |
|---|---|
| Accuracy (2-class @ 0.5) | 96.7% |
| ROC-AUC | 0.998 |
| Defective recall / Good recall | 95.3% / 98.4% |

Three-level output on the test set (thresholds `good_below=0.269`, `defective_above=0.65`):

| True \ Predicted | good | okay | defective |
|---|---|---|---|
| good (125) | 123 | 1 | 1 |
| defective (150) | **2** | 8 | 140 |

2 of 150 defective tyres (1.3%) were called good. Full numbers in `artifacts/metrics.json`.
Inference is ~80 ms per request on a laptop CPU, including the quality check.

## How the three levels work

The dataset only has two labels (`good`, `defective`), so the model outputs P(defective) and
`ml/evaluate.py` picks two thresholds on the validation set:

- `good_below` — at most 2% of defective tyres may fall below it (safety: rarely call a bad tyre good)
- `defective_above` — at most 5% of good tyres may exceed it
- everything in between is **okay**, i.e. "not clearly fine, get it checked"

Thresholds live in `artifacts/thresholds.json` and are loaded by the API at start-up, so they can be
tuned without retraining.

## Project layout

```
ml/
  config.py         paths, classes, image size, split ratios
  preprocessing.py  torch-free preprocessing shared by eval and the API
  prepare_data.py   dedupe + split + resize cache
  model.py          EfficientNet-B0 with 2-class head
  train.py          training loop (augmentations mimic phone photos)
  evaluate.py       metrics + threshold calibration
  export_onnx.py    ONNX export with softmax baked in + parity check
  quality.py        blur / brightness / resolution gate
  api/main.py       FastAPI service
artifacts/          model.onnx, thresholds.json, metrics.json (model.pt is git-ignored)
```

## Dataset

[Tyre Quality Classification](https://www.kaggle.com/datasets/warcoder/tyre-quality-classification)
by warcoder, CC BY 4.0. 1,856 photos (828 good, 1,028 defective; 28 exact duplicates removed).

## Known limitations

- Trained on close-up shots only; whole-wheel or whole-car photos are out of distribution.
- No "not a tyre" detection yet — any image gets a good/okay/defective answer.
- "okay" is a confidence band, not a learned class; a labelled middle class would be a later improvement.
