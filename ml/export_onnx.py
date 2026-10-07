"""Export the trained checkpoint to ONNX (with softmax baked in) and check it matches PyTorch.

Usage: python -m ml.export_onnx
"""
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from ml.config import CHECKPOINT_PATH, IMG_SIZE, ONNX_PATH, PROCESSED_DIR
from ml.model import build_model
from ml.preprocessing import load_image, to_model_input


def main() -> None:
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu")["state_dict"])
    wrapped = nn.Sequential(model, nn.Softmax(dim=1)).eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        wrapped, dummy, ONNX_PATH, input_names=["image"], output_names=["probs"],
        dynamic_axes={"image": {0: "batch"}, "probs": {0: "batch"}}, opset_version=17, dynamo=False,
    )

    # Parity check on a few real test images through the API's preprocessing path.
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    samples = sorted((PROCESSED_DIR / "test").glob("*/*.jpg"))[::40]
    max_diff = 0.0
    for f in samples:
        x = to_model_input(load_image(str(f)))
        onnx_p = session.run(None, {"image": x})[0]
        with torch.no_grad():
            torch_p = wrapped(torch.from_numpy(x)).numpy()
        max_diff = max(max_diff, float(np.abs(onnx_p - torch_p).max()))
    print(f"exported {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB); "
          f"max |onnx - torch| over {len(samples)} images = {max_diff:.2e}")
    assert max_diff < 1e-3, "ONNX output diverges from PyTorch"


if __name__ == "__main__":
    main()
