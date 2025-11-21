"""
Export a trained PyTorch spot classifier to ONNX and run spot-level scoring.

The script assumes a trained model checkpoint is stored at ``MODEL_WEIGHTS_PATH``
(in the form of a state_dict) and that inference images live under
``SPOT_IMAGE_ROOT``. Images are loaded individually (spot cut-outs or full
frames) and scored to produce per-class probabilities using the exported ONNX
model alone.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Fixed paths for model weights, ONNX output, and inference data.
MODEL_WEIGHTS_PATH = Path("/workspace/laser/data/spot_classifier.pt")
ONNX_OUTPUT_PATH = Path("/workspace/laser/artifacts/spot_classifier.onnx")
SPOT_IMAGE_ROOT = Path("/workspace/laser/data/images")
RESULT_PATH = Path("/workspace/laser/artifacts/spot_probabilities.json")

# Image preprocessing defaults.
IMAGE_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@dataclass
class ModelConfig:
    """Configuration required to rebuild the trained model."""

    in_channels: int = 3
    num_classes: int = 4
    dropout: float = 0.2


class SpotClassifier(nn.Module):
    """A compact CNN backbone for spot classification.

    The architecture must match the one used during training so the loaded
    ``state_dict`` aligns. Adjust the layers and ``ModelConfig`` as needed if
    your checkpoint expects something different.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(config.in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def load_trained_model(config: ModelConfig) -> SpotClassifier:
    """Instantiate the model and load weights from ``MODEL_WEIGHTS_PATH``."""

    if not MODEL_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained weights at {MODEL_WEIGHTS_PATH}. Provide a valid checkpoint."
        )

    model = SpotClassifier(config)
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_to_onnx(model: nn.Module, config: ModelConfig) -> None:
    """Export the PyTorch model to ONNX with dynamic batch size support."""

    ONNX_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, config.in_channels, *IMAGE_SIZE)
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_OUTPUT_PATH,
        export_params=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )
    print(f"Exported ONNX model to {ONNX_OUTPUT_PATH}")


def load_onnx_session() -> ort.InferenceSession:
    """Create an ONNX Runtime session for inference."""

    if not ONNX_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {ONNX_OUTPUT_PATH}. Run export_to_onnx first."
        )

    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(ONNX_OUTPUT_PATH.as_posix(), providers=providers)


def preprocess_image(image_path: Path) -> np.ndarray:
    """Load and normalize an image for model inference."""

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(IMAGE_SIZE)
        np_img = np.array(img).astype(np.float32) / 255.0
    np_img = (np_img - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    np_img = np.transpose(np_img, (2, 0, 1))  # CHW
    return np.expand_dims(np_img, axis=0)


def list_images(root: Path) -> Iterable[Path]:
    """Yield image files under ``root`` recursively."""

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for path in root.rglob("*"):
        if path.suffix.lower() in exts:
            yield path


def run_onnx_inference(session: ort.InferenceSession, images: Iterable[Path]) -> List[Tuple[Path, List[float]]]:
    """Run inference on each image and return softmax probabilities."""

    results: List[Tuple[Path, List[float]]] = []
    for image_path in images:
        inputs = preprocess_image(image_path)
        outputs = session.run(None, {"input": inputs})
        logits = outputs[0]
        probs = softmax_np(logits[0])
        results.append((image_path, probs.tolist()))
    return results


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """Stable softmax for numpy arrays."""

    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def save_probabilities(results: List[Tuple[Path, List[float]]]) -> None:
    """Persist probabilities to ``RESULT_PATH`` in JSON format."""

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": ONNX_OUTPUT_PATH.name,
        "source_root": SPOT_IMAGE_ROOT.as_posix(),
        "spots": [
            {"image": str(path.relative_to(SPOT_IMAGE_ROOT)), "probabilities": probs}
            for path, probs in results
        ],
    }
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved probabilities to {RESULT_PATH}")


def main() -> None:
    config = ModelConfig()
    model = load_trained_model(config)
    export_to_onnx(model, config)
    session = load_onnx_session()
    images = list(list_images(SPOT_IMAGE_ROOT))
    if not images:
        raise FileNotFoundError(f"No images found under {SPOT_IMAGE_ROOT}")
    results = run_onnx_inference(session, images)
    save_probabilities(results)


if __name__ == "__main__":
    main()