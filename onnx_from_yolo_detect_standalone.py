"""Standalone ONNX detection runner matching the ``onnx_produce.py`` export.

This script performs preprocessing, inference, non-max suppression, box
rescaling, visualization, and TXT export for every image in a target directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics.utils import ops

DEFAULT_MODEL_PATH = "/home/wensheng/jiaqi/ultralytics/runs/detect/train65/weights/best.onnx"
DEFAULT_SOURCE_DIR = "/home/wensheng/jiaqi/ultralytics/data/train/image"
DEFAULT_OUTPUT_DIR = "/home/wensheng/jiaqi/ultralytics/output_onnx_standalone"

# Keep the target image size aligned with the training/export pipeline.
TARGET_IMAGE_SIZE = 1240


class ONNXDetector:
    """Helper class to run ONNXRuntime inference with YOLO-style post-processing."""

    def __init__(self, model_path: Path, conf_thres: float, iou_thres: float):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.session = self._create_session()
        self.input_name = self.session.get_inputs()[0].name

    def _create_session(self) -> ort.InferenceSession:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ort.InferenceSession(str(self.model_path), providers=providers)

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        resized = cv2.resize(
            image, (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
        )
        scale_x = image.shape[1] / TARGET_IMAGE_SIZE
        scale_y = image.shape[0] / TARGET_IMAGE_SIZE

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))[None, ...]
        return chw, (scale_x, scale_y)

    def _run_session(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        return self.session.run(None, {self.input_name: input_tensor})

    def _postprocess_with_nms_layer(
        self, outputs: Sequence[np.ndarray]
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        detections: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        if not outputs:
            return detections

        output = outputs[0]
        if output.ndim == 3:
            output = np.squeeze(output, axis=0)
        if output.size == 0:
            return detections

        boxes, scores, class_ids = output[:, :4], output[:, 4], output[:, 5].astype(int)
        mask = scores >= self.conf_thres
        detections.append((boxes[mask], scores[mask], class_ids[mask]))
        return detections

    def _postprocess_manual_nms(
        self, outputs: Sequence[np.ndarray]
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not outputs:
            return []

        preds = outputs[0]

        preds_tensor = torch.from_numpy(preds)
        nms_results = ops.non_max_suppression(
            preds_tensor, conf_thres=self.conf_thres, iou_thres=self.iou_thres
        )

        detections: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for result in nms_results:
            if result is None or len(result) == 0:
                detections.append((np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)))
                continue
            boxes = result[:, :4].cpu().numpy()
            scores = result[:, 4].cpu().numpy()
            class_ids = result[:, 5].int().cpu().numpy()
            detections.append((boxes, scores, class_ids))
        return detections

    def postprocess(
        self, outputs: Sequence[np.ndarray]
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        first = outputs[0] if outputs else None
        if first is not None and first.shape[-1] == 6:
            return self._postprocess_with_nms_layer(outputs)
        return self._postprocess_manual_nms(outputs)

    def predict(self, image: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        input_tensor, scales = self.preprocess(image)
        raw_outputs = self._run_session(input_tensor)
        detections = self.postprocess(raw_outputs)
        return [self._scale_boxes(d, scales) for d in detections]

    def _scale_boxes(
        self, detection: Tuple[np.ndarray, np.ndarray, np.ndarray], scales: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes, scores, class_ids = detection
        if boxes.size == 0:
            return boxes, scores, class_ids

        scale_x, scale_y = scales
        boxes_scaled = boxes.copy()
        boxes_scaled[:, [0, 2]] *= scale_x
        boxes_scaled[:, [1, 3]] *= scale_y

        width = int(round(scale_x * TARGET_IMAGE_SIZE))
        height = int(round(scale_y * TARGET_IMAGE_SIZE))
        boxes_scaled[:, [0, 2]] = np.clip(boxes_scaled[:, [0, 2]], 0, width)
        boxes_scaled[:, [1, 3]] = np.clip(boxes_scaled[:, [1, 3]], 0, height)
        return boxes_scaled, scores, class_ids


def collect_images(source_dir: Path) -> List[Path]:
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in valid_suffixes
    )
    if not images:
        raise FileNotFoundError(f"No images found in {source_dir!s}")
    return images


def draw_detections(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
    colors = (
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    )
    annotated = image.copy()
    for (x1, y1, x2, y2), score, class_id in zip(boxes, scores, class_ids):
        color = colors[int(class_id) % len(colors)]
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{int(class_id)}:{score:.2f}"
        cv2.putText(
            annotated,
            label,
            (int(x1), max(0, int(y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def save_detections_to_txt(output_image_path: Path, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, image_shape: Tuple[int, int]) -> None:
    txt_path = output_image_path.with_suffix(".txt")
    height, width = image_shape
    with open(txt_path, "w", encoding="utf-8") as file:
        for (x1, y1, x2, y2), score, class_id in zip(boxes, scores, class_ids):
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            w_norm = (x2 - x1) / width
            h_norm = (y2 - y1) / height
            file.write(
                f"{int(class_id)} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} {float(score):.6f}\n"
            )


def process_directory(detector: ONNXDetector, source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = collect_images(source_dir)

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image {image_path!s}")

        detections = detector.predict(image)
        boxes, scores, class_ids = detections[0]

        annotated = draw_detections(image, boxes, scores, class_ids)
        output_image_path = output_dir / image_path.name
        cv2.imwrite(str(output_image_path), annotated)
        save_detections_to_txt(output_image_path, boxes, scores, class_ids, image.shape[:2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone ONNX detection on a directory of images.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to the ONNX model file.")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE_DIR, help="Directory containing images to detect.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save annotated outputs.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    source_dir = Path(args.source)
    output_dir = Path(args.output)

    detector = ONNXDetector(model_path, conf_thres=args.conf, iou_thres=args.iou)
    process_directory(detector, source_dir, output_dir)
    print(f"Detections saved to {output_dir!s}")


if __name__ == "__main__":
    main()
