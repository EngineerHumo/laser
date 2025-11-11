"""Spot classification inference script.

This utility loads detection results stored in YOLO text format alongside
their corresponding images, extracts per-spot crops and global context
images, and runs the trained dual-encoder model with the ArcFace head to
obtain final class predictions.  The predictions are visualised on the
original images with colour coded bounding boxes and a legend.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.transforms import functional as TF

from laser.datasets import SPOT_MEAN, SPOT_STD
from laser.models import DualEncoderMetricModel
from laser.models.losses import ArcMarginProduct


LOGGER = logging.getLogger("predict_classification")


# Mapping between class indices and human readable grade labels.
CLASS_GRADES = {
    0: "old",
    1: "1",
    2: "2",
    3: "2+",
    4: "3",
    5: "3+",
}


# Distinct colours (RGB) for bounding boxes/legend entries per class.
CLASS_COLORS = {
    0: (220, 20, 60),     # crimson
    1: (34, 139, 34),     # forest green
    2: (30, 144, 255),    # dodger blue
    3: (255, 140, 0),     # dark orange
    4: (138, 43, 226),    # blue violet
    5: (0, 206, 209),     # dark turquoise
}


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class Detection:
    """Container describing a YOLO-format detection."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run spot classification on YOLO detections")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/wensheng/jiaqi/laser/output_txt"),
        help="Directory containing images and YOLO detection txt files",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/wensheng/jiaqi/laser/logs/best.pt"),
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/wensheng/jiaqi/laser/output_classification"),
        help="Directory where annotated images and updated txt files are written",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device (e.g. 'cuda:0' or 'cpu'). Defaults to CUDA when available.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def select_device(explicit_device: str | None = None) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[DualEncoderMetricModel, ArcMarginProduct]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    LOGGER.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = DualEncoderMetricModel().to(device)
    model.load_state_dict(checkpoint["model_state"])

    arcface_state = checkpoint["arcface_state"]
    num_classes = arcface_state["weight"].shape[0]
    arcface = ArcMarginProduct(in_features=512, out_features=num_classes).to(device)
    arcface.load_state_dict(arcface_state)

    model.eval()
    arcface.eval()
    LOGGER.info("Loaded model with %d output classes", num_classes)
    return model, arcface


def read_detections(txt_path: Path) -> List[Detection]:
    detections: List[Detection] = []
    with txt_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if len(parts) < 5:
                LOGGER.warning("Skipping malformed line %d in %s", line_number, txt_path)
                continue
            try:
                class_id = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                LOGGER.warning("Non-numeric values on line %d in %s", line_number, txt_path)
                continue
            detections.append(Detection(class_id, x_center, y_center, width, height))
    return detections


def crop_spot(image: Image.Image, center_x: float, center_y: float, size: int = 64) -> Image.Image:
    half = size / 2.0
    left = int(round(center_x - half))
    top = int(round(center_y - half))
    right = left + size
    bottom = top + size

    patch = Image.new("RGB", (size, size), (0, 0, 0))
    src_left = max(left, 0)
    src_top = max(top, 0)
    src_right = min(right, image.width)
    src_bottom = min(bottom, image.height)

    if src_left >= src_right or src_top >= src_bottom:
        return patch

    crop = image.crop((src_left, src_top, src_right, src_bottom))
    paste_left = src_left - left
    paste_top = src_top - top
    patch.paste(crop, (paste_left, paste_top))
    return patch


def prepare_spot_tensor(spot_image: Image.Image, normalise: transforms.Normalize) -> torch.Tensor:
    tensor = TF.to_tensor(spot_image)
    tensor = normalise(tensor)
    return tensor


def prepare_global_tensor(image: Image.Image, normalise: transforms.Normalize) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    tensor = F.adaptive_avg_pool2d(tensor, (128, 128))
    tensor = normalise(tensor)
    return tensor


def iter_image_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and path.is_file():
            yield path


def legend_entries() -> Sequence[tuple[int, str]]:
    return [(cls, CLASS_GRADES.get(cls, str(cls))) for cls in sorted(CLASS_GRADES)]


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    """Return the width/height of *text* for the provided draw context."""

    if hasattr(draw, "textbbox"):
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
        except ValueError:
            pass
        else:
            return bbox[2] - bbox[0], bbox[3] - bbox[1]

    if hasattr(draw, "textsize"):
        width, height = draw.textsize(text, font=font)
        return int(width), int(height)

    if hasattr(font, "getsize"):
        width, height = font.getsize(text)  # type: ignore[attr-defined]
        return int(width), int(height)

    raise RuntimeError("Unable to measure text size with the provided font")


def draw_boxes(
    image: Image.Image,
    detections: Sequence[Detection],
    predictions: Sequence[int],
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    width, height = annotated.size
    for det, pred in zip(detections, predictions):
        box_width = det.width * width
        box_height = det.height * height
        center_x = det.x_center * width
        center_y = det.y_center * height

        left = center_x - box_width / 2.0
        top = center_y - box_height / 2.0
        right = center_x + box_width / 2.0
        bottom = center_y + box_height / 2.0

        color = CLASS_COLORS.get(pred, (255, 255, 255))
        draw.rectangle([left, top, right, bottom], outline=color, width=2)

        label = CLASS_GRADES.get(pred, str(pred))
        text = f"{label}"
        text_width, text_height = measure_text(draw, text, font)
        text_top = max(top - text_height - 4, 0)
        text_bottom = min(text_top + text_height + 4, height)
        text_bg = [left, text_top, left + text_width + 6, text_bottom]
        draw.rectangle(text_bg, fill=color)
        draw.text((left + 3, text_top + 2), text, fill=(0, 0, 0), font=font)

    draw_legend(draw, annotated.size, font)
    return annotated


def draw_legend(draw: ImageDraw.ImageDraw, image_size: tuple[int, int], font: ImageFont.ImageFont) -> None:
    padding = 10
    swatch_size = 18
    spacing = 6

    entries = legend_entries()
    if not entries:
        return

    text_widths = []
    text_heights = []
    for _, label in entries:
        width, height = measure_text(draw, label, font)
        text_widths.append(width)
        text_heights.append(height)

    max_text_width = max(text_widths)
    line_height = max(max(text_heights), swatch_size)
    legend_width = swatch_size + spacing + max_text_width
    legend_height = len(entries) * line_height + max(len(entries) - 1, 0) * spacing

    image_width, image_height = image_size
    legend_left = max(image_width - legend_width - 2 * padding, 0)
    legend_top = max(image_height - legend_height - 2 * padding, 0)
    legend_box = [
        legend_left,
        legend_top,
        min(image_width, legend_left + legend_width + 2 * padding),
        min(image_height, legend_top + legend_height + 2 * padding),
    ]

    if legend_box[2] <= legend_box[0] or legend_box[3] <= legend_box[1]:
        return

    draw.rectangle(legend_box, fill=(255, 255, 255))

    y = legend_box[1] + padding
    for class_id, label in entries:
        color = CLASS_COLORS.get(class_id, (0, 0, 0))
        swatch_left = legend_box[0] + padding
        swatch_top = y
        swatch_box = [swatch_left, swatch_top, swatch_left + swatch_size, swatch_top + swatch_size]
        draw.rectangle(swatch_box, fill=color, outline=None)
        text_position = (swatch_left + swatch_size + spacing, y)
        draw.text(text_position, label, fill=(0, 0, 0), font=font)
        y += line_height + spacing


def _counts_from_classes(classes: Sequence[int], num_classes: int) -> list[int]:
    counts = [0] * num_classes
    for cls in classes:
        if 0 <= cls < num_classes:
            counts[cls] += 1
    return counts


def _counts_to_distribution(counts: Sequence[int]) -> list[float]:
    total = float(sum(counts))
    if total <= 0:
        return [0.0 for _ in counts]
    return [c / total for c in counts]


def _remove_outlier_and_average(count_series: Sequence[Sequence[int]]) -> list[int]:
    if not count_series:
        return []

    num_classes = len(count_series[0])
    distributions = [_counts_to_distribution(counts) for counts in count_series]
    mean_distribution = [
        sum(dist[class_idx] for dist in distributions) / len(distributions)
        for class_idx in range(num_classes)
    ]

    distances = [
        sum(abs(dist[class_idx] - mean_distribution[class_idx]) for class_idx in range(num_classes))
        for dist in distributions
    ]
    outlier_index = int(np.argmax(distances))

    retained = [counts for idx, counts in enumerate(count_series) if idx != outlier_index]
    if not retained:
        retained = [count_series[outlier_index]]

    averaged = []
    for class_idx in range(num_classes):
        mean_count = sum(counts[class_idx] for counts in retained) / len(retained)
        averaged.append(max(0, int(round(mean_count))))
    return averaged


def _find_uncovered_zero(matrix: np.ndarray, row_cover: np.ndarray, col_cover: np.ndarray) -> tuple[int, int]:
    eps = 1e-9
    for i in range(matrix.shape[0]):
        if row_cover[i]:
            continue
        for j in range(matrix.shape[1]):
            if col_cover[j]:
                continue
            if abs(matrix[i, j]) <= eps:
                return i, j
    return -1, -1


def _find_star_in_row(star_matrix: np.ndarray, row: int) -> int | None:
    cols = np.where(star_matrix[row])[0]
    if cols.size == 0:
        return None
    return int(cols[0])


def _find_star_in_col(star_matrix: np.ndarray, col: int) -> int | None:
    rows = np.where(star_matrix[:, col])[0]
    if rows.size == 0:
        return None
    return int(rows[0])


def _find_prime_in_row(prime_matrix: np.ndarray, row: int) -> int | None:
    cols = np.where(prime_matrix[row])[0]
    if cols.size == 0:
        return None
    return int(cols[0])


def _hungarian_square(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = cost_matrix.copy()
    n = matrix.shape[0]

    matrix -= matrix.min(axis=1, keepdims=True)
    matrix -= matrix.min(axis=0, keepdims=True)

    star_matrix = np.zeros_like(matrix, dtype=bool)
    prime_matrix = np.zeros_like(matrix, dtype=bool)
    row_cover = np.zeros(n, dtype=bool)
    col_cover = np.zeros(n, dtype=bool)

    eps = 1e-9
    for i in range(n):
        for j in range(n):
            if row_cover[i] or col_cover[j]:
                continue
            if abs(matrix[i, j]) <= eps:
                star_matrix[i, j] = True
                row_cover[i] = True
                col_cover[j] = True
                break
    row_cover[:] = False
    col_cover[:] = False

    step = 3
    path: list[tuple[int, int]] = []

    while True:
        if step == 3:
            col_cover[:] = np.any(star_matrix, axis=0)
            if int(col_cover.sum()) == n:
                break
            step = 4
        elif step == 4:
            row, col = _find_uncovered_zero(matrix, row_cover, col_cover)
            if row == -1:
                step = 6
            else:
                prime_matrix[row, col] = True
                star_col = _find_star_in_row(star_matrix, row)
                if star_col is not None:
                    row_cover[row] = True
                    col_cover[star_col] = False
                else:
                    path = [(row, col)]
                    step = 5
        elif step == 5:
            done = False
            while not done:
                row, col = path[-1]
                star_row = _find_star_in_col(star_matrix, col)
                if star_row is None:
                    done = True
                    continue
                path.append((star_row, col))
                prime_col = _find_prime_in_row(prime_matrix, star_row)
                if prime_col is None:
                    done = True
                    continue
                path.append((star_row, prime_col))

            for r, c in path:
                if star_matrix[r, c]:
                    star_matrix[r, c] = False
                else:
                    star_matrix[r, c] = True

            prime_matrix[:, :] = False
            row_cover[:] = False
            col_cover[:] = False
            step = 3
        elif step == 6:
            uncovered_rows = ~row_cover
            uncovered_cols = ~col_cover
            if not np.any(uncovered_rows) or not np.any(uncovered_cols):
                step = 3
                continue
            uncovered_values = matrix[np.ix_(uncovered_rows, uncovered_cols)]
            min_uncovered = float(np.min(uncovered_values))
            matrix[row_cover, :] += min_uncovered
            matrix[:, ~col_cover] -= min_uncovered
            step = 4
        else:
            break

    row_indices, col_indices = np.where(star_matrix)
    return row_indices, col_indices


def _solve_linear_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if cost_matrix.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    num_rows, num_cols = cost_matrix.shape
    size = max(num_rows, num_cols)
    max_cost = float(np.max(cost_matrix)) if cost_matrix.size else 0.0
    padding_value = max_cost + abs(max_cost) + 1.0

    padded = np.full((size, size), padding_value, dtype=float)
    padded[:num_rows, :num_cols] = cost_matrix

    row_ind, col_ind = _hungarian_square(padded)
    valid = (row_ind < num_rows) & (col_ind < num_cols)
    return row_ind[valid], col_ind[valid]


def _apply_matching(
    similarity_matrix: np.ndarray,
    standard_counts: Sequence[int],
    fallback_classes: Sequence[int],
) -> list[int]:
    num_detections = similarity_matrix.shape[0]
    num_classes = similarity_matrix.shape[1] if similarity_matrix.ndim == 2 else 0

    target_labels: list[int] = []
    for class_idx, count in enumerate(standard_counts):
        if count <= 0:
            continue
        target_labels.extend([class_idx] * int(count))

    if not target_labels:
        return list(fallback_classes)

    selected = similarity_matrix[:, target_labels]
    cost_matrix = -selected
    row_ind, col_ind = _solve_linear_assignment(cost_matrix)

    assignments: list[int | None] = [None] * num_detections
    for row, col in zip(row_ind.tolist(), col_ind.tolist()):
        if 0 <= row < num_detections:
            assignments[row] = target_labels[col]

    final_classes: list[int] = []
    for idx in range(num_detections):
        assigned = assignments[idx]
        if assigned is None or not (0 <= assigned < num_classes):
            final_classes.append(int(fallback_classes[idx]))
        else:
            final_classes.append(int(assigned))
    return final_classes


def run_inference() -> None:
    args = parse_args()
    configure_logging()

    device = select_device(args.device)
    LOGGER.info("Using device: %s", device)

    model, arcface = load_checkpoint(args.checkpoint, device)
    normalise = transforms.Normalize(mean=SPOT_MEAN, std=SPOT_STD)

    input_dir = args.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(iter_image_files(input_dir))
    if not image_paths:
        LOGGER.warning("No images found in %s", input_dir)
        return

    reference_counts: list[int] | None = None
    argmax_history: list[list[int]] = []

    with torch.no_grad():
        for frame_index, image_path in enumerate(image_paths):
            txt_path = image_path.with_suffix(".txt")
            if not txt_path.exists():
                LOGGER.warning("Missing detection file for %s", image_path.name)
                continue

            detections = read_detections(txt_path)
            if not detections:
                LOGGER.info("No detections found for %s", image_path.name)
                # Still copy the original image for completeness.
                output_image_path = output_dir / image_path.name
                with Image.open(image_path) as img:
                    img.convert("RGB").save(output_image_path)
                continue

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                width, height = img.size

                spots = []
                for det in detections:
                    center_x = det.x_center * width
                    center_y = det.y_center * height
                    spot_patch = crop_spot(img, center_x, center_y)
                    spot_tensor = prepare_spot_tensor(spot_patch, normalise)
                    spots.append(spot_tensor)

                spot_batch = torch.stack(spots, dim=0).to(device)
                global_tensor = prepare_global_tensor(img, normalise)
                global_batch = global_tensor.unsqueeze(0).repeat(len(detections), 1, 1, 1).to(device)

                embeddings, _ = model(spot_batch, global_batch)
                logits = arcface.inference(embeddings)
                predicted = torch.argmax(logits, dim=1)

                similarity_matrix = logits.detach().cpu().numpy()
                fallback_classes = predicted.cpu().tolist()
                num_classes = similarity_matrix.shape[1]
                argmax_counts = _counts_from_classes(fallback_classes, num_classes)

                if reference_counts is None:
                    reference_counts = argmax_counts.copy()
                    standard_counts = reference_counts.copy()
                elif frame_index < 5:
                    standard_counts = reference_counts.copy()
                else:
                    window = argmax_history[-5:] + [argmax_counts]
                    if len(window) < 6:
                        standard_counts = reference_counts.copy()
                    else:
                        standard_counts = _remove_outlier_and_average(window)

                if len(standard_counts) != num_classes:
                    adjusted_counts = [0] * num_classes
                    usable = min(len(standard_counts), num_classes)
                    for idx in range(usable):
                        adjusted_counts[idx] = max(0, int(standard_counts[idx]))
                    standard_counts = adjusted_counts

                final_predictions = _apply_matching(
                    similarity_matrix, standard_counts, fallback_classes
                )
                argmax_history.append(argmax_counts)
                if len(argmax_history) > 5:
                    argmax_history = argmax_history[-5:]

                # Write updated detections with predicted class ids.
                output_txt_path = output_dir / txt_path.name
                with output_txt_path.open("w", encoding="utf-8") as handle:
                    for det, pred in zip(detections, final_predictions):
                        handle.write(
                            f"{pred} {det.x_center:.6f} {det.y_center:.6f} {det.width:.6f} {det.height:.6f}\n"
                        )

                annotated = draw_boxes(img, detections, final_predictions)
                output_image_path = output_dir / image_path.name
                annotated.save(output_image_path)

                grade_counts = {}
                for pred in final_predictions:
                    grade_counts[pred] = grade_counts.get(pred, 0) + 1
                counts_str = ", ".join(
                    f"{CLASS_GRADES.get(cls, cls)}: {count}" for cls, count in sorted(grade_counts.items())
                )
                LOGGER.info("Processed %s | %s", image_path.name, counts_str)


if __name__ == "__main__":
    run_inference()

