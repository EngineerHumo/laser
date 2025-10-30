"""Data preparation utilities for spot classification dataset.

This script iterates over the ``origin_data`` directory, where each sample is
stored as a pair of ``.json`` annotations and ``.bmp`` images. For every spot
annotation inside a JSON file, a 64x64 crop centred on the annotated rectangle
is extracted from the corresponding BMP image. The crops are saved into
``data/spot`` and the class labels are written to ``data/labels.txt``.

After all spots of an image are processed, the whole BMP image is down-sampled
using average pooling to ``128x128`` pixels and saved to ``data/image``.

Example usage::

    python data_produce.py --origin-dir origin_data --output-dir data

The resulting directory layout will be::

    data/
        image/          # down-sampled 128x128 images
        spot/           # 64x64 crops around each spot
        labels.txt      # mapping between spot crops and grade labels

The ``labels.txt`` file contains space separated ``<spot_name> <grade>`` pairs,
where grades follow the mapping specified in the user instructions.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageOps

LOGGER = logging.getLogger(__name__)

LABEL_TO_GRADE: Dict[str, int] = {
    "old": 0,
    "1": 1,
    "2": 2,
    "2+": 3,
    "3": 4,
    "3+": 5,
}


def _ensure_directories(base_dir: Path) -> Tuple[Path, Path, Path]:
    image_dir = base_dir / "image"
    spot_dir = base_dir / "spot"
    image_dir.mkdir(parents=True, exist_ok=True)
    spot_dir.mkdir(parents=True, exist_ok=True)
    label_path = base_dir / "labels.txt"
    return image_dir, spot_dir, label_path


def _parse_points(points: Sequence[Sequence[float]]) -> Tuple[float, float]:
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    return cx, cy


def _crop_with_padding(image: Image.Image, center: Tuple[float, float], size: int) -> Image.Image:
    half = size // 2
    cx, cy = center
    left = int(round(cx)) - half
    upper = int(round(cy)) - half
    right = left + size
    lower = upper + size

    pad_left = max(0, -left)
    pad_upper = max(0, -upper)
    pad_right = max(0, right - image.width)
    pad_lower = max(0, lower - image.height)

    if pad_left or pad_upper or pad_right or pad_lower:
        fill_value = (
            tuple(int(x) for x in image.getpixel((0, 0))) if image.mode == "RGB" else 0
        )
        image = ImageOps.expand(
            image,
            border=(pad_left, pad_upper, pad_right, pad_lower),
            fill=fill_value,
        )
        left += pad_left
        upper += pad_upper

    return image.crop((left, upper, left + size, upper + size))


def _load_annotation(path: Path) -> Iterable[Tuple[int, Tuple[float, float], int]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    shapes: List[Dict] = data.get("shapes", [])
    for index, shape in enumerate(shapes, start=1):
        label = str(shape.get("label", "")).strip()
        if label not in LABEL_TO_GRADE:
            LOGGER.warning("Unknown label '%s' in %s, skipping", label, path)
            continue
        grade = LABEL_TO_GRADE[label]
        points = shape.get("points")
        if not points:
            LOGGER.warning("Missing points for shape %s in %s", index, path)
            continue
        center = _parse_points(points)
        yield index, center, grade


def _downsample_to_128(image: Image.Image) -> Image.Image:
    # BOX filter corresponds to averaging when down-sampling.
    return image.resize((128, 128), resample=Image.BOX)


def process_dataset(origin_dir: Path, output_dir: Path, overwrite_labels: bool = True) -> None:
    image_dir, spot_dir, label_path = _ensure_directories(output_dir)

    json_files = sorted(origin_dir.glob("*.json"))
    if not json_files:
        LOGGER.error("No JSON files found in %s", origin_dir)
        raise FileNotFoundError(f"No JSON files found in {origin_dir}")

    if overwrite_labels and label_path.exists():
        label_path.unlink()

    with label_path.open("a", encoding="utf-8") as label_file:
        for json_path in json_files:
            stem = json_path.stem
            bmp_path = origin_dir / f"{stem}.bmp"
            if not bmp_path.exists():
                LOGGER.warning("Missing BMP file for %s, skipping", json_path.name)
                continue

            image = Image.open(bmp_path).convert("RGB")
            LOGGER.info("Processing %s (%dx%d)", json_path.name, image.width, image.height)

            for index, center, grade in _load_annotation(json_path):
                crop = _crop_with_padding(image, center, 64)
                crop_name = f"{stem}_{index:03d}.png"
                crop.save(spot_dir / crop_name)
                label_file.write(f"{stem}_{index:03d} {grade}\n")

            downsampled = _downsample_to_128(image)
            downsampled.save(image_dir / f"{stem}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare spot classification dataset")
    parser.add_argument("--origin-dir", type=Path, default=Path("origin_data"), help="Directory containing JSON and BMP files")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Output directory for processed dataset")
    parser.add_argument("--keep-labels", action="store_true", help="Append to existing labels.txt instead of overwriting")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    overwrite = not args.keep_labels
    process_dataset(args.origin_dir, args.output_dir, overwrite_labels=overwrite)
    LOGGER.info("Dataset preparation finished. Spot crops and labels are available in %s", args.output_dir)


if __name__ == "__main__":
    main()
