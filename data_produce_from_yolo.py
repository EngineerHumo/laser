"""Dataset preparation utilities for the fixed train/validation pipeline.

The dataset used by the training code is produced from the YOLO style
annotations that live under ``yolo_data``.  Each ``.bmp`` file inside the
``laser/images/<split>`` directory has a sibling ``.json`` file located in the
root of ``yolo_data``.  The JSON document contains a ``shapes`` list, where
every entry provides a ``label`` (grade name) and a ``points`` rectangle around
an annotated spot.

For every annotation the script copies a ``128x128`` crop centred on the
rectangle and stores it as ``data/<split>/spot/<image>_<spot>.png``.  The grades
are stored in ``data/<split>/labels.txt`` keeping the ``<crop_name> <grade>``
format used throughout the project.  Once all spots of an image are processed,
the original BMP is average-pooled down to ``128x128`` pixels and saved as
``data/<split>/image/<image>.png``.

Both the training and validation splits are generated in one go::

    python data_produce_from_yolo.py --yolo-root /path/to/yolo_data \
        --output-root data

The script overwrites the output directories by default.  Use ``--keep-labels``
when appending to existing label files is required.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

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

DEFAULT_YOLO_ROOT = Path("/home/wensheng/gjq_workspace/laser/yolo_data")


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
    """Down-sample ``image`` to ``128x128`` pixels using average pooling."""

    return image.resize((128, 128), resample=Image.BOX)


def _iter_bmp_files(directory: Path) -> Iterator[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    return iter(sorted(directory.glob("*.bmp")))


def process_split(
    *,
    split: str,
    image_dir: Path,
    annotation_dir: Path,
    output_dir: Path,
    overwrite_labels: bool = True,
) -> None:
    LOGGER.info("Processing split '%s'", split)
    image_output_dir, spot_output_dir, label_path = _ensure_directories(output_dir)

    if overwrite_labels and label_path.exists():
        label_path.unlink()

    total_spots = 0
    processed_images = 0

    with label_path.open("a", encoding="utf-8") as label_file:
        for bmp_path in _iter_bmp_files(image_dir):
            stem = bmp_path.stem
            json_path = annotation_dir / f"{stem}.json"
            if not json_path.exists():
                LOGGER.warning("Annotation not found for %s", bmp_path.name)
                continue

            with Image.open(bmp_path) as image:
                image = image.convert("RGB")
                LOGGER.debug(
                    "Split %s | %s (%dx%d)", split, bmp_path.name, image.width, image.height
                )

                annotation_iter = list(_load_annotation(json_path))
                if not annotation_iter:
                    LOGGER.warning("No valid annotations in %s", json_path.name)
                for index, center, grade in annotation_iter:
                    crop = _crop_with_padding(image, center, 128)
                    crop_name = f"{stem}_{index:03d}.png"
                    crop.save(spot_output_dir / crop_name)
                    label_file.write(f"{stem}_{index:03d} {grade}\n")
                    total_spots += 1

                downsampled = _downsample_to_128(image)
                downsampled.save(image_output_dir / f"{stem}.png")
                processed_images += 1

    LOGGER.info(
        "Finished split '%s': %d images, %d spot crops", split, processed_images, total_spots
    )


def process_dataset(yolo_root: Path, output_root: Path, overwrite_labels: bool = True) -> None:
    annotation_dir = yolo_root
    image_root = yolo_root / "laser" / "images"

    for split in ("train", "val"):
        split_image_dir = image_root / split
        split_output_dir = output_root / split
        process_split(
            split=split,
            image_dir=split_image_dir,
            annotation_dir=annotation_dir,
            output_dir=split_output_dir,
            overwrite_labels=overwrite_labels,
        )


def _default_yolo_root() -> Path:
    if DEFAULT_YOLO_ROOT.exists():
        return DEFAULT_YOLO_ROOT
    return Path("yolo_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare train/val spot datasets from YOLO annotations")
    parser.add_argument(
        "--yolo-root",
        type=Path,
        default=_default_yolo_root(),
        help="Root directory containing YOLO JSON files and laser/images/<split> folders",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Destination directory where train/ and val/ folders will be created",
    )
    parser.add_argument(
        "--keep-labels",
        action="store_true",
        help="Append to existing labels.txt instead of overwriting them",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    overwrite = not args.keep_labels
    process_dataset(args.yolo_root, args.output_root, overwrite_labels=overwrite)
    LOGGER.info(
        "Dataset preparation finished. Spot crops and labels are available in %s",
        args.output_root,
    )


if __name__ == "__main__":
    main()
