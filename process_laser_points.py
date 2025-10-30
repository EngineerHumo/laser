import json
import logging
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except ImportError:  # pragma: no cover - Pillow is optional.
    Image = ImageDraw = ImageFont = None  # type: ignore

LOGGER = logging.getLogger(__name__)

Rectangle = Tuple[float, float, float, float, str]


@dataclass
class SimpleBmpImage:
    """Minimal BMP reader/writer used when Pillow is not available."""

    path: Path
    header: bytes
    width: int
    height: int
    bytes_per_pixel: int
    row_stride: int
    top_down: bool
    pixels: bytearray

    @classmethod
    def open(cls, path: Path) -> "SimpleBmpImage":
        with path.open("rb") as fh:
            header = fh.read(14)
            if len(header) != 14 or header[0:2] != b"BM":
                raise ValueError(f"Unsupported BMP header in {path}")

            file_size, _, _, pixel_array_offset = struct.unpack("<IHHI", header[2:14])

            dib_header_size_data = fh.read(4)
            if len(dib_header_size_data) != 4:
                raise ValueError(f"Incomplete DIB header in {path}")
            (dib_header_size,) = struct.unpack("<I", dib_header_size_data)
            dib_header_rest = fh.read(dib_header_size - 4)
            if len(dib_header_rest) != dib_header_size - 4:
                raise ValueError(f"Incomplete DIB header in {path}")

            dib_header = dib_header_size_data + dib_header_rest
            header += dib_header

            width, height, planes, bits_per_pixel = struct.unpack("<iiHH", dib_header[4:16])
            if planes != 1:
                raise ValueError("Unsupported BMP file: planes must equal 1")
            compression = struct.unpack("<I", dib_header[16:20])[0]
            if compression != 0:
                raise ValueError("Only uncompressed BMP files are supported")

            if bits_per_pixel not in (24, 32):
                raise ValueError("Only 24-bit and 32-bit BMP files are supported")

            bytes_per_pixel = bits_per_pixel // 8
            row_stride = ((bits_per_pixel * width + 31) // 32) * 4
            abs_height = abs(height)
            top_down = height < 0

            # Read any remaining header bytes (colour tables, etc.)
            header_size = 14 + dib_header_size
            if pixel_array_offset < header_size:
                raise ValueError("Invalid pixel offset in BMP file")
            header += fh.read(pixel_array_offset - header_size)

            pixels = bytearray(width * abs_height * bytes_per_pixel)
            for row_index in range(abs_height):
                row_data = fh.read(row_stride)
                if len(row_data) != row_stride:
                    raise ValueError("Incomplete pixel data in BMP file")
                effective_row = row_index if top_down else abs_height - 1 - row_index
                start = effective_row * width * bytes_per_pixel
                pixels[start : start + width * bytes_per_pixel] = row_data[: width * bytes_per_pixel]

        return cls(
            path=path,
            header=header,
            width=width,
            height=abs_height,
            bytes_per_pixel=bytes_per_pixel,
            row_stride=row_stride,
            top_down=top_down,
            pixels=pixels,
        )

    def _set_pixel(self, x: int, y: int, color: Tuple[int, int, int]) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        index = (y * self.width + x) * self.bytes_per_pixel
        bgr = color[2], color[1], color[0]
        self.pixels[index : index + 3] = bytes(bgr)
        if self.bytes_per_pixel == 4:
            # Preserve alpha channel (set to fully opaque)
            self.pixels[index + 3] = 255

    def draw_filled_circle(self, cx: float, cy: float, radius: int, color: Tuple[int, int, int]) -> None:
        if radius <= 0:
            return
        x0 = max(0, int(math.floor(cx - radius)))
        x1 = min(self.width - 1, int(math.ceil(cx + radius)))
        y0 = max(0, int(math.floor(cy - radius)))
        y1 = min(self.height - 1, int(math.ceil(cy + radius)))
        radius_sq = radius * radius
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius_sq:
                    self._set_pixel(x, y, color)

    def draw_rectangle(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: Tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        if thickness <= 0:
            return
        x_start = max(0, int(math.floor(min(x0, x1))))
        x_end = min(self.width - 1, int(math.ceil(max(x0, x1))))
        y_start = max(0, int(math.floor(min(y0, y1))))
        y_end = min(self.height - 1, int(math.ceil(max(y0, y1))))
        if x_end < x_start or y_end < y_start:
            return
        thickness = int(thickness)
        for offset in range(thickness):
            top = y_start + offset
            bottom = y_end - offset
            if top > bottom:
                break
            for x in range(x_start, x_end + 1):
                self._set_pixel(x, top, color)
                self._set_pixel(x, bottom, color)
            left = x_start + offset
            right = x_end - offset
            if left > right:
                break
            for y in range(top, bottom + 1):
                self._set_pixel(left, y, color)
                self._set_pixel(right, y, color)

    def save(self, output_path: Path) -> None:
        with output_path.open("wb") as fh:
            fh.write(self.header)
            for row_index in range(self.height):
                effective_row = row_index if self.top_down else self.height - 1 - row_index
                start = effective_row * self.width * self.bytes_per_pixel
                row = self.pixels[start : start + self.width * self.bytes_per_pixel]
                fh.write(row)
                padding = self.row_stride - len(row)
                if padding:
                    fh.write(b"\x00" * padding)


def iter_json_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Data directory {root} does not exist")
    yield from sorted(path for path in root.glob("*.json") if path.is_file())


def load_rectangles(annotation: dict) -> List[Rectangle]:
    points = annotation.get("points") or []
    if not points:
        return []
    xs, ys = zip(*points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    label = str(annotation.get("label", ""))
    return [(min_x, min_y, max_x, max_y, label)]


def draw_with_pillow(image_path: Path, rectangles: Iterable[Rectangle], output_path: Path) -> None:
    if Image is None:
        raise RuntimeError("Pillow backend is not available")
    with Image.open(image_path) as image:  # type: ignore[call-arg]
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        thickness = max(1, round(min(image.size) * 0.005))
        for x0, y0, x1, y1, label in rectangles:
            bbox = (x0, y0, x1, y1)
            draw.rectangle(bbox, outline="red", width=thickness)
            if label:
                draw.text((x0, max(y0 - thickness * 2, 0)), label, fill="yellow", font=font)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)


def draw_with_builtin(image_path: Path, rectangles: Iterable[Rectangle], output_path: Path) -> None:
    bmp = SimpleBmpImage.open(image_path)
    thickness = max(1, round(min(bmp.width, bmp.height) * 0.005))
    for x0, y0, x1, y1, _label in rectangles:
        bmp.draw_rectangle(x0, y0, x1, y1, (255, 0, 0), thickness)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bmp.save(output_path)


def process_file(json_path: Path, image_path: Path, output_dir: Path) -> Optional[Path]:
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    shapes = data.get("shapes") or []
    rectangles: List[Rectangle] = []
    for shape in shapes:
        rectangles.extend(load_rectangles(shape))

    if not image_path.exists():
        raise FileNotFoundError(f"Missing image file for {json_path.name}: {image_path}")

    output_filename = f"{image_path.stem}_masked{image_path.suffix}"
    output_path = output_dir / output_filename
    try:
        if Image is not None:
            draw_with_pillow(image_path, rectangles, output_path)
        else:
            draw_with_builtin(image_path, rectangles, output_path)
    except Exception as exc:  # pragma: no cover - defensive programming
        LOGGER.error("Failed to process %s: %s", json_path.name, exc)
        return None
    return output_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data_dir = Path("data")
    output_dir = data_dir / "output"

    for json_file in iter_json_files(data_dir):
        image_file = json_file.with_suffix(".bmp")
        try:
            output_path = process_file(json_file, image_file, output_dir)
        except FileNotFoundError as exc:
            LOGGER.warning(str(exc))
        else:
            if output_path:
                LOGGER.info("Saved annotated image to %s", output_path)


if __name__ == "__main__":
    main()
