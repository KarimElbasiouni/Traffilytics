"""DRIFT OBB annotation adapter for YOLO training datasets.

DRIFT labels use Ultralytics OBB corner format per line:
  class_id x1 y1 x2 y2 x3 y3 x4 y4
(normalized coordinates). Images/labels live under train|valid|test splits.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

CLASS_NAMES = {0: "bus", 1: "car", 2: "truck"}


@dataclass(frozen=True)
class OBBBox:
    """One oriented bounding box: vehicle class plus eight normalized corner coords."""

    class_id: int
    corners: tuple[float, float, float, float, float, float, float, float]  # x1..y4

    @property
    def class_name(self) -> str:
        """Human-readable class label (bus / car / truck), or the raw id if unknown."""
        return CLASS_NAMES.get(self.class_id, str(self.class_id))


def parse_obb_label_line(line: str) -> OBBBox:
    """Parse one YOLO-OBB label line into an OBBBox (class + 8 corner values)."""
    parts = line.strip().split()
    if len(parts) != 9:
        raise ValueError(f"Expected 9 OBB fields (class + 8 coords), got {len(parts)}: {line!r}")
    class_id = int(float(parts[0]))
    coords = tuple(float(x) for x in parts[1:])
    return OBBBox(class_id=class_id, corners=coords)  # type: ignore[arg-type]


def load_obb_label_file(path: Path) -> list[OBBBox]:
    """Read every non-empty line of a ``.txt`` label file into a list of OBBBox objects."""
    if not path.is_file():
        raise FileNotFoundError(path)
    boxes: list[OBBBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        boxes.append(parse_obb_label_line(line))
    return boxes


def iter_split_pairs(split_dir: Path) -> Iterator[tuple[Path, Path]]:
    """Yield (image_path, label_path) pairs for a YOLO split directory.

    Expects either:
      split_dir/images/*.jpg + split_dir/labels/*.txt
    or images/labels as siblings under a parent with named splits.
    """
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels dir: {labels_dir}")

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            yield image_path, label_path


@dataclass
class DriftOBBDataset:
    """Adapter over a local DRIFT-style YOLO OBB directory tree."""

    root: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path | None = None

    @classmethod
    def from_annotations_root(cls, annotations_root: str | Path) -> DriftOBBDataset:
        """Locate train/valid (or val) and optional test folders under an annotations root."""
        root = Path(annotations_root).resolve()
        train = root / "train"
        val = root / "valid"
        if not val.is_dir():
            val = root / "val"
        test = root / "test" if (root / "test").is_dir() else None
        if not train.is_dir():
            raise FileNotFoundError(f"Expected train/ under {root}")
        if not val.is_dir():
            raise FileNotFoundError(f"Expected valid/ or val/ under {root}")
        return cls(root=root, train_dir=train, val_dir=val, test_dir=test)

    def count_pairs(self, split: str = "train") -> int:
        """Count how many image/label pairs exist for the given split name."""
        return sum(1 for _ in self.iter_pairs(split))

    def iter_pairs(self, split: str = "train") -> Iterator[tuple[Path, Path]]:
        """Iterate image/label pairs for ``train``, ``val``/``valid``, or ``test``."""
        directory = {
            "train": self.train_dir,
            "val": self.val_dir,
            "valid": self.val_dir,
            "test": self.test_dir,
        }.get(split)
        if directory is None:
            raise ValueError(f"Unknown or missing split: {split}")
        yield from iter_split_pairs(directory)

    def write_ultralytics_data_yaml(
        self,
        out_path: str | Path,
        *,
        names: dict[int, str] | None = None,
    ) -> Path:
        """Write a YOLO data.yaml pointing at this dataset's absolute paths."""
        names = names or CLASS_NAMES
        payload = {
            "path": str(self.root),
            "train": "train/images",
            "val": "valid/images" if (self.root / "valid").is_dir() else "val/images",
            "names": {int(k): v for k, v in names.items()},
        }
        if self.test_dir is not None:
            payload["test"] = "test/images"
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)
        return out


def sync_github_style_layout(src_annotations: Path, dest: Path) -> Path:
    """Copy a DRIFT GitHub-style model/{train,valid,test} tree into dest."""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        src_split = src_annotations / split
        if src_split.is_dir():
            shutil.copytree(src_split, dest / split, dirs_exist_ok=True)
    src_yaml = src_annotations / "data.yaml"
    if src_yaml.is_file():
        shutil.copy2(src_yaml, dest / "data.yaml")
    return dest
