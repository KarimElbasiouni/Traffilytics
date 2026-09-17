"""UAV-OBB dataset adapter: split discovery and Ultralytics ``data.yaml`` generation.

UAV-OBB ships a YOLO-style tree::

    UAV-OBB/
      train/{images,labels}
      valid/{images,labels}
      test/{images,labels}
      data.yaml

Label parsing itself lives in :mod:`computer_vision.detection.obb_labels`; this
module only understands the directory layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

from computer_vision.detection.types import CLASS_NAMES

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_split_pairs(split_dir: Path) -> Iterator[tuple[Path, Path]]:
    """Yield (image_path, label_path) pairs for a YOLO split directory.

    Expects ``split_dir/images/*.jpg`` alongside ``split_dir/labels/*.txt``.
    Images without a matching label file are skipped.
    """
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels dir: {labels_dir}")

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            yield image_path, label_path


@dataclass
class ObbDataset:
    """Adapter over a local UAV-OBB style YOLO OBB directory tree."""

    root: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path | None = None

    @classmethod
    def from_annotations_root(cls, annotations_root: str | Path) -> ObbDataset:
        """Locate train/valid (or val) and optional test folders under a dataset root."""
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
