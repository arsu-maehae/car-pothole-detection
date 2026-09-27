"""Build the E3 training dataset from the Roboflow daylight dataset.

The output contains:
  * every original training image/label;
  * one adverse-condition copy of every training image;
  * unchanged validation and test splits.

Place this file in the project's ``src`` directory, activate ``.venv``, and run:
    python src/build_e3_dataset.py
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "datasets" / "Pothole Detection.v9i.yolov8"
FINAL_ROOT = PROJECT_ROOT / "datasets" / "E3_adverse"
BUILD_ROOT = PROJECT_ROOT / "datasets" / "E3_adverse_building"
SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# Project methodology settings. These transforms change pixels only, so the
# YOLO bounding-box/polygon coordinates can be copied without modification.
ADVERSE = A.Compose(
    [
        A.RandomRain(p=0.15),
        A.RandomFog(fog_coef_range=(0.10, 0.40), p=0.15),
        A.MotionBlur(blur_limit=7, p=0.20),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.50, 0.10),
            contrast_limit=(-0.20, 0.20),
            p=0.30,
        ),
        A.ISONoise(p=0.15),
    ],
    seed=SEED,
    strict=True,
)

# Used only if all five probabilistic transforms are skipped repeatedly.
FALLBACK_DARKEN = A.Compose(
    [
        A.RandomBrightnessContrast(
            brightness_limit=(-0.50, -0.20),
            contrast_limit=(-0.15, 0.00),
            p=1.0,
        )
    ],
    seed=SEED,
    strict=True,
)


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def verify_pairs(split_root: Path) -> tuple[int, int]:
    images = image_files(split_root / "images")
    labels = sorted((split_root / "labels").glob("*.txt"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)
    if missing_labels or missing_images:
        raise RuntimeError(
            f"Unmatched files in {split_root}: "
            f"missing labels={missing_labels[:10]}, "
            f"missing images={missing_images[:10]}"
        )
    return len(images), len(labels)


def make_adverse_copy(image_bgr: np.ndarray) -> tuple[np.ndarray, int, bool]:
    """Return a changed image, retry count, and whether fallback was needed."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    for attempt in range(1, 21):
        changed_rgb = ADVERSE(image=image_rgb)["image"]
        if not np.array_equal(changed_rgb, image_rgb):
            return cv2.cvtColor(changed_rgb, cv2.COLOR_RGB2BGR), attempt, False

    changed_rgb = FALLBACK_DARKEN(image=image_rgb)["image"]
    return cv2.cvtColor(changed_rgb, cv2.COLOR_RGB2BGR), 20, True


def main() -> None:
    if FINAL_ROOT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing dataset: {FINAL_ROOT}\n"
            "Rename or remove it only if you intentionally want to rebuild E3."
        )
    if BUILD_ROOT.exists():
        raise FileExistsError(
            f"An incomplete build folder already exists: {BUILD_ROOT}\n"
            "Inspect it, then remove it before retrying."
        )

    source_train = SOURCE_ROOT / "train"
    source_train_count, _ = verify_pairs(source_train)
    if source_train_count == 0:
        raise RuntimeError(f"No training images found in {source_train / 'images'}")

    output_train_images = BUILD_ROOT / "train" / "images"
    output_train_labels = BUILD_ROOT / "train" / "labels"
    output_train_images.mkdir(parents=True)
    output_train_labels.mkdir(parents=True)

    retry_total = 0
    fallback_count = 0

    for index, image_path in enumerate(image_files(source_train / "images"), start=1):
        label_path = source_train / "labels" / f"{image_path.stem}.txt"

        # Keep the original example.
        shutil.copy2(image_path, output_train_images / image_path.name)
        shutil.copy2(label_path, output_train_labels / label_path.name)

        # Add one adverse-condition version; coordinates remain unchanged.
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV cannot read {image_path}")
        adverse_image, attempts, used_fallback = make_adverse_copy(image)
        retry_total += attempts - 1
        fallback_count += int(used_fallback)

        adverse_stem = f"{image_path.stem}__adv"
        adverse_image_path = output_train_images / f"{adverse_stem}{image_path.suffix.lower()}"
        adverse_label_path = output_train_labels / f"{adverse_stem}.txt"

        if not cv2.imwrite(str(adverse_image_path), adverse_image):
            raise RuntimeError(f"OpenCV cannot write {adverse_image_path}")
        shutil.copy2(label_path, adverse_label_path)

        if index % 250 == 0 or index == source_train_count:
            print(f"Augmented {index}/{source_train_count} training images")

    # Validation and test must stay clean and unchanged.
    for split in ("valid", "test"):
        source_split = SOURCE_ROOT / split
        verify_pairs(source_split)
        shutil.copytree(source_split, BUILD_ROOT / split)

    train_images, train_labels = verify_pairs(BUILD_ROOT / "train")
    valid_images, valid_labels = verify_pairs(BUILD_ROOT / "valid")
    test_images, test_labels = verify_pairs(BUILD_ROOT / "test")

    if train_images != source_train_count * 2 or train_labels != train_images:
        raise RuntimeError(
            f"Unexpected E3 training count: images={train_images}, labels={train_labels}, "
            f"expected={source_train_count * 2}"
        )

    yaml_text = f'''path: "{FINAL_ROOT.as_posix()}"
train: train/images
val: valid/images
test: test/images

nc: 3
names:
  0: Drain Hole
  1: Pothole
  2: Sewer Cover
'''
    (BUILD_ROOT / "E3_data.yaml").write_text(yaml_text, encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "source": str(SOURCE_ROOT),
        "policy": "original train + one adverse copy per train image; valid/test unchanged",
        "transforms": {
            "RandomRain": {"p": 0.15},
            "RandomFog": {"fog_coef_range": [0.10, 0.40], "p": 0.15},
            "MotionBlur": {"blur_limit": 7, "p": 0.20},
            "RandomBrightnessContrast": {
                "brightness_limit": [-0.50, 0.10],
                "contrast_limit": [-0.20, 0.20],
                "p": 0.30,
            },
            "ISONoise": {"p": 0.15},
        },
        "counts": {
            "train_images": train_images,
            "train_labels": train_labels,
            "valid_images": valid_images,
            "valid_labels": valid_labels,
            "test_images": test_images,
            "test_labels": test_labels,
        },
        "extra_retries_when_all_transforms_skipped": retry_total,
        "fallback_darkening_count": fallback_count,
    }
    (BUILD_ROOT / "augmentation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Rename only after the whole dataset passes validation.
    BUILD_ROOT.rename(FINAL_ROOT)

    print("\n=== E3 dataset completed ===")
    print(f"Train images/labels: {train_images}/{train_labels}")
    print(f"Valid images/labels: {valid_images}/{valid_labels}")
    print(f"Test images/labels : {test_images}/{test_labels}")
    print(f"YAML               : {FINAL_ROOT / 'E3_data.yaml'}")
    print(f"Manifest           : {FINAL_ROOT / 'augmentation_manifest.json'}")


if __name__ == "__main__":
    main()
