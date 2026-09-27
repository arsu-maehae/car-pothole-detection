from collections import Counter
from pathlib import Path
import shutil

PROJECT = Path(
    r"E:\works\cpre\term7\image and processing\project"
)

SOURCE_ROOT = PROJECT / "datasets" / "Potholes_dataset"
OUTPUT_ROOT = PROJECT / "datasets" / "T3_mapped"

SOURCE_FOLDERS = ("Evening", "Night")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Bučko -> Roboflow v9
# 0 Pothole -> 1 Pothole
# 1 Manhole -> 2 Sewer Cover (taxonomy harmonisation)
CLASS_MAP = {
    0: 1,
    1: 2,
}

output_images = OUTPUT_ROOT / "images"
output_labels = OUTPUT_ROOT / "labels"

output_images.mkdir(parents=True, exist_ok=True)
output_labels.mkdir(parents=True, exist_ok=True)

# Prevent accidental duplication or overwriting.
if any(output_images.iterdir()) or any(output_labels.iterdir()):
    raise RuntimeError(
        "T3_mapped is not empty. Stop and inspect it before running again."
    )

image_count = 0
label_count = 0
box_count = 0
empty_labels = 0
mapped_counts = Counter()

for condition in SOURCE_FOLDERS:
    source_folder = SOURCE_ROOT / condition

    image_by_stem = {}

    for image_path in source_folder.iterdir():
        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            if image_path.stem in image_by_stem:
                raise RuntimeError(
                    f"Duplicate image stem: {condition}/{image_path.stem}"
                )

            image_by_stem[image_path.stem] = image_path

    for label_path in sorted(source_folder.glob("*.txt")):
        if label_path.stem not in image_by_stem:
            raise RuntimeError(
                f"No matching image for {label_path}"
            )

        image_path = image_by_stem[label_path.stem]

        # Prefix condition to prevent duplicate filenames.
        output_stem = f"{condition}_{label_path.stem}"
        destination_image = (
            output_images / f"{output_stem}{image_path.suffix.lower()}"
        )
        destination_label = output_labels / f"{output_stem}.txt"

        output_lines = []

        for line_number, raw_line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"{label_path}:{line_number}: expected 5 fields"
                )

            original_class = int(parts[0])

            if original_class not in CLASS_MAP:
                raise ValueError(
                    f"{label_path}:{line_number}: "
                    f"unexpected class {original_class}"
                )

            coordinates = [float(value) for value in parts[1:]]

            if not all(0.0 <= value <= 1.0 for value in coordinates):
                raise ValueError(
                    f"{label_path}:{line_number}: "
                    "coordinate outside 0-1"
                )

            if coordinates[2] <= 0 or coordinates[3] <= 0:
                raise ValueError(
                    f"{label_path}:{line_number}: "
                    "width and height must be positive"
                )

            mapped_class = CLASS_MAP[original_class]

            # Coordinate text remains exactly unchanged.
            output_lines.append(
                f"{mapped_class} {' '.join(parts[1:])}"
            )

            mapped_counts[mapped_class] += 1
            box_count += 1

        if not output_lines:
            empty_labels += 1

        shutil.copy2(image_path, destination_image)

        label_text = "\n".join(output_lines)

        if label_text:
            label_text += "\n"

        destination_label.write_text(label_text, encoding="utf-8")

        image_count += 1
        label_count += 1

expected_counts = Counter({1: 506, 2: 95})

if image_count != 560:
    raise RuntimeError(f"Expected 560 images, found {image_count}")

if label_count != 560:
    raise RuntimeError(f"Expected 560 labels, found {label_count}")

if box_count != 601:
    raise RuntimeError(f"Expected 601 boxes, found {box_count}")

if mapped_counts != expected_counts:
    raise RuntimeError(
        f"Unexpected mapped counts: {dict(mapped_counts)}"
    )

yaml_path = OUTPUT_ROOT / "T3_data.yaml"

yaml_path.write_text(
    f'''path: "{OUTPUT_ROOT.as_posix()}"
train: images
val: images
test: images

nc: 3
names:
  0: Drain Hole
  1: Pothole
  2: Sewer Cover
''',
    encoding="utf-8",
)

print("=== T3 mapping completed ===")
print(f"Images       : {image_count}")
print(f"Labels       : {label_count}")
print(f"Empty labels : {empty_labels}")
print(f"Total boxes  : {box_count}")
print(f"Class 1 Pothole      : {mapped_counts[1]}")
print(f"Class 2 Sewer Cover  : {mapped_counts[2]}")
print(f"YAML          : {yaml_path}")