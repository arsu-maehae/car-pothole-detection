from pathlib import Path
import cv2

PROJECT = Path(
    r"E:\works\cpre\term7\image and processing\project"
)

SOURCE_ROOT = PROJECT / "datasets" / "Potholes_dataset"
OUTPUT_ROOT = PROJECT / "verification" / "bucko_original_classes"

SOURCE_FOLDERS = ("Evening", "Night")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Expected from the observed 506/95 instance counts.
CLASS_NAMES = {
    0: "Expected Pothole",
    1: "Expected Manhole",
}

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 165, 255),
}

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def read_boxes(label_path):
    boxes = []

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

        class_id = int(parts[0])
        coordinates = [float(value) for value in parts[1:]]

        if class_id not in CLASS_NAMES:
            raise ValueError(
                f"{label_path}:{line_number}: "
                f"unexpected class {class_id}"
            )

        boxes.append((class_id, *coordinates))

    return boxes


def find_image(folder, stem):
    for extension in IMAGE_EXTENSIONS:
        candidate = folder / f"{stem}{extension}"

        if candidate.exists():
            return candidate

        candidate_upper = folder / f"{stem}{extension.upper()}"

        if candidate_upper.exists():
            return candidate_upper

    raise FileNotFoundError(f"No image found for {folder / stem}")


candidates = []

for condition in SOURCE_FOLDERS:
    folder = SOURCE_ROOT / condition

    for label_path in sorted(folder.glob("*.txt")):
        boxes = read_boxes(label_path)

        if boxes:
            candidates.append((condition, label_path, boxes))

selected = []
used_files = set()

# Select five different images containing each source class.
for required_class in (0, 1):
    selected_for_class = 0

    for condition, label_path, boxes in candidates:
        if label_path in used_files:
            continue

        if not any(box[0] == required_class for box in boxes):
            continue

        selected.append((condition, label_path, boxes))
        used_files.add(label_path)
        selected_for_class += 1

        if selected_for_class == 5:
            break

    if selected_for_class < 5:
        raise RuntimeError(
            f"Could not find 5 images for class {required_class}"
        )

for condition, label_path, boxes in selected:
    source_folder = SOURCE_ROOT / condition
    image_path = find_image(source_folder, label_path.stem)

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Cannot open {image_path}")

    height, width = image.shape[:2]

    for class_id, xc, yc, box_width, box_height in boxes:
        x1 = int((xc - box_width / 2) * width)
        y1 = int((yc - box_height / 2) * height)
        x2 = int((xc + box_width / 2) * width)
        y2 = int((yc + box_height / 2) * height)

        color = CLASS_COLORS[class_id]
        class_name = CLASS_NAMES[class_id]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        cv2.putText(
            image,
            f"Source {class_id}: {class_name}",
            (max(0, x1), max(35, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

    output_path = (
        OUTPUT_ROOT / f"{condition}_{label_path.stem}_CHECK.jpg"
    )

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not save {output_path}")

    print("Saved:", output_path)

print(f"\nVerification images created: {len(selected)}")
print("Inspect every box manually before accepting the mapping.")