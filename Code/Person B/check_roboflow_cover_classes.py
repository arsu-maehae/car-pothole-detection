from pathlib import Path
import cv2

PROJECT = Path(
    r"E:\works\cpre\term7\image and processing\project"
)

DATASET = PROJECT / "datasets" / "Pothole Detection.v9i.yolov8"
IMAGE_DIR = DATASET / "train" / "images"
LABEL_DIR = DATASET / "train" / "labels"
OUTPUT_DIR = PROJECT / "verification" / "roboflow_cover_classes"

CLASS_NAMES = {
    0: "Drain Hole",
    1: "Pothole",
    2: "Sewer Cover",
}

COLORS = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 165, 255),
}

TARGET_CLASSES = (0, 2)
SAMPLES_PER_CLASS = 5
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_image(stem):
    for extension in IMAGE_EXTENSIONS:
        for candidate in (
            IMAGE_DIR / f"{stem}{extension}",
            IMAGE_DIR / f"{stem}{extension.upper()}",
        ):
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"No image for label: {stem}")


def read_boxes(label_path):
    boxes = []

    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError as error:
            raise ValueError(
                f"Invalid value in {label_path}, line {line_number}: {line}"
            ) from error

        # Standard YOLO detection:
        # class_id x_center y_center width height
        if len(parts) == 5:
            xc, yc, width, height = values

        # YOLO segmentation:
        # class_id x1 y1 x2 y2 x3 y3 ...
        elif len(values) >= 6 and len(values) % 2 == 0:
            xs = values[0::2]
            ys = values[1::2]

            x_min = min(xs)
            x_max = max(xs)
            y_min = min(ys)
            y_max = max(ys)

            xc = (x_min + x_max) / 2
            yc = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

        else:
            raise ValueError(
                f"Malformed label in {label_path}, "
                f"line {line_number}: {line}"
            )

        boxes.append((class_id, xc, yc, width, height))

    return boxes


selected = []
used = set()

for target_class in TARGET_CLASSES:
    found = 0

    for label_path in sorted(LABEL_DIR.glob("*.txt")):
        if label_path in used:
            continue

        boxes = read_boxes(label_path)

        if not any(box[0] == target_class for box in boxes):
            continue

        selected.append((target_class, label_path, boxes))
        used.add(label_path)
        found += 1

        if found == SAMPLES_PER_CLASS:
            break

    if found < SAMPLES_PER_CLASS:
        raise RuntimeError(
            f"Only found {found} samples for class {target_class}"
        )

for target_class, label_path, boxes in selected:
    image_path = find_image(label_path.stem)
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Cannot open {image_path}")

    height, width = image.shape[:2]

    for class_id, xc, yc, bw, bh in boxes:
        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)

        color = COLORS[class_id]
        name = CLASS_NAMES[class_id]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            image,
            f"{class_id}: {name}",
            (max(0, x1), max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    output = OUTPUT_DIR / (
        f"target_{target_class}_{label_path.stem}.jpg"
    )
    cv2.imwrite(str(output), image)
    print("Saved:", output)

print(f"\nCreated {len(selected)} verification images.")