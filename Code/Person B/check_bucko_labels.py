from pathlib import Path
from collections import Counter

ROOT = Path(
    r"E:\works\cpre\term7\image and processing\project\datasets\Potholes_dataset"
)

folders = ["Evening", "Night"]

class_counts = Counter()
label_files = 0
empty_files = 0
total_boxes = 0
unexpected = []

for folder_name in folders:
    folder = ROOT / folder_name

    for txt in folder.glob("*.txt"):
        label_files += 1

        lines = [
            line.strip()
            for line in txt.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if not lines:
            empty_files += 1
            continue

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                unexpected.append(
                    (txt.name, f"Bad field count: {line}")
                )
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                unexpected.append(
                    (txt.name, f"Bad class ID: {line}")
                )
                continue

            class_counts[class_id] += 1
            total_boxes += 1

print("=== Bučko T3 inspection ===")
print("Label files :", label_files)
print("Empty labels:", empty_files)
print("Total boxes :", total_boxes)
print("Class counts:")

for class_id, count in sorted(class_counts.items()):
    print(f"  class {class_id}: {count}")

if unexpected:
    print("\nProblems:")
    for item in unexpected[:20]:
        print(item)
else:
    print("\nNo malformed YOLO labels found.")