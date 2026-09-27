"""Train E3 from fresh YOLO11n pretrained weights.

Place this file in the project's ``src`` directory, activate ``.venv``, and run:
    python src/train_e3.py
"""

from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "datasets" / "E3_adverse" / "E3_data.yaml"
STARTING_WEIGHTS = PROJECT_ROOT / "yolo11n.pt"
RUNS_ROOT = PROJECT_ROOT / "runs" / "experiments"


def main() -> None:
    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Missing {DATA_YAML}. Run src/build_e3_dataset.py first."
        )
    if not STARTING_WEIGHTS.is_file():
        raise FileNotFoundError(f"Missing starting weights: {STARTING_WEIGHTS}")

    # Start fresh from yolo11n.pt, not from E0 best.pt. This makes E0 and E3
    # comparable: architecture, seed, optimizer, and training length are equal.
    model = YOLO(str(STARTING_WEIGHTS))
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=16,
        optimizer="Adam",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        patience=25,
        seed=42,
        deterministic=True,
        device=0,
        workers=0,
        cache=False,
        amp=True,
        # E3 online augmentation settings from the project methodology.
        hsv_h=0.015,
        hsv_s=0.70,
        hsv_v=0.70,
        degrees=10.0,
        scale=0.50,
        fliplr=0.50,
        mosaic=1.0,
        project=str(RUNS_ROOT),
        name="E3_adverse",
        exist_ok=False,
        save=True,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
