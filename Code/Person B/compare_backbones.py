from ultralytics import YOLO

models = {
    "V0_YOLOv8n": "yolov8n.pt",
    "V1_YOLO11n": "yolo11n.pt",
    "V2_YOLO12n": "yolo12n.pt",
}

for run_name, weights in models.items():
    model = YOLO(weights)

    model.train(
        data="datasets/Pothole Detection.v9i.yolov8/data.yaml",
        epochs=40,
        imgsz=640,
        batch=16,
        seed=42,
        device=0,
        project="runs/backbone_comparison",
        name=run_name,
    )