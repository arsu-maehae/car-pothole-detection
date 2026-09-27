# Low-Light Robust Road Hazard Detection

This folder contains the datasets, source code, trained models, evaluation results, figures, demonstration video, logs, and related papers for the Low-Light Robust Road Hazard Detection project.

## Project objective

The project studies road-hazard detection under low-light conditions. It evaluates whether results obtained from synthetically darkened daylight images transfer to real dusk and night images.

The detector recognizes three classes:

```text
0 = Drain Hole
1 = Pothole
2 = Sewer Cover
```

## Folder structure

```text
Image_Project/
├── Code/
│   ├── Person B/
│   └── Person C/
├── Dataset/                 # Shared dataset shortcut
├── Test set/
│   ├── T1/
│   ├── T2/
│   ├── T3/                  # Shortcut to mapped real dusk/night data
│   └── CLAHE/
├── Model/
│   ├── E0_baseline/
│   └── E3_adverse/
├── output/
├── logs/
└── Related Papers/
```

## Experiments

| Experiment | Model and processing |
|---|---|
| E0 | Baseline detector |
| E1 | E0 with CLAHE during inference |
| E3 | Detector trained with adverse-condition augmentation |
| E4 | E3 with CLAHE during inference |

E1 and E4 apply CLAHE during inference and do not require new model training.

## Training dataset

The Roboflow Pothole Detection v9 dataset was used for training and validation.

| Split | Images |
|---|---:|
| Train | 3,033 |
| Validation | 491 |
| Test | 246 |

## Backbone comparison

Person B trained YOLOv8n, YOLO11n, and YOLO12n for 40 epochs each under comparable settings.

| Run | Backbone | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---:|---:|---:|---:|
| V0 | YOLOv8n | 0.873 | 0.856 | 0.908 | 0.575 |
| V1 | YOLO11n | **0.924** | 0.847 | 0.904 | **0.585** |
| V2 | YOLO12n | 0.888 | **0.881** | **0.910** | 0.573 |

YOLO11n was selected for E0 and E3 because it achieved the highest precision and mAP@0.5:0.95 while using 6.4 GFLOPs and the shortest recorded training time.

## Test sets

| Test set | Description | Size |
|---|---|---:|
| T1 | Original Roboflow daylight test split | 246 images |
| T2 | T1 darkened at gamma 1.5, 2.0, and 2.5 | 738 images |
| T3 | Real Evening and Night images from the Bučko dataset | 560 images |

T3 contains 560 matching label files, 234 valid empty labels, and 601 annotated objects. Its verified mapping is:

```text
Bučko class 0 Pothole  -> target class 1 Pothole
Bučko class 1 Manhole -> target class 2 Sewer Cover
```

T3 is used for evaluation only.

## Adverse-condition training

E3 uses the 3,033 original Roboflow training images plus one generated adverse copy of each image, giving 6,066 training images. The offline augmentation pipeline includes rain, fog, motion blur, brightness and contrast changes, and ISO noise. Validation and test images remain unchanged.

## Main evaluation results

The `output/table2_main_results.csv` file records the following mAP@0.5 results:

| Experiment | T1 daylight | T3 at 640 | T3 at 1280 | T3 at 1920 |
|---|---:|---:|---:|---:|
| E0 | 0.7471 | 0.0092 | 0.0158 | 0.0147 |
| E1 | 0.6619 | 0.0223 | 0.0269 | 0.0258 |
| E3 | 0.7350 | 0.0025 | 0.0131 | — |
| E4 | 0.7309 | 0.0083 | 0.0161 | — |

The value 0.7471 is the E0 score on the T1 daylight test set. It is different from E0's training validation result.

For T2, CLAHE had little effect at gamma 1.5, improved mAP@0.5 from 0.6078 to 0.6713 at gamma 2.0, and improved it from 0.3303 to 0.6263 at gamma 2.5.

Real-night performance is also affected by object scale. At an inference size of 640, the median T3 object area presented to the detector is 302.5 px², compared with 12,604 px² for T1.

## Code

### Person B

| File | Purpose |
|---|---|
| `compare_backbones.py` | Trains YOLOv8n, YOLO11n, and YOLO12n for 40 epochs each. |
| `build_e3_dataset.py` | Builds the E3 dataset from original and adverse training images. |
| `train_e3.py` | Trains the E3 YOLO11n model for 100 epochs. |
| `build_t3_mapped.py` | Builds T3 and maps the Bučko classes to the project schema. |
| `check_bucko_labels.py` | Checks Bučko label structure and class counts. |
| `check_bucko_boxes.py` | Renders boxes for visual verification. |
| `check_roboflow_cover_classes.py` | Verifies the Roboflow target classes. |

### Person C

Person C's folder contains the T2 synthetic-dark and CLAHE preprocessing work.

## Models

- `Model/E0_baseline/` contains the baseline training artifacts and weights.
- `Model/E3_adverse/` contains the adverse-trained model artifacts and weights.
- Use `weights/best.pt` for evaluation.
- `args.yaml`, `results.csv`, plots, confusion matrices, and validation previews document each run.

## Outputs

The `output` folder contains:

- Main and partial evaluation result tables
- T2 severity results
- Transfer-ratio summaries
- Resolution and object-size analyses
- Figures used in the report and presentation
- `demo_side_by_side.mp4`

## Environment

The Person B training environment was:

```text
Windows
NVIDIA GeForce RTX 4060 Laptop GPU
Python 3.13.0
PyTorch 2.11.0+cu128
CUDA runtime 12.8
Ultralytics 8.4.153
OpenCV 5.0.0
Albumentations 2.0.8
Seed 42
```

## Recommended workflow

1. Obtain the Roboflow v9 dataset through the `Dataset` shortcut.
2. Use `Code/Person B/compare_backbones.py` to reproduce the 40-epoch backbone comparison.
3. Use the E0 artifacts in `Model/E0_baseline` for the baseline detector.
4. Run `build_e3_dataset.py`, then `train_e3.py`, to reproduce E3.
5. Use T1, T2, and T3 under `Test set` for evaluation.
6. Apply CLAHE to obtain E1 and E4 without retraining.
7. Use the files in `output` for the final tables, figures, and demonstration.

## Notes

- Preserve the class order: Drain Hole, Pothole, Sewer Cover.
- Preserve T3 empty-label files because they are valid negative examples.
- Do not use T1, T2, or T3 for model training.
- Local absolute paths in YAML files should be changed to portable relative paths before another person runs the code.
- The log in `logs/log_B_2026-09-15.md` documents the T3 construction session; later installed software versions are listed in this README.
