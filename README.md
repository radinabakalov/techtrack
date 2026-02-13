# TechTrack: AI-Enabled Object Detection System

This project implements an end-to-end object detection pipeline for the TechTrack assignment using a pre-trained YOLO model. The system handles video input, runs inference, post-processes detections, and evaluates model performance.

In addition to the core detection system, this repository includes a full case study (CASE_ANALYSIS.md) analyzing model selection, dataset sampling, NMS configuration, augmentation robustness, and hard negative mining strategies.

The code is organized into separate modules for inference, preprocessing, and evaluation to keep things clean and testable.

---

## Project Structure

```
├── README.md                       # Project overview and usage instructions
├── CASE_ANALYSIS.md                # Final case study report
├── ANALYSIS_DESIGN_ASSIGNMENT.md
├── IMPLEMENTATION_ASSIGNMENT.md
├── Dockerfile
├── requirements.txt
├── setup.sh
├── .gitignore
│
├── techtrack/                      # Core object detection system
│   ├── modules/
│   │   ├── inference/              # Model loading, prediction, NMS, preprocessing
│   │   │   ├── model.py
│   │   │   ├── nms.py
│   │   │   └── preprocessing.py
│   │   ├── rectification/          # Loss computation and hard negative mining
│   │   │   ├── augmentation.py
│   │   │   └── hard_negative_mining.py
│   │   └── utils/                  # Metrics and loss utilities
│   │       ├── loss.py
│   │       └── metrics.py
│   ├── storage/                    # Model weights, configs, dataset
│   │   ├── yolo_model_1/
│   │   ├── yolo_model_2/
│   │   ├── logistics/
│   │   └── test_videos/
│   └── app.py                      # Main entry point
│
├── analysis/                       # Case study analysis artifacts
│   ├── figures/                    # All figures used in CASE_ANALYSIS.md
│   ├── cache/                      # Cached model outputs and loss data (ignored if large)
│   ├── sample_images.txt           # Fixed 6,000-image subset
│   ├── task1.ipynb
│   ├── task2.ipynb
│   ├── task3.ipynb
│   ├── task4.ipynb
│   └── task5.ipynb
│
└── test/                           # Unit tests
```

---

## Setup

### 1. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application

Run object detection on video input:
```bash
python techtrack/app.py
```

The app loads the YOLO model from `storage/`, processes the video frames, and outputs detections.

---

## Docker Usage (Optional)

You can also build and run the project using Docker.

### Build the image
```bash
docker build -t techtrack .
```

### Run the container
```bash
docker run --rm techtrack
```

---

## Testing

All unit tests are located in the `test/` directory.
To run all the tests locally:
```bash
python -m unittest discover -s test -p "unit_test*.py"
```

These tests validate:
- Model predictions 
- Post-processing and thresholding
- Loss and metrics 
- Hard negative mining 
- End-to-end detector integration

---

## Notes

- YOLO config (`.cfg`), weights, and class names (`.names`) must be in `storage/` for the detector to work.
- Code prioritizes readability and modularity over performance optimization.

---
