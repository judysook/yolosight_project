# YOLOSight: Real-Time Webcam Object Detection & TTS Demo

> A lightweight framework focused on real-time webcam-based object detection using YOLOv5, paired with zero-shot and MeloTTS speech synthesis for audio alerts.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [YOLOv5 Inference Module](#yolov5-inference-module)
6. [TTS Modules](#tts-modules)
7. [Webcam Demo](#webcam-demo)
8. [Model Weights](#model-weights)
9. [Contributing](#contributing)
10. [License](#license)

---

## Overview

**YOLOSight** is centered on a real-time **webcam demo** that combines:

* **YOLOv5** for live object detection
* **Zero-shot TTS** and **MeloTTS** for on-the-fly audio announcements

The modular design allows you to swap detection or speech components independently while keeping the webcam integration seamless.

## Directory Structure

```
YOLOSight/
├── inference/                 # YOLOv5 code & detect scripts
│   └── yolov5/
├── models/                    # Pre-trained 
│   └── best_cpu_win_final.pt
├── tts/                       # Text-to-Speech (Zero-shot & MeloTTS)
│   ├── RealTime_zeroshot_TTS_ko/
│   └── Melotts/
├── webcam/                    # Core webcam demo scripts & assets
│   ├── webcam_detect_final.py
│   └── requirements.txt      # Dependencies for webcam demo
├── .gitattributes
├── .gitmodules
└── README.md                  # ← this file
```

## Prerequisites

* **Python 3.7+** (recommend 3.8)
* **Git** (for submodules)
* **PyTorch** (CPU or CUDA)
* **pip** or **conda**

## Installation

1. **Clone repository with submodules**

   ```bash
   git clone https://github.com/Nyan-SouthKorea/RealTime_zeroshot_TTS_ko.git YOLOSight
   cd YOLOSight
   git submodule update --init --recursive
   ```
2. **(Optional) Create virtual environment**

   ```bash
   conda create -n yolosight python=3.8 -y
   conda activate yolosight
   ```
3. **Install dependencies**

   * YOLOv5 inference:

     ```bash
     cd inference/yolov5
     pip install -r requirements.txt
     cd ../../
     ```
   * Webcam demo:

     ```bash
     pip install -r webcam/requirements.txt
     ```
   * Zero-shot & MeloTTS:

     ```bash
     cd tts/RealTime_zeroshot_TTS_ko
     pip install -r requirements.txt
     cd ../Melotts
     pip install -r requirements.txt
     cd ../../
     ```

## YOLOv5 Inference Module

Run offline inference with:

```bash
python inference/yolov5/detect.py \
  --weights ../../models/best_cpu_win_final.pt \
  --source path/to/image_or_video \
  --img 640 --conf 0.4 --save-txt --save-conf
```

Refer to `inference/yolov5/README.md` for more options.

## TTS Modules

* **Zero-shot TTS** (`tts/RealTime_zeroshot_TTS_ko`)
* **MeloTTS** (`tts/Melotts`)

Each folder contains its own README and requirements. Follow those for setup.

## Webcam Demo

Detailed breakdown of the live **webcam** integration pipeline:

### 1. Environment Setup & Module Paths

* **Project root calculation** (`ROOT` variable)
* **Path compatibility patch** for Windows (`pathlib.PosixPath = pathlib.WindowsPath`)
* **Dynamic import** of YOLOv5 and TTS modules via `sys.path.insert`

### 2. Video Pre‑processing

* **Letterbox** resizing to 640×640 with padding
* **Channel & dimension** conversion: BGR→RGB, HWC→CHW, normalization

### 3. Object Detection & Post‑processing

* **Model loading** using `DetectMultiBackend`
* **Inference & NMS** (`non_max_suppression`)
* **Coordinate inverse mapping** (`scale_coords`)
* **Bounding‑box & label overlay**:

  * Random color per class
  * `cv2.rectangle` + `cv2.putText`
* **Overlap‑free labels**: Y‑axis offset collision avoidance
* **Spatial logic** (direction + crosswalk/road):

  * Center x-coordinate < 1/3 of frame width → “Left”; > 2/3 → “Right”; otherwise → “Front”
  * “crosswalk” detected only in the bottom 35% of the frame (y > 65% height) → “Crosswalk (road) detected”
  * “road” class → “Road detected"

### 4. Asynchronous TTS Integration

* **TTS initialization** (`Custom_TTS`, `MeloTTS`)
* **Output folder cleanup** per utterance
* **2‑second debounce** for speech notifications
* **Background thread** for non‑blocking audio playback

### 5. User Key Controls

* **Screenshot** (`s`): capture frame → `screenshots/<timestamp>.jpg`
* **Recording toggle** (`c`): start/stop MP4 → `recordings/<timestamp>.mp4`
* **Quit** (`q`): release capture & close window

## Model Weights

&#x20;

## Contributing

1. Fork the repo
2. Create branch: `git checkout -b feature/xyz`
3. Commit: `git commit -m "Add feature xyz"`
4. Push & PR: `git push origin feature/xyz`

Please adhere to style guides and include tests/demos.

## License

Licensed under MIT. See [LICENSE](LICENSE) for details.

