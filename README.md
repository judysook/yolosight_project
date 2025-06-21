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
├── inference/                # YOLOv5 code & detect scripts
│   └── yolov5/
├── models/                   # Pre-trained .pt weights
├── tts/                      # Text-to-Speech (Zero-shot & MeloTTS)
│   ├── RealTime_zeroshot_TTS_ko/
│   └── Melotts/
├── webcam/                   # Core webcam demo scripts & assets
│   └── requirements.txt      # Dependencies for webcam demo
├── .gitattributes
├── .gitmodules
└── README.md                 # ← this file
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
4. **Place model weights** into `models/`:

   * `best_cpu_win_123.pt`
   * `best_cpu_win_45.pt`
   * `best_cpu_win_final.pt`

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

Central focus: live webcam + detection + audio.

```bash
python webcam/webcam_detect_final.py --weights models/best_cpu_win_final.pt
```

**Runtime controls:**

* `s`: save screenshot to `screenshots/`
* `c`: toggle video recording to `recordings/`
* `q`: quit demo

## Model Weights

Store `.pt` files in `models/`. Use `--weights` flag to select:

* `best_cpu_win_123.pt`
* `best_cpu_win_45.pt`
* `best_cpu_win_final.pt`

## Contributing

1. Fork the repo
2. Create branch: `git checkout -b feature/xyz`
3. Commit: `git commit -m "Add feature xyz"`
4. Push & PR: `git push origin feature/xyz`

Please adhere to style guides and include tests/demos.

## License

Licensed under MIT. See [LICENSE](LICENSE) for details.
