# RealTime\_zeroshot\_TTS\_ko

> A unified framework for real-time object detection (YOLOv5) and zero-shot/MeloTTS speech synthesis.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Inference Module (YOLOv5)](#inference-module-yolov5)
6. [TTS Modules](#tts-modules)
7. [Webcam Demo](#webcam-demo)
8. [Models](#models)
9. [Contributing](#contributing)
10. [License](#license)

---

## Overview

This repository showcases a modular pipeline that combines:

* **YOLOv5-based object detection** (inference/yolov5)
* **Zero-shot TTS** and **MeloTTS** voice synthesis (tts folder)
* **Webcam demo** integrating detection + audio alerts (webcam folder)

Each component can be used independently or together for rapid prototyping of assistive or vision-driven applications.

## Directory Structure

```
RealTime_zeroshot_TTS_ko/
├── inference/          # Embedded YOLOv5 code for offline inference
│   └── yolov5/         # YOLOv5 v6.x source & detect scripts
├── models/             # Pre-trained model weights (.pt files)
├── tts/                # Text-to-Speech modules
│   ├── RealTime_zeroshot_TTS_ko/  # Zero-shot TTS implementation
│   └── Melotts/        # MeloTTS API & models
├── webcam/             # Real-time webcam + TTS demo scripts
├── .gitattributes
├── .gitmodules
└── README.md           # ← You are here
```

## Prerequisites

* **Python 3.7+** (recommended 3.8)
* **Git** (for submodules)
* **PyTorch** (CPU or CUDA)
* **pip** or **conda** for dependency management

## Installation

1. **Clone the repository with submodules**

   ```bash
   git clone https://github.com/Nyan-SouthKorea/RealTime_zeroshot_TTS_ko.git
   cd RealTime_zeroshot_TTS_ko
   git submodule update --init --recursive
   ```

2. **(Optional) Create a virtual environment**

   ```bash
   conda create -n rt_tts python=3.8 -y
   conda activate rt_tts
   ```

3. **Install dependencies**

   * YOLOv5 inference:

     ```bash
     cd inference/yolov5
     pip install -r requirements.txt
     cd ../../
     ```
   * Webcam demo & core libs:

     ```bash
     pip install -r webcam/requirements.txt
     ```
   * Zero-shot TTS:

     ```bash
     cd tts/RealTime_zeroshot_TTS_ko
     pip install -r requirements.txt
     cd ../Melotts
     pip install -r requirements.txt
     cd ../../
     ```

4. **Place model weights** in the `models/` directory:

   * `best_cpu_win_123.pt`
   * `best_cpu_win_45.pt`
   * `best_cpu_win_final.pt`

## Inference Module (YOLOv5)

Use the built‑in `detect.py` script for offline image/video inference:

```bash
python inference/yolov5/detect.py \
  --weights ../../models/best_cpu_win_final.pt \
  --source path/to/images_or_video \
  --img 640 --conf 0.4 --save-txt --save-conf
```

Refer to `inference/yolov5/README.md` for advanced flags and options.

## TTS Modules

* **Zero-shot TTS** (`tts/RealTime_zeroshot_TTS_ko`): generate speech from arbitrary text without speaker‑specific training.
* **MeloTTS** (`tts/Melotts`): fine‑tuned neural TTS with reference speaker support.

Each subfolder contains its own README and requirement list. Follow those for initialization and sample usage.

## Webcam Demo

A live integration of detection + speech:

```bash
python webcam/webcam_detect_final.py --weights models/best_cpu_win_final.pt
```

**Controls during runtime:**

* `s`: save a screenshot to `screenshots/`
* `c`: toggle video recording to `recordings/`
* `q`: quit

## Models

All pre-trained weights are stored in `models/`. Rename or add new `.pt` files and pass them via the `--weights` flag in inference or demo scripts.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add new feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

Please adhere to existing style conventions and include tests or demos for new features.

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
