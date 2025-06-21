# YOLOSight: Real-Time Webcam Object Detection & TTS Demo

> A lightweight framework focused on real-time webcam-based object detection using YOLOv5, paired with zero-shot and MeloTTS speech synthesis for audio alerts.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [YOLOv5 Inference Module](#yolov5-inference-module)
6. [TTS Models and Experimental Framework](#TTS-Models-and-Experimental-Framework)
7. [TTS Modules](#tts-modules)
8. [Webcam Demo](#webcam-demo)
9. [Model Weights](#model-weights)
10. [Contributing](#contributing)
11. [License](#license)
12. [References](#references)

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

## TTS Models and Experimental Framework
This project incorporates experimental testing and comparative analysis using the following TTS models:

MeloTTS: Multi-lingual TTS library supporting CPU-based real-time inference, used for lightweight TTS experimentation
SCE-TTS Demo: Pre-trained multi-speaker Korean TTS model used for Korean speech synthesis comparison testing
Coqui TTS v0.11.0: High-quality TTS framework utilized for advanced model experimentation and comparative analysis


⚠️ Note: Experimental code and development iterations are maintained in separate branches for testing and future integration considerations.

Development Branches

judysook-tts(infer): SCE-TTS inference testing (training data stored in external drive)
tts-clean: Initial MeloTTS experimental implementation
tts-clean_06.12progress_MeloTTS: Development branch with Colab integration and error handling iterations

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

## References

* Inspired by:

  * [RealTime\_zeroshot\_TTS\_ko](https://github.com/Nyan-SouthKorea/RealTime_zeroshot_TTS_ko)
  * [OpenVoice](https://github.com/myshell-ai/OpenVoice)
  * [MeloTTS](https://github.com/myshell-ai/MeloTTS)
  * [SCE-TTS](https://sce-tts.github.io/#/v2/test) - Pre-trained multi-speaker Korean TTS
  * [Coqui TTS v0.11.0](https://github.com/coqui-ai/TTS/tree/v0.11.0) - High-quality TTS framework

### Contributors
- [Judysook] - TTS integration, model experimentation, and comparative analysis
- [seogin031]
- [snowball518]


### Acknowledgments
Special thanks to the open-source TTS community and the developers of MeloTTS, SCE-TTS, and Coqui TTS for their valuable contributions to the field.

For detailed discussions and design rationales, see our project meeting notes and issues tracker.


