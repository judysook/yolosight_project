# YOLOSight: Real-Time Webcam Object Detection with TTS

> A Python-based real-time object detection system using YOLOv5 and Text-to-Speech (TTS), showcasing incremental feature development through multiple script versions.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Project Structure](#project-structure)
5. [Script Versions & Mapping](#script-versions--mapping)
6. [Configuration](#configuration)
7. [Contributing](#contributing)
8. [License](#license)
9. [Authors & References](#authors--references)

---

## Overview

YOLOSight provides a step-by-step evolution of a real-time object detection pipeline using YOLOv5 on webcam feeds, enhanced with features like capture, recording, overlap-free labels, and TTS for accessibility.

## Features

* **Incremental Development**: Eight detection scripts (`webcam_detect_1.py` through `webcam_detect_8.py`), each adding new capabilities.
* **Model Switching**: Easy swapping of different `.pt` model weights.
* **Direction Detection**: Classify objects as left / front / right based on bounding-box centers.
* **Crosswalk & Road Logic**: Detect and interpret ‘crosswalk’ + ‘road’ combinations as vehicle zones.
* **Capture & Record**:

  * Press `s` to save snapshots.
  * Press `c` to start/stop video recording.
* **Label Overlap Prevention**: Automatic vertical spacing when labels are too close.
* **TTS Integration**: Zero-shot TTS and MeloTTS playback for audio alerts.

## Prerequisites

* Python 3.7+ (recommended 3.8)
* [PyTorch](https://pytorch.org/) with CPU support
* OpenCV (`opencv-python`)
* NumPy, PyYAML, other dependencies listed in `requirements.txt`

## Project Structure

```
webcam/
├── models/               # YOLOv5 model weights
│   ├── best_cpu_win_123.pt
│   ├── best_cpu_win_45.pt
│   └── best_cpu_win_final.pt
├── scripts/              # All detection scripts
│   ├── webcam_pre_yolo.py
│   ├── webcam_detect_1.py
│   ├── webcam_detect_2.py
│   ├── webcam_detect_3.py
│   ├── webcam_detect_4.py
│   ├── webcam_detect_5.py
│   ├── webcam_detect_6.py
│   ├── webcam_detect_7.py
│   └── webcam_detect_8.py      # Alias: webcam_detect_final.py
├── detections/           # Saved snapshots & videos per script
│   ├── detect_1/
│   ├── detect_2/
│   └── detect_3/
│   └── detect_4/
│   └── detect_5/
├── README.md             # Project overview & instructions
└── requirements.txt      # Python dependencies
```

## Script Versions & Mapping

| Script               | Model File              | Key Enhancements                         |
| -------------------- | ----------------------- | ---------------------------------------- |
| `webcam_pre_yolo.py` | *—*                     | Initial YOLO inference test              |
| `webcam_detect_1.py` | `best_cpu_win_123.pt`   | Basic detection (bbox + labels)          |
| `webcam_detect_2.py` | `best_cpu_win_123.pt`   | Class-based random colors + direction    |
| `webcam_detect_3.py` | `best_cpu_win_123.pt`   | Overlap-free label placement             |
| `webcam_detect_4.py` | `best_cpu_win_45.pt`    | Model & PosixPath fixes                  |
| `webcam_detect_5.py` | `best_cpu_win_45.pt`    | Video recording toggle (`c` key)         |
| `webcam_detect_6.py` | `best_cpu_win_45.pt`    | Zero-shot TTS import                     |
| `webcam_detect_7.py` | `best_cpu_win_45.pt`    | MeloTTS integration & playback fixes     |
| `webcam_detect_8.py` | `best_cpu_win_final.pt` | Final stable version w/ full TTS support |

## Configuration

You can customize:

* `--conf-thres`: Detection confidence threshold
* `--iou-thres`: NMS IoU threshold
* `--half`: Use FP16 inference (if GPU available)
* YOLOv5 flags (refer to [YOLOv5 docs](https://github.com/ultralytics/yolov5))

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please adhere to the existing code style and include tests when appropriate.

## License

This project is licensed under the [MIT License](LICENSE).

## Authors & References

* **Seojin031** – Initial development & documentation
* Inspired by:

  * [RealTime\_zeroshot\_TTS\_ko](https://github.com/Nyan-SouthKorea/RealTime_zeroshot_TTS_ko)
  * [OpenVoice](https://github.com/myshell-ai/OpenVoice)

For detailed discussions and design rationales, see our project meeting notes and issues tracker.




