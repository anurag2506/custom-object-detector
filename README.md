# Custom Faster R-CNN Object Detector

A **from-scratch** PyTorch implementation of Faster R-CNN for real-time street object detection.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

<p align="center">
  <img src="assets/detection_demo.gif" alt="Detection Demo" width="600"/>
</p>

---

## Highlights

- **Custom Implementation**: Faster R-CNN built from scratch (no torchvision detection models)
- **Pretrained Backbone**: ResNet-50 with ImageNet weights for better feature extraction
- **Mixed Precision Training**: 2-3x faster training with AMP
- **Real-time Inference**: ~22 FPS on T4 GPU

---

## Classes

| # | Class |
|---|-------|
| 1 | Person |
| 2 | Car |
| 3 | Bus |
| 4 | Bicycle |
| 5 | Motorbike |

---

## Results

| Metric | Value |
|--------|-------|
| **mAP@0.5** | 0.58 |
| **Inference Time** | 45ms |
| **FPS** | 22 |
| **Parameters** | 136M |

---

## Quick Start

### Installation

```bash
git clone https://github.com/anurag2506/custom-object-detector.git
cd custom-object-detector
pip install -r requirements.txt
```

### Inference

```bash
# Run on a single image
python inference.py --weights model/best.pth --source path/to/image.jpg --output tests/result --conf 0.3

# Run on video
python inference.py --weights model/best.pth --source video.mp4 --output tests/video_result --conf 0.3

# Benchmark inference speed
python inference.py --weights model/best.pth --benchmark
```

**Arguments:**
- `--weights`: Path to trained model weights (default: `model/best.pth`)
- `--source`: Input image or video path
- `--output`: Output path for results (without extension)
- `--conf`: Confidence threshold for detections (default: 0.5)

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Input     │ ──▶ │   ResNet-50  │ ──▶ │     RPN      │ ──▶ │   RoI Head   │
│  (416×416)   │     │   Backbone   │     │  (Proposals) │     │ (Detection)  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

**Components:**
- **Backbone**: ResNet-50 with ImageNet pretrained weights
- **RPN**: Region Proposal Network with 15 anchors per location
- **RoI Head**: RoI Align + FC layers for classification & regression

---

## Project Structure

```
custom-object-detector/
├── src/
│   ├── __init__.py
│   ├── backbone.py      # ResNet backbone with pretrained weights
│   ├── model.py         # Faster R-CNN main model
│   ├── rpn.py           # Region Proposal Network
│   ├── roi_head.py      # RoI Head for classification/regression
│   └── utils.py         # NMS, IoU, box utilities
├── model/
│   └── best.pth         # Trained model weights
├── tests/               # Test detection outputs
├── config.py            # Training configuration
├── dataset.py           # Data loading and augmentation
├── train.py             # Training script with validation
├── inference.py         # Inference and visualization
├── requirements.txt
└── README.md
```

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD (momentum=0.9) |
| Learning Rate | 0.005 |
| LR Schedule | MultiStep [16, 22] |
| Batch Size | 2 |
| Epochs | 15 |
| Image Size | 416×416 |

### Data Augmentation
- Horizontal Flip (50%)
- Random Scale (0.8-1.2×)
- Color Jitter (brightness, contrast, saturation, hue)
- Gaussian Blur (20%)
- Random Grayscale (10%)

---

## Test Output

<p align="center">
  <img src="tests/result_4.jpg" alt="Test Detection" width="600"/>
</p>

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497) - Ren et al.
- [Deep Residual Learning](https://arxiv.org/abs/1512.03385) - He et al.
