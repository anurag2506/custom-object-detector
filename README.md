# Faster RCNN Object Detection

Object detection model trained from scratch to detect street objects.

## Classes
- Person
- Car
- Truck
- Bicycle
- Traffic light

## Setup

```bash
pip install -r requirements.txt
```

## Prepare Data

First prepare the dataset from archive folder:

```bash
python prepare_data.py
```

This creates train/val splits with annotations.

## Train

```bash
python train.py
```

Training runs for 24 epochs. Model saved to `output/best.pth`.

## Evaluate

```bash
python evaluate.py --checkpoint output/best.pth
```

Shows mAP, per-class AP, and FPS.

## Detect

Single image:
```bash
python detect.py --image test.jpg --output result.jpg
```

Webcam:
```bash
python detect.py --video 0
```

Video file:
```bash
python detect.py --video input.mp4 --output output.mp4
```

## Project Structure

```
object_detection/
├── src/
│   ├── backbone.py    # ResNet
│   ├── rpn.py         # Region proposal network
│   ├── roi_head.py    # Detection head
│   ├── model.py       # Full Faster RCNN
│   └── utils.py       # Helper functions
├── config.py          # Settings
├── dataset.py         # Data loading
├── train.py           # Training script
├── evaluate.py        # Evaluation
├── detect.py          # Inference
└── prepare_data.py    # Dataset prep
```

## Config

Edit `config.py` to change:
- Batch size, epochs, learning rate
- Backbone (resnet18 or resnet50)
- Anchor sizes
- Detection thresholds
