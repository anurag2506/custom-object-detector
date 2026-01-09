import torch
import math
from torchvision.ops import nms as tv_nms, box_iou as tv_box_iou


def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes"""
    return tv_box_iou(boxes1, boxes2)


def nms(boxes, scores, thresh):
    """Non-maximum suppression using torchvision's optimized implementation"""
    if len(boxes) == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    return tv_nms(boxes, scores, thresh)


def decode_boxes(deltas, anchors):
    """Decode box deltas to actual boxes"""
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    ax = anchors[:, 0] + aw / 2
    ay = anchors[:, 1] + ah / 2

    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    # clamp to prevent overflow
    dw = dw.clamp(max=math.log(1000/16))
    dh = dh.clamp(max=math.log(1000/16))

    px = dx * aw + ax
    py = dy * ah + ay
    pw = torch.exp(dw) * aw
    ph = torch.exp(dh) * ah

    x1 = px - pw / 2
    y1 = py - ph / 2
    x2 = px + pw / 2
    y2 = py + ph / 2

    return torch.stack([x1, y1, x2, y2], dim=1)


def clip_boxes(boxes, img_size):
    """Clip boxes to image boundaries"""
    h, w = img_size
    x1 = boxes[:, 0].clamp(0, w)
    y1 = boxes[:, 1].clamp(0, h)
    x2 = boxes[:, 2].clamp(0, w)
    y2 = boxes[:, 3].clamp(0, h)
    return torch.stack([x1, y1, x2, y2], dim=1)
