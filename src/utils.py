import torch
import math


def box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)

    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter

    return inter / (union + 1e-7)


def nms(boxes, scores, thresh):
    """Non-maximum suppression"""
    if len(boxes) == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    _, order = scores.sort(descending=True)
    keep = []

    while len(order) > 0:
        i = order[0].item()
        keep.append(i)
        if len(order) == 1:
            break

        iou = box_iou(boxes[i:i+1], boxes[order[1:]])[0]
        mask = iou <= thresh
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


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
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0].clamp(0, w)
    boxes[:, 1] = boxes[:, 1].clamp(0, h)
    boxes[:, 2] = boxes[:, 2].clamp(0, w)
    boxes[:, 3] = boxes[:, 3].clamp(0, h)
    return boxes
