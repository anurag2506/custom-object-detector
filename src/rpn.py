import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils import box_iou, decode_boxes, clip_boxes, nms


class AnchorGenerator:
    """Generates anchor boxes at each feature map position"""

    def __init__(self, sizes, ratios):
        self.sizes = sizes
        self.ratios = ratios
        self.num_anchors = len(sizes) * len(ratios)

    def generate(self, feat_size, stride, device):
        h, w = feat_size
        anchors = []

        # base anchors centered at origin
        base = []
        for size in self.sizes:
            for ratio in self.ratios:
                ah = size / math.sqrt(ratio)
                aw = size * math.sqrt(ratio)
                base.append([-aw/2, -ah/2, aw/2, ah/2])
        base = torch.tensor(base, device=device)

        # shift to each grid position
        shifts_x = torch.arange(0, w, device=device) * stride + stride // 2
        shifts_y = torch.arange(0, h, device=device) * stride + stride // 2
        sy, sx = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shifts = torch.stack([sx, sy, sx, sy], dim=-1).reshape(-1, 1, 4)

        anchors = (shifts + base.view(1, -1, 4)).reshape(-1, 4)
        return anchors


class RPN(nn.Module):
    """Region Proposal Network"""

    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.cls_layer = nn.Conv2d(512, num_anchors * 2, 1)
        self.reg_layer = nn.Conv2d(512, num_anchors * 4, 1)

        # init
        for l in [self.conv, self.cls_layer, self.reg_layer]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, features, anchors, img_size):
        batch = features.shape[0]

        x = F.relu(self.conv(features))
        cls_out = self.cls_layer(x).permute(0, 2, 3, 1).reshape(batch, -1, 2)
        reg_out = self.reg_layer(x).permute(0, 2, 3, 1).reshape(batch, -1, 4)

        # generate proposals
        proposals = []
        for i in range(batch):
            props = self._get_proposals(anchors, cls_out[i], reg_out[i], img_size)
            proposals.append(props)

        return cls_out, reg_out, proposals

    def _get_proposals(self, anchors, cls, reg, img_size, pre_nms=2000, post_nms=1000):
        scores = F.softmax(cls, dim=-1)[:, 1]
        boxes = decode_boxes(reg, anchors)
        boxes = clip_boxes(boxes, img_size)

        # filter small boxes
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        keep = (ws >= 16) & (hs >= 16)
        boxes, scores = boxes[keep], scores[keep]

        # top k
        k = min(pre_nms, len(scores))
        _, idx = scores.topk(k)
        boxes, scores = boxes[idx], scores[idx]

        # nms
        keep = nms(boxes, scores, 0.7)
        keep = keep[:post_nms]

        return boxes[keep]


class RPNLoss:
    """Computes RPN loss"""

    def __init__(self, pos_thresh=0.7, neg_thresh=0.3, batch_size=256):
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.batch_size = batch_size

    def __call__(self, cls_out, reg_out, anchors, targets):
        device = cls_out.device
        batch = len(targets)

        cls_loss = 0
        reg_loss = 0

        for i in range(batch):
            gt_boxes = targets[i]['boxes']
            if len(gt_boxes) == 0:
                continue

            # match anchors to gt
            iou = box_iou(anchors, gt_boxes)
            max_iou, matched_idx = iou.max(dim=1)

            # labels: 1=pos, 0=neg, -1=ignore
            labels = torch.full((len(anchors),), -1, device=device, dtype=torch.long)
            labels[max_iou >= self.pos_thresh] = 1
            labels[max_iou < self.neg_thresh] = 0

            # also mark best anchor for each gt as positive
            best_anchor = iou.argmax(dim=0)
            labels[best_anchor] = 1

            # sample
            pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
            neg_idx = (labels == 0).nonzero(as_tuple=True)[0]

            n_pos = min(self.batch_size // 2, len(pos_idx))
            n_neg = min(self.batch_size - n_pos, len(neg_idx))

            pos_idx = pos_idx[torch.randperm(len(pos_idx))[:n_pos]]
            neg_idx = neg_idx[torch.randperm(len(neg_idx))[:n_neg]]
            sampled = torch.cat([pos_idx, neg_idx])

            # cls loss
            cls_loss += F.cross_entropy(cls_out[i][sampled], labels[sampled])

            # reg loss (only for positives)
            if n_pos > 0:
                pos_anchors = anchors[pos_idx]
                pos_gt = gt_boxes[matched_idx[pos_idx]]
                targets_reg = self._encode(pos_gt, pos_anchors)
                reg_loss += F.smooth_l1_loss(reg_out[i][pos_idx], targets_reg)

        return {
            'rpn_cls': cls_loss / batch,
            'rpn_reg': reg_loss / batch
        }

    def _encode(self, gt, anchors):
        # encode gt boxes relative to anchors
        aw = anchors[:, 2] - anchors[:, 0]
        ah = anchors[:, 3] - anchors[:, 1]
        ax = anchors[:, 0] + aw / 2
        ay = anchors[:, 1] + ah / 2

        gw = gt[:, 2] - gt[:, 0]
        gh = gt[:, 3] - gt[:, 1]
        gx = gt[:, 0] + gw / 2
        gy = gt[:, 1] + gh / 2

        dx = (gx - ax) / aw
        dy = (gy - ay) / ah
        dw = torch.log(gw / aw)
        dh = torch.log(gh / ah)

        return torch.stack([dx, dy, dw, dh], dim=1)
