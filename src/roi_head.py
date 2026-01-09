import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

from .utils import box_iou, decode_boxes, clip_boxes, nms


class RoIHead(nn.Module):
    """Detection head with RoI Align"""

    def __init__(self, in_channels, num_classes, roi_size=7):
        super().__init__()
        self.num_classes = num_classes

        self.roi_align = RoIAlign((roi_size, roi_size), spatial_scale=1/32, sampling_ratio=2)

        feat_dim = in_channels * roi_size * roi_size
        self.fc1 = nn.Linear(feat_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

        # init
        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(l.weight)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, features, proposals, img_sizes, targets=None):
        if self.training:
            proposals, labels, reg_targets = self._sample_proposals(proposals, targets)

        # prepare rois for roi_align
        rois = []
        for i, props in enumerate(proposals):
            idx = torch.full((len(props), 1), i, device=props.device)
            rois.append(torch.cat([idx, props], dim=1))
        rois = torch.cat(rois)

        # roi align + fc layers
        roi_feats = self.roi_align(features, rois)
        roi_feats = roi_feats.flatten(1)
        x = F.relu(self.fc1(roi_feats))
        x = F.relu(self.fc2(x))

        cls_logits = self.cls_score(x)
        box_reg = self.bbox_pred(x)

        if self.training:
            return self._compute_loss(cls_logits, box_reg, labels, reg_targets)
        else:
            return self._postprocess(cls_logits, box_reg, proposals, img_sizes)

    def _sample_proposals(self, proposals, targets, batch_size=512, pos_frac=0.25):
        sampled_props = []
        all_labels = []
        all_targets = []

        for props, tgt in zip(proposals, targets):
            gt_boxes = tgt['boxes']
            gt_labels = tgt['labels']
            device = props.device

            if len(gt_boxes) == 0:
                # no gt - all background
                n = min(batch_size, len(props))
                idx = torch.randperm(len(props))[:n]
                sampled_props.append(props[idx])
                all_labels.append(torch.zeros(n, dtype=torch.long, device=device))
                all_targets.append(torch.zeros(n, 4, device=device))
                continue

            iou = box_iou(props, gt_boxes)
            max_iou, matched = iou.max(dim=1)

            labels = gt_labels[matched].clone()
            labels[max_iou < 0.5] = 0

            pos_idx = (labels > 0).nonzero(as_tuple=True)[0]
            neg_idx = (labels == 0).nonzero(as_tuple=True)[0]

            n_pos = min(int(batch_size * pos_frac), len(pos_idx))
            n_neg = min(batch_size - n_pos, len(neg_idx))

            pos_idx = pos_idx[torch.randperm(len(pos_idx))[:n_pos]]
            neg_idx = neg_idx[torch.randperm(len(neg_idx))[:n_neg]]
            sampled = torch.cat([pos_idx, neg_idx])

            sampled_props.append(props[sampled])
            all_labels.append(labels[sampled])

            # regression targets
            reg_tgt = torch.zeros(len(sampled), 4, device=device)
            if n_pos > 0:
                reg_tgt[:n_pos] = self._encode(gt_boxes[matched[pos_idx]], props[pos_idx])
            all_targets.append(reg_tgt)

        return sampled_props, torch.cat(all_labels), torch.cat(all_targets)

    def _encode(self, gt, props):
        pw = props[:, 2] - props[:, 0]
        ph = props[:, 3] - props[:, 1]
        px = props[:, 0] + pw / 2
        py = props[:, 1] + ph / 2

        gw = gt[:, 2] - gt[:, 0]
        gh = gt[:, 3] - gt[:, 1]
        gx = gt[:, 0] + gw / 2
        gy = gt[:, 1] + gh / 2

        return torch.stack([
            (gx - px) / pw,
            (gy - py) / ph,
            torch.log(gw / pw),
            torch.log(gh / ph)
        ], dim=1)

    def _compute_loss(self, cls_logits, box_reg, labels, reg_targets):
        cls_loss = F.cross_entropy(cls_logits, labels)

        pos_mask = labels > 0
        if pos_mask.sum() > 0:
            box_reg = box_reg.view(-1, self.num_classes, 4)
            pos_labels = labels[pos_mask]
            idx = torch.arange(len(pos_labels), device=labels.device)
            pos_reg = box_reg[pos_mask][idx, pos_labels]
            reg_loss = F.smooth_l1_loss(pos_reg, reg_targets[pos_mask])
        else:
            reg_loss = torch.tensor(0.0, device=labels.device)

        return {'det_cls': cls_loss, 'det_reg': reg_loss}

    def _postprocess(self, cls_logits, box_reg, proposals, img_sizes, thresh=0.05, nms_thresh=0.5):
        probs = F.softmax(cls_logits, dim=-1)
        box_reg = box_reg.view(-1, self.num_classes, 4)

        # split by image
        splits = [len(p) for p in proposals]
        probs_split = probs.split(splits)
        reg_split = box_reg.split(splits)

        results = []
        for prob, reg, props, img_size in zip(probs_split, reg_split, proposals, img_sizes):
            boxes_all = []
            scores_all = []
            labels_all = []

            for c in range(1, self.num_classes):
                scores = prob[:, c]
                keep = scores > thresh
                if keep.sum() == 0:
                    continue

                scores = scores[keep]
                deltas = reg[keep, c]
                class_props = props[keep]

                boxes = decode_boxes(deltas, class_props)
                boxes = clip_boxes(boxes, img_size)

                boxes_all.append(boxes)
                scores_all.append(scores)
                labels_all.append(torch.full_like(scores, c, dtype=torch.long))

            if len(boxes_all) == 0:
                results.append({
                    'boxes': torch.empty(0, 4, device=probs.device),
                    'labels': torch.empty(0, dtype=torch.long, device=probs.device),
                    'scores': torch.empty(0, device=probs.device)
                })
                continue

            boxes_all = torch.cat(boxes_all)
            scores_all = torch.cat(scores_all)
            labels_all = torch.cat(labels_all)

            # per-class nms
            keep = self._batched_nms(boxes_all, scores_all, labels_all, nms_thresh)
            keep = keep[:100]

            results.append({
                'boxes': boxes_all[keep],
                'labels': labels_all[keep],
                'scores': scores_all[keep]
            })

        return results

    def _batched_nms(self, boxes, scores, labels, thresh):
        if len(boxes) == 0:
            return torch.empty(0, dtype=torch.long, device=boxes.device)
        offset = labels.float() * (boxes.max() + 1)
        boxes_offset = boxes.clone()
        boxes_offset[:, 0] += offset
        boxes_offset[:, 2] += offset
        return nms(boxes_offset, scores, thresh)
