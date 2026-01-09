import torch
import torch.nn as nn

from .backbone import resnet50, resnet18
from .rpn import RPN, AnchorGenerator, RPNLoss
from .roi_head import RoIHead


class FasterRCNN(nn.Module):
    """Faster RCNN object detector"""

    def __init__(self, num_classes, backbone='resnet50'):
        super().__init__()

        # backbone
        if backbone == 'resnet50':
            self.backbone = resnet50()
        else:
            self.backbone = resnet18()

        in_ch = self.backbone.out_channels

        # anchor generator
        self.anchor_gen = AnchorGenerator(
            sizes=[32, 64, 128, 256, 512],
            ratios=[0.5, 1.0, 2.0]
        )

        # rpn
        self.rpn = RPN(in_ch, self.anchor_gen.num_anchors)
        self.rpn_loss = RPNLoss()

        # roi head
        self.roi_head = RoIHead(in_ch, num_classes)

        self.stride = 32

    def forward(self, images, targets=None):
        if isinstance(images, list):
            images = torch.stack(images)

        batch_size = images.shape[0]
        img_size = (images.shape[2], images.shape[3])

        # backbone
        features = self.backbone(images)

        # generate anchors
        feat_size = (features.shape[2], features.shape[3])
        anchors = self.anchor_gen.generate(feat_size, self.stride, features.device)

        # rpn
        cls_out, reg_out, proposals = self.rpn(features, anchors, img_size)

        if self.training:
            rpn_losses = self.rpn_loss(cls_out, reg_out, anchors, targets)
            img_sizes = [img_size] * batch_size
            roi_losses = self.roi_head(features, proposals, img_sizes, targets)
            return {**rpn_losses, **roi_losses}
        else:
            img_sizes = [img_size] * batch_size
            return self.roi_head(features, proposals, img_sizes)
