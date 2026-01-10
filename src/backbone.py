import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as tv_resnet18, resnet50 as tv_resnet50
from torchvision.models import ResNet18_Weights, ResNet50_Weights


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * 4)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)


class ResNetBackbone(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.out_channels = 512 * block.expansion

    def _make_layer(self, block, out_ch, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_ch * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_ch * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch * block.expansion)
            )

        layers = [block(self.in_channels, out_ch, stride, downsample)]
        self.in_channels = out_ch * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_ch))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet50(pretrained=True):
    backbone = ResNetBackbone(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        print("Loading pretrained ResNet50 weights...")
        pretrained_model = tv_resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        pretrained_dict = pretrained_model.state_dict()
        backbone_dict = backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone_dict}
        backbone_dict.update(pretrained_dict)
        backbone.load_state_dict(backbone_dict)
        print(f"Loaded {len(pretrained_dict)} pretrained layers")
    return backbone


def resnet18(pretrained=True):
    backbone = ResNetBackbone(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        print("Loading pretrained ResNet18 weights...")
        pretrained_model = tv_resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        pretrained_dict = pretrained_model.state_dict()
        backbone_dict = backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in backbone_dict}
        backbone_dict.update(pretrained_dict)
        backbone.load_state_dict(backbone_dict)
        print(f"Loaded {len(pretrained_dict)} pretrained layers")
    return backbone
