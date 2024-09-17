from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from functools import partial

from utils import downsample_basic_block


class BaseNet(ABC, nn.Module):

    def __init__(self, features_only: bool, num_classes: int, in_channels: int):
        super().__init__()
        self.features_only = features_only
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    @abstractmethod
    def forward(self, x: torch.Tensor):
        pass

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class BaseResNet(BaseNet):
    def __init__(self, features_only: bool, num_classes: int, in_channels: int):
        super().__init__(features_only=features_only, num_classes=num_classes, in_channels=in_channels)
        self.inplanes = 64
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False
        )
        self.reduction = (2, 4, 8, 16, 32)

    def _make_layer(self, block, planes, blocks, shortcut_type: str, stride: int, *args):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False
                    ),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, *args))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x)
        if self.features_only:
            return x, x1, x2, x3, x4
        else:
            x = self.avgpool(x4)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x
