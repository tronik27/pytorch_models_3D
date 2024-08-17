import torch.nn as nn

from base_blocks import BaseBottleneck
from base_models import BaseResNet

__all__ = ['ResNeXt', 'resnext50', 'resnext101', 'resnext152']


class ResNeXtBottleneck(BaseBottleneck):
    expansion = 2

    def __init__(
            self, inplanes, planes, cardinality, stride=1, downsample=None
    ):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


class ResNeXt(BaseResNet):

    def __init__(
            self, block, layers, features_only: bool, in_channels=3, shortcut_type='B', cardinality=32, num_classes=400
                 ):
        super(ResNeXt, self).__init__(features_only=features_only, num_classes=num_classes, in_channels=in_channels)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type, 1, cardinality)
        self.layer2 = self._make_layer(block, 256, layers[1], shortcut_type,  2, cardinality)
        self.layer3 = self._make_layer(block, 512, layers[2], shortcut_type, 2, cardinality)
        self.layer4 = self._make_layer(block, 1024, layers[3], shortcut_type, 2, cardinality)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)
        if not self.features_only:
            self.init_weights()

        self.out_channels = [
            128 * block.expansion, 256 * block.expansion, 512 * block.expansion, 1024 * block.expansion
        ]


def resnext50(in_channels: int, num_classes: int, features_only: bool = False):
    """Constructs a ResNext-50 model.
    """
    model = ResNeXt(
        block=ResNeXtBottleneck,
        layers=[3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        features_only=features_only
    )
    return model


def resnext101(in_channels: int, num_classes: int, features_only: bool = False):
    """Constructs a ResNext-101 model.
    """
    model = ResNeXt(
        block=ResNeXtBottleneck,
        layers=[3, 4, 23, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        features_only=features_only
    )
    return model


def resnext152(in_channels: int, num_classes: int, features_only: bool = False):
    """Constructs a ResNext-152 model.
    """
    model = ResNeXt(
        block=ResNeXtBottleneck,
        layers=[3, 8, 36, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        features_only=features_only
    )
    return model
