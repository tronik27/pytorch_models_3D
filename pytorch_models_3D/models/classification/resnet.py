import torch.nn as nn

from base_blocks import ResBottleneck, BasicBlock
from base_models import BaseResNet

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200'
]


class Bottleneck(ResBottleneck):
    expansion = 4


class ResNet(BaseResNet):

    def __init__(
            self, block, layers, features_only: bool, shortcut_type='B', num_classes=400, in_channels=3,
                 ):
        super(ResNet, self).__init__(features_only=features_only, num_classes=num_classes, in_channels=in_channels)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        if not self.features_only:
            self.init_weights()

        self.out_channels = [64 * block.expansion, 128 * block.expansion, 256 * block.expansion, 512 * block.expansion]


def resnet10(in_channels: int, num_classes: int, features_only: bool = False):
    """Constructs a ResNet-10 model.
    """
    model = ResNet(
        block=BasicBlock,
        layers=[1, 1, 1, 1],
        in_channels=in_channels,
        num_classes=num_classes,
        features_only=features_only
    )
    return model


def resnet18(in_channels: int, num_classes: int, features_only: bool = False):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        in_channels=in_channels,
        num_classes=num_classes,
        features_only=features_only
    )
    return model


def resnet34(in_channels: int, num_classes: int, features_only: bool = False):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        features_only=features_only
    )
    return model


def resnet50(in_channels: int, num_classes: int, features_only: bool = False):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        features_only=features_only
    )
    return model


def resnet101(in_channels: int, num_classes: int, features_only: bool = False):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        features_only=features_only
    )
    return model


def resnet152(in_channels: int, num_classes: int, features_only: bool = False):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        features_only=features_only
    )
    return model


def resnet200(in_channels: int, num_classes: int, features_only: bool = False):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(
        block=Bottleneck,
        layers=[3, 24, 36, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        features_only=features_only
    )
    return model
