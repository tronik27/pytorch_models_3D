import torch.nn as nn

from base_blocks import ResBottleneck
from base_models import BaseResNet

__all__ = ['WideResNet', 'wide_resnet18', 'wide_resnet50', 'wide_resnet101']


class WideBottleneck(ResBottleneck):
    expansion = 2


class WideResNet(BaseResNet):

    def __init__(
            self, block, layers, features_only: bool, in_channels=3, k=1, shortcut_type='B', num_classes=400
    ):
        super(WideResNet, self).__init__(features_only=features_only, num_classes=num_classes, in_channels=in_channels)
        self.layer1 = self._make_layer(block, 64 * k, layers[0], shortcut_type, stride=1)
        self.layer2 = self._make_layer(block, 128 * k, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256 * k, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512 * k, layers[3], shortcut_type, stride=2)
        self.fc = nn.Linear(512 * k * block.expansion, num_classes)
        if not self.features_only:
            self.init_weights()

        self.out_channels = [
            64 * k * block.expansion, 128 * k * block.expansion, 256 * k * block.expansion, 512 * k * block.expansion
        ]


def wide_resnet18(in_channels: int, num_classes: int, features_only: bool = False):
    """Constructs a WideResNet-18 model.
    """
    model = WideResNet(
        block=WideBottleneck,
        layers=[2, 2, 2, 2],
        in_channels=in_channels,
        num_classes=num_classes,
        features_only=features_only
    )
    return model


def wide_resnet50(in_channels: int, num_classes: int, features_only: bool = False):
    """Constructs a WideResNet-50 model.
    """
    model = WideResNet(
        block=WideBottleneck,
        layers=[3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        features_only=features_only
    )
    return model


def wide_resnet101(in_channels: int, num_classes: int, features_only: bool = False):
    """Constructs a WideResNet-101 model.
    """
    model = WideResNet(
        block=WideBottleneck,
        layers=[3, 4, 23, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        features_only=features_only
    )
    return model
