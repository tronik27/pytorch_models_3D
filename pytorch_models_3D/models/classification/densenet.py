import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from base_models import BaseNet

__all__ = [
    'DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet264'
]


class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, features_only=False):
        super().__init__(*args)
        self.features_only = features_only

    def forward(self, x):
        if self.features_only:
            intermediate_outputs = dict()
            output = x
            for name, module in self.named_children():
                intermediate_outputs[name] = module(output)

            return output, intermediate_outputs
        else:
            return super().forward(x)


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu.1', nn.ReLU(inplace=True))
        self.add_module('conv.1',
                        nn.Conv3d(
                            num_input_features,
                            bn_size * growth_rate,
                            kernel_size=1,
                            stride=1,
                            bias=False
                        ))
        self.add_module('norm.2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu.2', nn.ReLU(inplace=True))
        self.add_module('conv.2',
                        nn.Conv3d(
                            bn_size * growth_rate,
                            growth_rate,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False
                        ))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat(tensors=[x, new_features], dim=1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv',
                        nn.Conv3d(
                            num_input_features,
                            num_output_features,
                            kernel_size=1,
                            stride=1,
                            bias=False
                        ))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(BaseNet):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification num_classes
    """

    def __init__(
            self,
            features_only: bool,
            in_channels=3,
            growth_rate=32,
            block_config=(6, 12, 24, 16),
            num_init_features=64,
            bn_size=4,
            drop_rate=0,
            num_classes=1000
    ):
        super(DenseNet, self).__init__(features_only=features_only, num_classes=num_classes, in_channels=in_channels)

        dense_num_features = [num_init_features]
        transition_num_features = list()
        for i, num_layers in enumerate(block_config):
            transition_num_features.append(dense_num_features[i] + (i + 1) * num_layers * growth_rate)
            dense_num_features[i + 1] = transition_num_features[i] // 2

        self.start_block = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     in_channels=in_channels,
                     out_channels=num_init_features,
                     kernel_size=7,
                     stride=(1, 2, 2),
                     padding=(3, 3, 3),
                     bias=False)
                 ),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))
        self.dense_block1 = nn.Sequential(
            OrderedDict([
                ('dense1',
                 _DenseBlock(
                     num_layers=block_config[0],
                     num_input_features=dense_num_features[0],
                     bn_size=bn_size,
                     growth_rate=growth_rate,
                     drop_rate=drop_rate
                 )),
                (
                    'transition1',
                    _Transition(
                     num_input_features=transition_num_features[0], num_output_features=dense_num_features[1]
                    )
                 ),
            ]))
        self.dense_block2 = nn.Sequential(
            OrderedDict([
                ('dense2',
                 _DenseBlock(
                     num_layers=block_config[0],
                     num_input_features=dense_num_features[1],
                     bn_size=bn_size,
                     growth_rate=growth_rate,
                     drop_rate=drop_rate
                 )),
                (
                    'transition2',
                    _Transition(
                        num_input_features=transition_num_features[1], num_output_features=dense_num_features[2]
                                )
                ),
            ]))
        self.dense_block3 = nn.Sequential(
            OrderedDict([
                ('dense3',
                 _DenseBlock(
                     num_layers=block_config[0],
                     num_input_features=dense_num_features[2],
                     bn_size=bn_size,
                     growth_rate=growth_rate,
                     drop_rate=drop_rate
                 )),
                (
                    'transition3',
                    _Transition(
                        num_input_features=transition_num_features[2], num_output_features=dense_num_features[3]
                    )
                ),
            ]))
        self.dense_block4 = nn.Sequential(
            OrderedDict([
                ('dense4',
                 _DenseBlock(
                     num_layers=block_config[0],
                     num_input_features=dense_num_features[3],
                     bn_size=bn_size,
                     growth_rate=growth_rate,
                     drop_rate=drop_rate
                 )),
                ('norm4', nn.BatchNorm3d(transition_num_features[3]))
            ]))
        if not self.features_only:
            # Linear layer
            self.classifier = nn.Linear(transition_num_features[3], num_classes)
            self.init_weights()

        self.out_channels = [
            num_init_features,
            dense_num_features[1],
            dense_num_features[2],
            dense_num_features[3],
            transition_num_features[3]
        ]
        self.reduction = (2, 4, 8, 16, 32)

    def forward(self, x: torch.Tensor):
        x0 = self.start_block(x)
        x1 = self.dense_block1(x0)
        x2 = self.dense_block2(x1)
        x3 = self.dense_block3(x2)
        x4 = self.dense_block4(x3)
        if self.features_only:
            return x0, x1, x2, x3, x4
        else:
            out = F.relu(x4, inplace=True)
            out = self.avgpool(out).view(x.size(0), -1)
            out = self.classifier(out)
            return out


def densenet121(in_channels: int, num_classes: int, features_only: bool = False):
    """Constructs a DenseNet-121 model.
    """
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        in_channels=in_channels,
        num_classes=num_classes,
        features_only=features_only
    )
    return model


def densenet169(in_channels: int, num_classes: int, features_only: bool = False):
    """Constructs a DenseNet-169 model.
    """
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        in_channels=in_channels,
        num_classes=num_classes,
        features_only=features_only
    )
    return model


def densenet201(in_channels: int, num_classes: int, features_only: bool = False):
    """Constructs a DenseNet-201 model.
    """
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        in_channels=in_channels,
        num_classes=num_classes,
        features_only=features_only
    )
    return model


def densenet264(in_channels: int, num_classes: int, features_only: bool = False):
    """Constructs a DenseNet-264 model.
    """
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 64, 48),
        in_channels=in_channels,
        num_classes=num_classes,
        features_only=features_only
    )
    return model
