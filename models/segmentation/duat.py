import torch.nn.functional as F
import warnings
import torch
from torch import nn
from typing import Tuple, Dict, List, Optional
from rich import print as rprint

from base_model import BaseSegmentationModel
from base_modules import ClassificationHead, Activation

warnings.filterwarnings('ignore')


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(
        module: nn.Module,
        a: float = 0,
        mode: str = 'fan_out',
        nonlinearity: str = 'relu',
        bias: float = 0,
        distribution: str = 'normal'
) -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def upsample(x, size, align_corners=False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv3d, self).__init__()

        self.conv = nn.Conv3d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block(nn.Sequential):
    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True, norm_layer=nn.BatchNorm3d):
        super(Block, self).__init__()
        if bn_start:
            self.add_module(name='norm1', module=norm_layer(input_num))

        self.add_module(name='relu1', module=nn.ReLU(inplace=True))
        self.add_module(name='conv1', module=nn.Conv3d(in_channels=input_num, out_channels=num1, kernel_size=1))

        self.add_module(name='norm2', module=norm_layer(num1))
        self.add_module(name='relu2', module=nn.ReLU(inplace=True))
        self.add_module(
            name='conv2',
            module=nn.Conv3d(
                in_channels=num1, out_channels=num2, kernel_size=3, dilation=dilation_rate, padding=dilation_rate
            )
        )
        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(Block, self).forward(_input)
        if self.drop_rate > 0:
            feature = F.dropout3d(feature, p=self.drop_rate, training=self.training)
        return feature


class ContextBlock(nn.Module):

    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_mul',)):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv3d(in_channels=inplanes, out_channels=1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv3d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv3d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv3d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv3d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, depth, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, D * H * W]
            input_x = input_x.view(batch, channel, depth * height * width)
            # [N, 1, C, D * H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, D, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, depth * height * width)
            # [N, 1, D * H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, D * H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1, 1]
            context = context.view(batch, channel, 1, 1, 1)
        else:
            # [N, C, 1, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out + out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat(tensors=[avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvBranch(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm3d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm3d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm3d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv3d(hidden_features, out_features, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        res1 = x
        res2 = x
        x = self.conv1(x)
        x = x + self.conv2(x)
        x = self.conv3(x)
        x = x + self.conv4(x)
        x = self.conv5(x)
        x = x + self.conv6(x)
        x = self.conv7(x)
        x_mask = self.sigmoid_spatial(x)
        res1 = res1 * x_mask
        return res2 + res1


class GLSA(nn.Module):

    def __init__(self, input_dim=512, embed_dim=32, kernel_size=1):
        super().__init__()

        self.conv1_1 = BasicConv3d(embed_dim * 2, embed_dim, kernel_size=kernel_size)
        self.conv1_1_1 = BasicConv3d(input_dim // 2, embed_dim, kernel_size=kernel_size)
        self.local_11conv = nn.Conv3d(input_dim // 2, embed_dim, kernel_size=kernel_size)
        self.global_11conv = nn.Conv3d(input_dim // 2, embed_dim, kernel_size=kernel_size)
        self.global_block = ContextBlock(inplanes=embed_dim, ratio=2)
        self.local = ConvBranch(in_features=embed_dim, hidden_features=embed_dim, out_features=embed_dim)

    def forward(self, x):
        x_0, x_1 = x.chunk(2, dim=1)
        # local block
        local_attention = self.local(self.local_11conv(x_0))
        # global block
        global_attention = self.global_block(self.global_11conv(x_1))
        # concat Global + local
        x = torch.cat(tensors=[local_attention, global_attention], dim=1)
        x = self.conv1_1(x)
        return x


class SBA(nn.Module):

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.d_in1 = BasicConv3d(input_dim // 2, input_dim // 2, kernel_size=1)
        self.d_in2 = BasicConv3d(input_dim // 2, input_dim // 2, kernel_size=1)
        self.fc1 = nn.Conv3d(input_dim, input_dim // 2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv3d(input_dim, input_dim // 2, kernel_size=1, bias=False)
        self.Sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(
            BasicConv3d(in_planes=input_dim, out_planes=input_dim, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(input_dim, num_classes, kernel_size=1, bias=False)
        )

    def forward(self, h_feature, l_feature):
        l_feature = self.fc1(l_feature)
        h_feature = self.fc2(h_feature)

        g_l_feature = self.Sigmoid(l_feature)
        g_h_feature = self.Sigmoid(h_feature)

        l_feature = self.d_in1(l_feature)
        h_feature = self.d_in2(h_feature)

        l_feature = l_feature + l_feature * g_l_feature + (1 - g_l_feature) * upsample(
            g_h_feature * h_feature, size=l_feature.size()[2:], align_corners=False
        )
        h_feature = h_feature + h_feature * g_h_feature + (1 - g_h_feature) * upsample(
            g_l_feature * l_feature, size=h_feature.size()[2:], align_corners=False
        )

        h_feature = upsample(h_feature, size=l_feature.size()[2:])
        out = self.conv(torch.cat(tensors=[h_feature, l_feature], dim=1))
        return out


class DuatDecoder(nn.Module):
    def __init__(self, num_classes: int, max_reduction: int, dims: Tuple[int, int, int, int], activation: str):
        super(DuatDecoder, self).__init__()

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]

        self.GLSA_c4 = GLSA(input_dim=c4_in_channels, embed_dim=max_reduction)
        self.GLSA_c3 = GLSA(input_dim=c3_in_channels, embed_dim=max_reduction)
        self.GLSA_c2 = GLSA(input_dim=c2_in_channels, embed_dim=max_reduction)
        self.L_feature = BasicConv3d(c1_in_channels, max_reduction, kernel_size=3, stride=1, padding=1)

        self.SBA = SBA(input_dim=max_reduction, num_classes=num_classes)
        self.fuse = BasicConv3d(max_reduction * 2, max_reduction, kernel_size=1)
        self.fuse2 = nn.Sequential(
            BasicConv3d(max_reduction * 3, max_reduction, kernel_size=1, stride=1),
            nn.Conv3d(max_reduction, num_classes, kernel_size=1, bias=False)
        )
        self.activation = Activation(activation)

    def forward(
            self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method for making predictions for single image.
        :param c1: tensor from encoder level 1.
        :param c2: tensor from encoder level 2.
        :param c3: tensor from encoder level 3.
        :param c4: tensor from encoder level 4.
        :returns torch tensors containing predicted masks and class predicted class labels.
        """
        _c4 = self.GLSA_c4(c4)  # [1, 64, 11, 11]
        _c4 = upsample(_c4, c3.size()[2:])
        _c3 = self.GLSA_c3(c3)  # [1, 64, 22, 22]
        _c2 = self.GLSA_c2(c2)  # [1, 64, 44, 44]

        output1 = self.fuse2(torch.cat([upsample(_c4, c2.size()[2:]), upsample(_c3, c2.size()[2:]), _c2], dim=1))

        l_feature = self.L_feature(c1)  # [1, 64, 88, 88]
        h_feature = self.fuse(torch.cat([_c4, _c3], dim=1))
        h_feature = upsample(h_feature, c2.size()[2:])

        output2 = self.SBA(h_feature, l_feature)

        output1 = self.activation(output1)
        output2 = self.activation(output2)

        output1 = F.interpolate(output1, scale_factor=8, mode='bilinear')
        output2 = F.interpolate(output2, scale_factor=4, mode='bilinear')

        return output1, output2


class DuAT(BaseSegmentationModel):

    def __init__(
            self, encoder, num_classes: int, aux_params: Optional[dict] = None, activation: Optional[str] = None
    ) -> None:
        """
        Dual Aggregation Transformer Network model class for 3D sematic segmentation.
        Args:
            encoder: Classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution.
            num_classes: A number of num_classes for output mask.
            activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
            aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is
            build on top of encoder if **aux_params** is not **None** (default). Supported params:
                - num_classes (int): A number of num_classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)
        """
        self.reduction = (4, 8, 16, 32)
        super(DuAT, self).__init__(encoder=encoder)
        self.decoder = DuatDecoder(
            num_classes=num_classes,
            max_reduction=self.reduction[-1],
            dims=self.encoder.out_channels,
            activation=activation
        )
        self.idxs = [i for i, e in enumerate(self.encoder.reduction) if e in self.reduction]
        if len(self.idxs) != len(self.reduction):
            print_info(
                text=
                f'Encoder reduction rates are {",".join(self.encoder.reduction)} which is not suitable for Duat model!',
                type_info=False
            )
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Method for making predictions for single image.
        :param x: tensor containing image under study.
        :returns torch tensors containing predicted masks and class predicted class labels.
        """
        c1, c2, c3, c4 = [e for i, e in enumerate(self.encoder(x)) if i in self.idxs]
        output1, output2 = self.decoder(c1=c1, c2=c2, c3=c3, c4=c4)
        labels = self.classification_head(c4)
        return labels, [output1, output2]


def print_info(text: str, type_info: bool = True):
    if type_info:
        rprint(f"[bold yellow][INFO][/bold yellow] [italic green]{text}[/italic green]")
    else:
        rprint(f"[bold orange_red1][WARNING][/bold orange_red1] [italic yellow]{text}[/italic yellow]")
