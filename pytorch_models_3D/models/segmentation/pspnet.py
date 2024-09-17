from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple

from base_modules import ClassificationHead, SegmentationHead, Conv3dReLU
from base_model import BaseSegmentationModel


class PSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, use_batchnorm=True):
        super().__init__()
        if pool_size == 1:
            use_batchnorm = False  # PyTorch does not support BatchNorm for 1x1 shape
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(output_size=(pool_size, pool_size, pool_size)),
            Conv3dReLU(in_channels, out_channels, kernel_size=(1, 1, 1), use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_batchnorm=True):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                PSPBlock(
                    in_channels,
                    in_channels // len(sizes),
                    size,
                    use_batchnorm=use_batchnorm,
                )
                for size in sizes
            ]
        )

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x


class PSPDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        use_batchnorm=True,
        out_channels=512,
        dropout=0.2,
    ):
        super().__init__()

        self.psp = PSPModule(
            in_channels=encoder_channels[-1],
            sizes=(1, 2, 3, 6),
            use_batchnorm=use_batchnorm,
        )

        self.conv = Conv3dReLU(
            in_channels=encoder_channels[-1] * 2,
            out_channels=out_channels,
            kernel_size=1,
            use_batchnorm=use_batchnorm,
        )

        self.dropout = nn.Dropout3d(p=dropout)

    def forward(self, *features):
        x = features[-1]
        x = self.psp(x)
        x = self.conv(x)
        x = self.dropout(x)

        return x


class PSPNet(BaseSegmentationModel):
    """PSPNet_ is a full convolution neural network for image semantic segmentation. Consist of
    *encoder* and *Spatial Pyramid* (decoder). Spatial Pyramid build on top of encoder and does not
    use "fine-features" (features of high spatial resolution). PSPNet can be used for multiclass segmentation
    of high resolution images, however it is not good for detecting small objects and producing accurate,
    pixel-level mask.

    Args:
        encoder: Classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        psp_out_channels: A number of filters in Spatial Pyramid
        psp_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        psp_dropout: Spatial dropout rate in [0, 1) used in Spatial Pyramid
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 8 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: **PSPNet**

    .. _PSPNet:
        https://arxiv.org/abs/1612.01105
    """

    def __init__(
        self,
        encoder,
        psp_out_channels: int = 512,
        psp_use_batchnorm: bool = True,
        psp_dropout: float = 0.2,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        upsampling: int = 8,
        aux_params: Optional[dict] = None,
    ):
        super().__init__(encoder=encoder)

        self.decoder = PSPDecoder(
            encoder_channels=self.encoder.out_channels,
            use_batchnorm=psp_use_batchnorm,
            out_channels=psp_out_channels,
            dropout=psp_dropout,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=psp_out_channels,
            out_channels=classes,
            kernel_size=3,
            activation=activation,
            upsampling=upsampling,
        )

        if aux_params:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.initialize()
