from torch import nn
import torch
from typing import Optional, Union

from base_modules import DecoderBlock, ClassificationHead, SegmentationHead
from base_model import BaseSegmentationModel


class Linknet(BaseSegmentationModel):
    """Linknet_ is a full convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *sum*
    for fusing decoder blocks with skip connections.

    Note:
        This implementation by default has 4 skip connections (original - 3).

    Args:
        encoder: Classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, D, H, W),], for depth 1 - [(N, C, D, H, W), (N, C, D // 2, H // 2, W // 2)] and so on).
            Default is 5
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: **Linknet**

    .. _Linknet:
        https://arxiv.org/abs/1707.03718
    """

    def __init__(
        self,
        encoder,
        encoder_depth: int = 5,
        decoder_use_batchnorm: bool = True,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__(encoder=encoder)

        self.decoder = LinknetDecoder(
            encoder_channels=self.encoder.out_channels,
            n_blocks=encoder_depth,
            prefinal_channels=32,
            use_batchnorm=decoder_use_batchnorm,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=32, out_channels=classes, activation=activation, kernel_size=1
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.initialize()


class LinknetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        prefinal_channels=32,
        n_blocks=5,
        use_batchnorm=True,
    ):
        super().__init__()

        # remove first skip
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        channels = list(encoder_channels) + [prefinal_channels]

        self.blocks = nn.ModuleList(
            [DecoderBlock(channels[i], channels[i + 1], use_batchnorm=use_batchnorm) for i in range(n_blocks)]
        )

    def forward(self, *features):
        features = features[1:]  # remove first skip
        features = features[::-1]  # reverse channels to start from head of encoder

        x = features[0]
        skips = features[1:]

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
