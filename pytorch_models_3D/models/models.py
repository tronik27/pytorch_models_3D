import functools
import numpy as np
from functools import wraps

from classification.resnet import resnet18, resnet10, resnet34, resnet50, resnet101, resnet152, resnet200
from classification.resnext import resnext50, resnext101, resnext152
from classification.wide_resnet import wide_resnet18, wide_resnet50, wide_resnet101
from classification.densenet import densenet121, densenet169, densenet201, densenet264
from classification.efficientnet import (efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
                                         efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,
                                         efficientnet_l2, efficientnet_b8)
from segmentation.deeplab import DeepLabV3Plus, DeepLabV3
from segmentation.unet import Unet
from segmentation.unetplusplus import UnetPlusPlus
from segmentation.duat import DuAT
from segmentation.fpn import FPN
from segmentation.linknet import Linknet
from segmentation.manet import MAnet
from segmentation.pspnet import PSPNet
from segmentation.pan import PAN

model2name = {
    'resnet10': resnet10,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50': resnext50,
    'resnext101': resnext101,
    'resnext152': resnext152,
    'wide_resnet18': wide_resnet18,
    'wide_resnet50': wide_resnet50,
    'wide_resnet101': wide_resnet101,
    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'densenet264': densenet264,
    'efficientnet_b0': efficientnet_b0,
    'efficientnet_b1': efficientnet_b1,
    'efficientnet_b2': efficientnet_b2,
    'efficientnet_b3': efficientnet_b3,
    'efficientnet_b4': efficientnet_b4,
    'efficientnet_b5': efficientnet_b5,
    'efficientnet_b6': efficientnet_b6,
    'efficientnet_b7': efficientnet_b7,
    'efficientnet_b8': efficientnet_b8,
    'efficientnet_l2': efficientnet_l2,
}
segmentation_model2name = {
    'duat': DuAT,
    'unet': Unet,
    'unetplusplus': UnetPlusPlus,
    'linknet': Linknet,
    'fpn': FPN,
    'deeplabv3plus': DeepLabV3Plus,
    'deeplabv3': DeepLabV3,
    'manet': MAnet,
    'pspnet': PSPNet,
    'pan': PAN,
}


def check_name(func):

    @wraps(func)
    def wrapper_func(name: str, *args, **kwargs):
        model2name.update(segmentation_model2name)
        if name not in model2name:
            raise NotImplementedError(
                f'3D model {name} have not been implemented! Use one of {list(model2name.keys())}'
            )
        func(*args, **kwargs)
    return wrapper_func


@check_name
def get_classification_model(name: str, in_channels: int, num_classes: int):
    """
    Create classification model
    Args:
        name: name of classification model
        in_channels: number of image channels for the model, default is 3 (RGB images)
        num_classes: A number of num_classes for output

    Returns:
        ``torch.nn.Module``: classification model
    """
    model = model2name[name]
    return model(in_channels=in_channels, num_classes=num_classes)


@check_name
def get_encoder(name: str, in_channels: int):
    """
    Create encoder model.
    Args:
        name: name of classification model
        in_channels:  number of image channels for the model, default is 3 (RGB images)

    Returns:
        ``torch.nn.Module``: encoder model
    """
    model = model2name[name]
    return model(in_channels=in_channels, num_classes=1, features_only=True)


@check_name
def get_segmentation_model(name: str, in_channels: int, num_classes: int, encoder_name: str, **kwargs):
    """
    Create segmentation model.
    Args:
        name: name of segmentation model
        in_channels: A number of image channels for the model, default is 3 (RGB images)
        num_classes: A number of num_classes for output mask (or you can think as a number of channels of output mask)
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        **kwargs:
        Other model params, like: aux_params, decoder_attention_type, activation, etc.
    Returns:
        ``torch.nn.Module``: segmentation model
    """
    encoder = get_encoder(name=encoder_name, in_channels=in_channels)
    model = segmentation_model2name[name](encoder=encoder, num_classes=num_classes, **kwargs)
    model.name = f'3D-{name}-{encoder_name}'
    return model


def get_preprocessing_fn():
    return functools.partial(preprocess_input)


def preprocess_input(x, input_space="RGB", input_range=None):
    std = [0.229, 0.224, 0.225]
    mean = [0.229, 0.224, 0.225]

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x

