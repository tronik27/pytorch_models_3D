from typing import Tuple
from collections import OrderedDict
import torch
from torch import nn
import os
import segmentation_models_pytorch as smp


config2model = {
    'deeplab': smp.DeepLabV3Plus,
    'unet++': smp.UnetPlusPlus,
    'manet': smp.MAnet,
    'unet': smp.Unet,
    'linknet': smp.Linknet,
    'fpn': smp.FPN,
    'pspnet': smp.PSPNet,
    'pan': smp.PAN,
    'deeplabv3': smp.DeepLabV3
}


class PathologyModel(nn.Module):
    def __init__(self, config: dict, num_classes: int) -> None:
        """
        Lungs segmentation nn model class.
        :parameter config: list of parameters indicating which nn weights to load.
        :param num_classes: the device on which the models working in the pipeline will be launched.
        """
        super().__init__()
        self.model = config2model[config['model_name']](
            encoder_name=config['encoder'],
            encoder_weights=config['encoder_weights'],
            in_channels=3,
            classes=num_classes,
            aux_params = dict(pooling='avg', classes=num_classes),
        )
        if config.setdefault('continue_training'):
            self.load_weights(os.path.join(config['save_path'], 'last.ckpt'))
        elif config.setdefault('pretrain'):
            self.load_pretrain(config['pretrain'])
        if config.setdefault('freeze'):
            if config['freeze'] == 'encoder':
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
            elif config['freeze'] == 'all':
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                for param in self.model.decoder.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method for making predictions for single image.
        :param x: tensor containing image under study.
        :returns torch tensors containing predicted masks and class predicted class labels.
        """
        masks, labels = self.model(x)
        return labels, masks

    def load_pretrain(self, weights_path: str) -> None:
        """
        Method for pretrained model weights loading.
        :param weights_path: path to pretrained weights.
        """
        weights_data = torch.load(weights_path, map_location='cpu')
        weights = weights_data['model_state_dict'] if 'model_state_dict' in weights_data else weights_data['state_dict']
        encoder_weights = OrderedDict()
        cls_head_weights = OrderedDict()
        segm_head_weights = OrderedDict()
        for key, val in weights.items():
            new_key = key
            if key.startswith("model."):
                new_key = key.replace('model', '').strip('.')

            if new_key.startswith("encoder"):
                new_key = new_key[len("encoder."):]
                encoder_weights[new_key] = val

            if new_key.startswith("classification_head"):
                new_key = new_key[len("classification_head."):]
                cls_head_weights[new_key] = val

            if new_key.startswith("decoder"):
                new_key = new_key[len("decoder."):]
                segm_head_weights[new_key] = val

        cls_head_weights.pop("3.weight", None)
        cls_head_weights.pop("3.bias", None)
        segm_head_weights.pop("SBA.conv.1.weight", None)
        segm_head_weights.pop("fuse2.1.weight", None)

        self.model.encoder.load_state_dict(encoder_weights, strict=True)
        self.classification_head.load_state_dict(cls_head_weights, strict=False)
        self.model.decoder.load_state_dict(cls_head_weights, strict=False)

    def load_weights(self, path_to_model: str) -> None:
        """
        Method for model weights loading.
        :param path_to_model: path to trained weights.
        """
        weights_data = torch.load(path_to_model, map_location='cpu')
        weights = weights_data['model_state_dict'] if 'model_state_dict' in weights_data else weights_data['state_dict']
        new_weights = OrderedDict()
        for key, val in weights.items():
            if key.startswith("model"):
                new_key = key.replace('model', '').strip('.')
                new_weights[new_key] = val

        new_weights.pop("criterion.cls_loss.bce.pos_weight", None)
        self.load_state_dict(new_weights, strict=False)


