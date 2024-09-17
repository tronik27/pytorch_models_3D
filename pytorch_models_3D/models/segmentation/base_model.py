import torch
from torch import nn
from typing import Tuple
from collections import OrderedDict
from rich import print as rprint

from base_modules import initialize_decoder, initialize_head


class BaseSegmentationModel(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = None
        self.classification_head = None
        self.name = ''

    def initialize(self):
        initialize_decoder(self.decoder)
        if self.decoder is not None:
            initialize_head(self.decoder)
        if self.classification_head is not None:
            initialize_head(self.classification_head)

    def check_input_shape(self, x: torch.Tensor) -> None:
        """
        Check shape of input data tensor.
        Args:
            x: input data tensor

        Returns:

        """
        d, h, w = x.shape[-3:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0 or d % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            new_d = (d // output_stride + 1) * output_stride if d % output_stride != 0 else d
            raise RuntimeError(
                f"Wrong x shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w}, {new_d})."
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] or torch.Tensor:
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.decoder(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def load_pretrain(self, weights_path: str) -> None:
        """
        Method for pretrained model weights loading.
        :param weights_path: path to pretrained weights.
        """
        weights_data = torch.load(weights_path, map_location='cpu')
        weights = weights_data['model_state_dict'] if 'model_state_dict' in weights_data else weights_data['state_dict']
        encoder_weights = OrderedDict()
        decoder_weights = OrderedDict()
        cls_head_weights = OrderedDict()
        segm_head_weights = OrderedDict()
        for key, val in weights.items():
            new_key = key
            if key.startswith("model."):
                new_key = key.replace('model', '').strip('.')

            if new_key.startswith("encoder"):
                new_key = new_key[len("encoder."):]
                encoder_weights[new_key] = val

            if new_key.startswith("decoder"):
                new_key = new_key[len("decoder."):]
                decoder_weights[new_key] = val

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

        self.encoder.load_state_dict(encoder_weights, strict=True)
        self.decoder.load_state_dict(cls_head_weights, strict=False)
        if self.decoder is not None:
            self.decoder.load_state_dict(segm_head_weights, strict=False)
        if self.classification_head is not None:
            self.classification_head.load_state_dict(cls_head_weights, strict=False)

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

        new_weights.pop("criterion.cls_loss.ce_loss.pos_weight", None)
        self.load_state_dict(new_weights, strict=False)

    def freeze_model(self, freeze_configuration: str):
        assert freeze_configuration in ('encoder', 'all'), \
            f'Incorrect type of freeze {freeze_configuration}. Avaliable types: "encoder", "all".'
        if freeze_configuration == 'encoder':
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif freeze_configuration == 'all':
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False


def freeze_layers(model, layer_freeze=0.8):
    depth = len(list(model.parameters()))
    layer_train = int(depth * (1 - layer_freeze))
    if layer_train % 2:
        layer_train += 1

    print(f"Model depth: {depth}, Layers to train: {layer_train}")

    for i, parameter in enumerate(model.parameters()):
        if i < depth - layer_train:
            parameter.requires_grad = False

    return model


def print_info(text: str, type_info: bool = True):
    if type_info:
        rprint(f"[bold yellow][INFO][/bold yellow] [italic green]{text}[/italic green]")
    else:
        rprint(f"[bold orange_red1][WARNING][/bold orange_red1] [italic yellow]{text}[/italic yellow]")
