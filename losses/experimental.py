import torch.nn as nn
import torch
from typing import Optional, List
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
from abc import ABC, abstractmethod


class SegmentationTorchLoss(nn.Module):

    def __init__(
            self, pos_weight=None, reduction='mean', from_logits=True, mode: str = 'binary', ignore_value=-1
    ) -> None:
        """
        Binary cross entropy segmentation loss class.
        """
        super().__init__()
        self.reduction = reduction
        self.pos_weight = pos_weight
        assert mode in ["binary", "multi", "multi-binary"], 'Incorrect task type!'
        self.mode = mode
        if mode in {'binary', 'multi-binary'}:
            self.bce = nn.BCEWithLogitsLoss(reduction='none') if from_logits else nn.BCELoss(reduction='none')
        else:
            if not from_logits:
                raise NotImplementedError('Multiclass cross-entropy loss implemented only for logits x!')
            self.bce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_value)
        self.ignore_value = ignore_value


    @abstractmethod
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None):
        pass

    def filter_uncertain_annotation(self, loss: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Method for ignoring uncertain annotated masks.
        :param loss: loss value tensor.
        :param gt_mask: ground truth masks.
        :return filtered loss tensor value.
        """
        ignore = (gt_mask != self.ignore_value).float()
        return loss * ignore

    def add_weights(self, loss: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Method for weighting loss value.
        :param loss: loss value tensor.
        :param gt_mask: ground truth masks.
        :return weighted loss tensor value.
        """
        if self.pos_weight is not None:
            weight = self.pos_weight.repeat(
                gt_mask.shape[0], gt_mask.shape[2], gt_mask.shape[3], 1
            ).permute((0, 3, 1, 2))
            weight = weight.to(gt_mask.device)
            weight = weight * gt_mask + (1 - gt_mask)
            return loss * weight
        else:
            return loss


class MaskBCE(SegmentationTorchLoss):

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Method for loss value calculation.
        :param y_pred: model predicted output.
        :param y_true: ground truth.
        :param filtration_mask: area of interest masks
        """
        print(y_pred.size(), y_true.size())
        loss = self.bce(y_pred, y_true)
        if self.mode in {'binary', 'multi-binary'}:
            loss = self.filter_uncertain_annotation(loss=loss, gt_mask=y_true)
            loss = self.add_weights(loss=loss, gt_mask=y_true)
            if filtration_mask is not None:
                loss = loss * filtration_mask

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss


class MaskFocal(SegmentationTorchLoss):

    def __init__(
            self,
            pos_weight=None,
            reduction='mean',
            from_logits=True,
            mode: str = 'binary',
            ignore_value = -1,
            gamma=2,
    ) -> None:
        """
        Binary cross entropy segmentation loss class.
        """
        super().__init__(
            reduction=reduction, pos_weight=pos_weight, mode=mode,ignore_value=ignore_value, from_logits=from_logits
        )
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Method for loss value calculation.
        :param y_pred: model predicted output.
        :param y_true: ground truth.
        :param filtration_mask: area of interest masks
        """
        cent_loss = self.bce(y_pred, y_true)
        pt = torch.exp(-cent_loss)
        loss = (1 - pt) ** self.gamma * cent_loss

        if self.mode in {'binary', 'multi-binary'}:
            loss = self.filter_uncertain_annotation(loss=loss, gt_mask=y_true)
            loss = self.add_weights(loss=loss, gt_mask=y_true)
            if filtration_mask is not None:
                loss = loss * filtration_mask

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss



class DiceLoss(_Loss):

    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        ignore_mask=False,
        batchwise=False,
        eps=1e-7,
    ):
        """Implementation of Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases
        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of num_classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes x is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {"binary", "multi", "multi-binary"}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != "binary", "Masking num_classes is not supported with mode=binary"
            classes = self.__to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index
        self.ignore_mask = ignore_mask
        self.bacthwise = batchwise

    def forward(self, y_pred: torch.tensor, y_true: torch.tensor):

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable clinic_score and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == "multi":
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = torch.sigmoid(y_pred)

        bs = y_true.size(0)
        num_classes = y_pred.size(1)

        dims = (0, 2) if self.bacthwise else 2

        if self.mode == "binary":
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        # if self.mode == "multi":
        #     y_true = y_true.view(bs, -1)
        #     y_pred = y_pred.view(bs, num_classes, -1)
        #
        #     if self.ignore_index is not None:
        #         masks = y_true != self.ignore_index
        #         y_pred = y_pred * masks.unsqueeze(1)
        #
        #         y_true = func.one_hot((y_true * masks).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
        #         y_true = y_true.permute(0, 2, 1) * masks.unsqueeze(1)  # H, C, H*W
        #     else:
        #         y_true = func.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
        #         y_true = y_true.permute(0, 2, 1)  # H, C, H*W

            if self.mode in ["multi-binary", "multi"]:
                y_true = y_true.view(bs, num_classes, -1)
                y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(output=y_pred, target=y_true.type_as(y_pred), dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty num_classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        if self.ignore_mask:
            mask = y_true.clip(0, 1).sum(dims) > 0
            loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()

    def compute_score(self, output: torch.Tensor, target: torch.Tensor, dims=None) -> torch.Tensor:

        assert output.size() == target.size()
        ignore = (target != -1.0).float()
        intersection = output * target * ignore
        cardinality = (output + target) * ignore

        if dims is not None:
            intersection = torch.sum(intersection, dim=dims)
            cardinality = torch.sum(cardinality, dim=dims)
        else:
            intersection = torch.sum(intersection)
            cardinality = torch.sum(cardinality)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth).clamp_min(self.eps)
        return dice_score

    @staticmethod
    def __to_tensor(x, dtype=None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            if dtype is not None:
                x = x.type(dtype)
            return x
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            if dtype is not None:
                x = x.type(dtype)
            return x
        if isinstance(x, (list, tuple)):
            x = np.array(x)
            x = torch.from_numpy(x)
            if dtype is not None:
                x = x.type(dtype)
            return x


class TverskyLoss(DiceLoss):
    """Tversky loss for image segmentation task.
    Where TP and FP is weighted by alpha and beta params.
    With alpha == beta == 0.5, this loss becomes equal DiceLoss.
    It supports binary, multiclass and multilabel cases
    Args:
        mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        classes: Optional list of num_classes that contribute in loss computation;
        By default, all channels are included.
        log_loss: If True, loss computed as ``-log(tversky)`` otherwise ``1 - tversky``
        from_logits: If True assumes x is raw logits
        smooth:
        ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        eps: Small epsilon for numerical stability
        alpha: Weight constant that penalize model for FPs (False Positives)
        beta: Weight constant that penalize model for FNs (False Positives)
        gamma: Constant that squares the error function. Defaults to ``1.0``
    Return:
        loss: torch.Tensor
    """

    def __init__(
        self,
        mode: str,
        classes: List[int] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        ignore_mask=False,
        batchwise=False,
        eps: float = 1e-7,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
    ):
        super().__init__(mode, classes, log_loss, from_logits, smooth, ignore_index, ignore_mask, batchwise, eps)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def aggregate_loss(self, loss):
        return loss.mean() ** self.gamma

    def compute_score(self, output: torch.Tensor, target: torch.Tensor, dims=None) -> torch.Tensor:
        assert output.size() == target.size()
        ignore = (target != -1.0).float()
        intersection = output * target * ignore
        fp = output * (1.0 - target) * ignore
        fn = (1 - output) * target * ignore

        if dims is not None:
            intersection = torch.sum(intersection, dim=dims)  # TP
            fp = torch.sum(fp, dim=dims)
            fn = torch.sum(fn, dim=dims)
        else:
            intersection = torch.sum(intersection)  # TP
            fp = torch.sum(fp)
            fn = torch.sum(fn)

        tversky_score = (intersection + self.smooth) / (intersection + self.alpha * fp + self.beta * fn + self.smooth).clamp_min(self.eps)
        return tversky_score
