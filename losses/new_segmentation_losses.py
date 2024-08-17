import torch.nn as nn
import torch
from typing import List, Tuple
from torch.nn.modules.loss import _Loss
import numpy as np
from abc import ABC, abstractmethod


class SegmentationLoss(torch.nn.Module, ABC):
    """
    Base loss function class.
    """
    def __init__(self, ignore_value: float, pos_weight: torch.Tensor):
        super().__init__()
        self.ignore_value = ignore_value
        self.pos_weight = pos_weight

    @abstractmethod
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None):
        pass

    def filter_uncertain_annotation(self, data_tensor: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Method for ignoring uncertain annotated masks.
        :param data_tensor: data value tensor.
        :param gt_mask: ground truth masks.
        :return filtered loss tensor value.
        """
        ignore = (gt_mask != self.ignore_value).float()
        return data_tensor * ignore

    def ignore(self, y_true: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        """
        Method for ignoring uncertain annotated masks.
        :param loss: calculated loss output.
        :param y_true: ground truth.
        :return filtered loss tensor value.
        """
        y_true = torch.mean(y_true, dim=(2, 3))
        y_true = torch.flatten(y_true)
        if len(loss.size()) == 4:
            loss = torch.mean(loss, dim=(2, 3))
        if len(loss.size()) == 3:
            loss = torch.mean(loss, dim=2)
        
        loss = torch.flatten(loss)
        indices = y_true != self.ignore_value
        filtered_loss = loss[indices]
        if not filtered_loss.numel():
            filtered_loss = torch.mean(loss)
        return filtered_loss

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
            # weight = torch.ones(size=())
            weight = weight.to(gt_mask.device)
            weight = weight * gt_mask + (1 - gt_mask)
            return loss * weight
        else:
            return loss

    def aggregate_loss(self, loss: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Method for loss value aggregation by loss tensor reduction.
        :param loss: loss value tensor.
        :param y_true: ground truth tensor.
        return reduced loss value tensor.
        """
        loss = self.ignore(y_true=y_true, loss=loss)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

    @staticmethod
    def roi_filtration(filtration_mask: torch.Tensor, data_tensor: torch.Tensor) -> torch.Tensor:
        if filtration_mask is not None:
            data_tensor = data_tensor * filtration_mask
        return data_tensor


class CrossEntropy(SegmentationLoss):
    def __init__(
            self, pos_weight=None, reduction='mean', from_logits=True, mode: str = 'binary', ignore_value: float = -1
    ) -> None:
        """
        Cross entropy segmentation loss class.
        """
        super().__init__(pos_weight=pos_weight, ignore_value=ignore_value)
        self.reduction = reduction
        assert mode in ["binary", "multi-binary"], 'Incorrect task type!'
        self.mode = mode
        if from_logits:
            self.bce = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.bce = nn.BCELoss(reduction='none')

    @abstractmethod
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None):
        pass


class MaskBCE(CrossEntropy):
    """
    Binary cross entropy segmentation loss class.
    """
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Method for loss value calculation.
        :param y_pred: model predicted output.
        :param y_true: ground truth.
        :param filtration_mask: area of interest masks
        """
        loss = self.bce(y_pred, y_true)
        loss = self.filter_uncertain_annotation(data_tensor=loss, gt_mask=y_true)
        loss = self.add_weights(loss=loss, gt_mask=y_true)
        loss = self.roi_filtration(data_tensor=loss, filtration_mask=filtration_mask)
        return self.aggregate_loss(loss=loss, y_true=y_true)


class MaskFocal(CrossEntropy):

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
        Focal binary cross entropy segmentation loss class.
        """
        super().__init__(
            reduction=reduction, pos_weight=pos_weight, mode=mode, ignore_value=ignore_value, from_logits=from_logits
        )
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Method for focal loss value calculation.
        :param y_pred: model predicted output.
        :param y_true: ground truth.
        :param filtration_mask: area of interest masks
        """
        cent_loss = self.bce(y_pred, y_true)
        pt = torch.exp(-cent_loss)
        loss = (1 - pt) ** self.gamma * cent_loss
        loss = self.filter_uncertain_annotation(data_tensor=loss, gt_mask=y_true)
        loss = self.add_weights(loss=loss, gt_mask=y_true)
        loss = self.roi_filtration(data_tensor=loss, filtration_mask=filtration_mask)
        return self.aggregate_loss(loss=loss, y_true=y_true)


class WeightedLoss(CrossEntropy):
    """
    Combined BCE and IoU weighted by gt mask loss function.
    """
    def __init__(
            self,
            reduction='mean',
            from_logits=True,
            mode: str = 'binary',
            ignore_value = -1,
            batchwise=False,
    ) -> None:
        """
        Focal binary cross entropy segmentation loss class.
        """
        super().__init__(
            reduction=reduction, mode=mode, ignore_value=ignore_value, from_logits=from_logits
        )
        self.batchwise = batchwise
        self.from_logits = from_logits

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Method for focal loss value calculation.
        :param y_pred: model predicted output.
        :param y_true: ground truth.
        :param filtration_mask: area of interest masks
        """
        batch_size = y_true.size(0)
        num_classes = y_pred.size(1)
        gt = y_true.clone()
        y_true = self.add_weights(loss=y_true, gt_mask=y_true)
        y_pred = self.add_weights(loss=y_pred, gt_mask=y_true)

        dims = (0, 2) if self.batchwise else 2

        if self.mode == "binary":
            y_true = y_true.view(batch_size, 1, -1)
            y_pred = y_pred.view(batch_size, 1, -1)

        if self.mode == "multi-binary":
            y_true = y_true.view(batch_size, num_classes, -1)
            y_pred = y_pred.view(batch_size, num_classes, -1)

        y_true = self.filter_uncertain_annotation(data_tensor=y_true, gt_mask=y_true)
        y_true = self.roi_filtration(filtration_mask=filtration_mask, data_tensor=y_true)

        y_pred = self.filter_uncertain_annotation(data_tensor=y_pred, gt_mask=y_true)
        y_pred = self.roi_filtration(filtration_mask=filtration_mask, data_tensor=y_pred)

        weight = 1 + 5 * torch.abs(nn.functional.avg_pool2d(y_true, kernel_size=31, stride=1, padding=15) - y_true)
        weighted_bce = self.__bce(y_true=y_true, y_pred=y_pred, weight=weight, dims=dims)
        weighted_iou = self.__iou(y_true=y_true, y_pred=y_pred, weight=weight, dims=dims)
        loss = weighted_iou + weighted_bce

        return self.aggregate_loss(loss=loss, y_true=gt)

    def __bce(
            self, y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor, dims: int or Tuple[int, int]
    ) -> torch.Tensor:
        """
        Method for BCE calculation
        """
        bce = self.bce(y_pred, y_true)
        weighted_bce = (weight * bce).sum(dim=dims) / weight.sum(dim=dims)
        return weighted_bce

    def __iou(
            self, y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor, dims: int or Tuple[int, int]
    ) -> torch.Tensor:
        """
        Method for iou calculation
        """
        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)
        y_pred = torch.sigmoid(y_pred)
        inter = ((y_pred * y_true) * weight).sum(dim=dims)
        union = ((y_pred + y_true) * weight).sum(dim=dims)
        weighted_iou = 1 - (inter + 1) / (union - inter + 1)
        return weighted_iou


class DiceLoss(SegmentationLoss, _Loss):

    def __init__(
        self,
        mode,
        pos_weight: torch.Tensor = None,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_mask=False,
        ignore_value=None,
        batchwise=False,
        eps=1e-7,
    ):
        """
        Implementation of Dice loss for image segmentation task. It supports binary and multilabel cases.
        Args:
            mode: Loss mode 'binary' or 'multi-binary';
            classes:  List of num_classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes x is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_value: Label value that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
        """
        assert mode in {"binary", "multi-binary"}
        super().__init__(pos_weight=pos_weight, ignore_value=ignore_value)
        self.mode = mode
        if classes is not None:
            assert mode != "binary", "Masking num_classes is not supported with mode=binary"
            classes = self.__to_tensor(classes, dtype=torch.long)
        self.ignore_mask = ignore_mask
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.bacthwise = batchwise

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None):

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)

        batch_size = y_true.size(0)
        num_classes = y_pred.size(1)
        gt = y_true.clone()

        dims = (0, 2) if self.bacthwise else 2

        if self.mode == "binary":
            y_true = y_true.view(batch_size, 1, -1)
            y_pred = y_pred.view(batch_size, 1, -1)

        if self.mode == "multi-binary":
            y_true = y_true.view(batch_size, num_classes, -1)
            y_pred = y_pred.view(batch_size, num_classes, -1)

        y_true = self.filter_uncertain_annotation(data_tensor=y_true, gt_mask=y_true)
        y_true = self.add_weights(loss=y_true, gt_mask=y_true)
        y_true = self.roi_filtration(filtration_mask=filtration_mask, data_tensor=y_true)

        y_pred = self.filter_uncertain_annotation(data_tensor=y_pred, gt_mask=y_true)
        y_pred = self.add_weights(loss=y_pred, gt_mask=y_true)
        y_pred = self.roi_filtration(filtration_mask=filtration_mask, data_tensor=y_pred)

        loss = self.compute_loss(y_pred=y_pred, y_true=y_true.type_as(y_pred), dims=dims)

        if self.ignore_mask:
            mask = y_true.clip(min=0, max=1).sum(dims) > 0
            loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss=loss, y_true=gt)

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, dims: int or Tuple[int, int]) -> torch.Tensor:
        """
        Method for dice score computation.
        :param y_pred:
        :param y_true:
        :param dims:
        :return tensor with computed dice loss value.
        """
        intersection = y_pred * y_true
        cardinality = (y_pred + y_true)

        intersection = torch.sum(intersection, dim=dims)
        cardinality = torch.sum(cardinality, dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth).clamp_min(self.eps)
        
        dice_score[dice_score > 1] = cardinality[dice_score > 1]
        dice_score = torch.nan_to_num(dice_score)

        if self.log_loss:
            loss = -torch.log(dice_score.clamp_min(self.eps))
        else:
            loss = 1.0 - dice_score
            
        return loss

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


class AdaptiveTvMFDiceLoss(DiceLoss):

    def __init__(
        self,
        mode: str,
        classes: List[int] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        pos_weight: torch.Tensor = None,
        ignore_value: float = -1,
        ignore_mask: bool = False,
        batchwise=False,
        eps: float = 1e-7,
        kappa: float = 1.,
    ):
        """Adaptive dice loss for image segmentation task.
        Where TP and FP is weighted by alpha and beta params.
        With alpha == beta == 0.5, this loss becomes equal DiceLoss.
        It supports binary, multiclass and multilabel cases
        Args:
            mode: Loss mode 'binary' or 'multi-binary';
            classes: Optional list of num_classes that contribute in loss computation;
            By default, all channels are included.
            log_loss: If True, loss computed as ``-log(tversky)`` otherwise ``1 - tversky``
            from_logits: If True assumes x is raw logits
            smooth:
            ignore_value: Label that indicates ignored pixels (does not contribute to loss)
            eps: Small epsilon for numerical stability
            kappa: Weight constant that penalize model for FPs (False Positives)
        Return:
            loss: torch.Tensor
        """
        super().__init__(
            mode=mode, classes=classes, log_loss=log_loss, from_logits=from_logits, ignore_mask=ignore_mask,
            smooth=smooth, ignore_value=ignore_value, batchwise=batchwise, eps=eps, pos_weight=pos_weight
        )
        self.kappa = kappa
        if self.bacthwise:
            raise NotImplementedError('AdaptiveTvMFDiceLoss implemented only for imagewise loss calculation!')
        
    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, dims: int) -> torch.Tensor:
        y_pred = nn.functional.normalize(y_pred, p=2, dim=dims)
        y_true = nn.functional.normalize(y_true, p=2, dim=dims)
        cosine = torch.sum(y_pred * y_true, dim=dims)
        intersect = (1. + cosine).div(1. + (1. - cosine).mul(self.kappa)) - 1.
        loss = (1 - intersect)**2.0
        return loss


class TverskyLoss(DiceLoss):
    def __init__(
        self,
        mode: str,
        classes: List[int] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        pos_weight: torch.Tensor = None,
        ignore_value: float = -1,
        ignore_mask: bool = False,
        batchwise=False,
        eps: float = 1e-7,
        alpha: float = 0.5,
        beta: float = 0.5
    ):
        """Tversky loss for image segmentation task.
        Where TP and FP is weighted by alpha and beta params.
        With alpha == beta == 0.5, this loss becomes equal DiceLoss.
        It supports binary, multiclass and multilabel cases
        Args:
            mode: Loss mode 'binary' or 'multi-binary';
            classes: Optional list of num_classes that contribute in loss computation;
            By default, all channels are included.
            log_loss: If True, loss computed as ``-log(tversky)`` otherwise ``1 - tversky``
            from_logits: If True assumes x is raw logits
            smooth:
            ignore_value: Label that indicates ignored pixels (does not contribute to loss)
            eps: Small epsilon for numerical stability
            alpha: Weight constant that penalize model for FPs (False Positives)
            beta: Weight constant that penalize model for FNs (False Positives)
        Return:
            loss: torch.Tensor
        """
        super().__init__(
            mode=mode, classes=classes, log_loss=log_loss, from_logits=from_logits, ignore_mask=ignore_mask,
            smooth=smooth, ignore_value=ignore_value, batchwise=batchwise, eps=eps, pos_weight=pos_weight
        )
        self.alpha = alpha
        self.beta = beta

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, dims: int or Tuple[int, int]) -> torch.Tensor:

        intersection = y_pred * y_true
        fp = y_pred * (1.0 - y_true)
        fn = (1 - y_pred) * y_true

        intersection = torch.sum(intersection, dim=dims)
        fp = torch.sum(fp, dim=dims)
        fn = torch.sum(fn, dim=dims)
        
        tversky_score = (intersection + self.smooth) / (
                    intersection + self.alpha * fp + self.beta * fn + self.smooth
            ).clamp_min(self.eps)
        tversky_score[tversky_score > 1] = intersection[tversky_score > 1]
        tversky_score = torch.nan_to_num(tversky_score)

        if self.log_loss:
            loss = -torch.log(tversky_score.clamp_min(self.eps))
        else:
            loss = 1.0 - tversky_score

        return loss


class FocalTverskyLoss(DiceLoss):
    def __init__(
        self,
        mode: str,
        classes: List[int] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        pos_weight: torch.Tensor = None,
        ignore_value: float = -1,
        ignore_mask: bool = False,
        batchwise=False,
        eps: float = 1e-7,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
    ):
        """Tversky loss for image segmentation task.
        Where TP and FP is weighted by alpha and beta params.
        With alpha == beta == 0.5, this loss becomes equal DiceLoss.
        It supports binary, multiclass and multilabel cases
        Args:
            mode: Loss mode 'binary' or 'multi-binary';
            classes: Optional list of num_classes that contribute in loss computation;
            By default, all channels are included.
            log_loss: If True, loss computed as ``-log(tversky)`` otherwise ``1 - tversky``
            from_logits: If True assumes x is raw logits
            smooth:
            ignore_value: Label that indicates ignored pixels (does not contribute to loss)
            eps: Small epsilon for numerical stability
            alpha: Weight constant that penalize model for FPs (False Positives)
            beta: Weight constant that penalize model for FNs (False Positives)
            gamma: Constant that squares the error function. Defaults to ``1.0``
        Return:
            loss: torch.Tensor
        """
        super().__init__(
            mode=mode, classes=classes, log_loss=log_loss, from_logits=from_logits, ignore_mask=ignore_mask,
            smooth=smooth, ignore_value=ignore_value, batchwise=batchwise, eps=eps, pos_weight=pos_weight
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, dims: int or Tuple[int, int]) -> torch.Tensor:
        """
        Method for focal tversky loss function value computation.
        """
        intersection = y_pred * y_true
        fp = y_pred * (1.0 - y_true)
        fn = (1 - y_pred) * y_true

        intersection = torch.sum(intersection, dim=dims)
        fp = torch.sum(fp, dim=dims)
        fn = torch.sum(fn, dim=dims)

        tversky_score = (intersection + self.smooth) / (
                intersection + self.alpha * fp + self.beta * fn + self.smooth
        ).clamp_min(self.eps)
        tversky_score[tversky_score > 1] = intersection[tversky_score > 1]
        tversky_score = torch.nan_to_num(tversky_score)

        if self.log_loss:
            loss = -torch.log(tversky_score.clamp_min(self.eps))
        else:
            loss = 1.0 - tversky_score

        loss = loss ** self.gamma
        return loss
