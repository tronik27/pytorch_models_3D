from abc import ABC, abstractmethod
from typing import Optional

import torch.nn as nn
import torch
from typing import List, Tuple
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import numpy as np

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


class SegmentationLoss(torch.nn.Module, ABC):

    def __init__(self, ignore_value: float, pos_weight: torch.Tensor):
        """
        Base segmentation loss function class
        Args:
            ignore_value:
            pos_weight:
        """
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
    """Binary cross entropy segmentation loss class."""
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
        Args:
            pos_weight:
            reduction:
            from_logits:
            mode:
            ignore_value:
            gamma:
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
        Method for weighted  BCE calculation
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


class MCCLoss(SegmentationLoss):
    def __init__(self, eps: float = 1e-5):
        """Compute Matthews Correlation Coefficient Loss for image segmentation task.
        It only supports binary mode.

        Args:
            eps (float): Small epsilon to handle situations where all the samples in the dataset belong to one class

        Reference:
            https://github.com/kakumarabhishek/MCC-Loss
        """
        super().__init__()
        self.eps = eps

    def forward(
            self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute MCC loss

        Args:
            y_pred (torch.Tensor): model prediction of shape (N, H, W) or (N, 1, H, W)
            y_true (torch.Tensor): ground truth labels of shape (N, H, W) or (N, 1, H, W)
            filtration_mask (torch.Tensor): ground truth labels of shape (N, H, W) or (N, 1, H, W)

        Returns:
            torch.Tensor: loss value (1 - mcc)
        """

        bs = y_true.shape[0]

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        tp = torch.sum(torch.mul(y_pred, y_true)) + self.eps
        tn = torch.sum(torch.mul((1 - y_pred), (1 - y_true))) + self.eps
        fp = torch.sum(torch.mul(y_pred, (1 - y_true))) + self.eps
        fn = torch.sum(torch.mul((1 - y_pred), y_true)) + self.eps

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(torch.add(tp, fp) * torch.add(tp, fn) * torch.add(tn, fp) * torch.add(tn, fn))

        mcc = torch.div(numerator.sum(), denominator.sum())
        loss = 1.0 - mcc

        return loss


class LovaszLoss(SegmentationLoss):
    def __init__(
        self,
        mode: str,
        per_image: bool = False,
        ignore_index: Optional[int] = None,
        from_logits: bool = True,
    ):
        """Lovasz loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            per_image: If True loss computed per each image and then averaged, else computed per whole batch

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in ["binary", "multi-binary", "multiclass"], 'Incorrect task type!'
        super().__init__()

        self.mode = mode
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None):

        if self.mode in ["binary", "multi-binary"]:
            loss = _lovasz_hinge(y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index)
        elif self.mode == "multiclass":
            y_pred = y_pred.softmax(dim=1)
            loss = _lovasz_softmax(y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index)
        else:
            raise ValueError("Wrong mode {}.".format(self.mode))
        return loss


def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
        logits: [B, H, W] Logits at each pixel (between -infinity and +infinity)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
    """
    if per_image:
        loss = mean(
            _lovasz_hinge_flat(*_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
            for log, lab in zip(logits, labels)
        )
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore))
    return loss


def _lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss
    Args:
        logits: [P] Logits at each prediction (between -infinity and +infinity)
        labels: [P] Tensor, binary ground truth labels (0 or 1)
        ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def _flatten_binary_scores(scores, labels, ignore=None):
    """Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


# --------------------------- MULTICLASS LOSSES ---------------------------


def _lovasz_softmax(probas, labels, classes="present", per_image=False, ignore=None):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [B, C, H, W] Class probabilities at each prediction (between 0 and 1).
        Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        @param labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        @param per_image: compute the loss per image instead of per batch
        @param ignore: void class labels
    """
    if per_image:
        loss = mean(
            _lovasz_softmax_flat(*_flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
            for prob, lab in zip(probas, labels)
        )
    else:
        loss = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def _lovasz_softmax_flat(probas, labels, classes="present"):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [P, C] Class probabilities at each prediction (between 0 and 1)
        @param labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        fg = (labels == c).type_as(probas)  # foreground for class c
        if classes == "present" and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError("Sigmoid output possible only with 1 class")
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    return mean(losses)


def _flatten_probas(probas, labels, ignore=None):
    """Flattens predictions in the batch"""
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)

    C = probas.size(1)
    probas = torch.movedim(probas, 1, -1)  # [B, C, Di, Dj, ...] -> [B, Di, Dj, ..., C]
    probas = probas.contiguous().view(-1, C)  # [P, C]

    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid]
    vlabels = labels[valid]
    return vprobas, vlabels


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    """Nanmean compatible with generators."""
    values = iter(values)
    if ignore_nan:
        values = ifilterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n