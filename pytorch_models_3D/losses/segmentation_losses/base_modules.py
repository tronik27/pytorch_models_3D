from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import math

import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch
import torch.nn.functional as F
import numpy as np

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


class SegmentationLoss(torch.nn.Module, ABC):

    def __init__(self, ignore_value: float, pos_weight: torch.Tensor, reduction: str):
        """
        Base segmentation loss function class
        Args:
            ignore_value: value to beignored when loss calculated.
            pos_weight:
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.pos_weight = pos_weight
        self.reduction = reduction

    @abstractmethod
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None):
        pass

    def filter_uncertain_annotation(self, data_tensor: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Method for ignoring uncertain annotated masks.

        Args:
            data_tensor (torch.Tensor): Data value tensor.
            gt_mask (torch.Tensor): Ground truth masks.

        Returns:
            torch.Tensor: Filtered loss tensor value.
        """
        ignore = (gt_mask != self.ignore_value).float()
        return data_tensor * ignore

    def add_weights(self, loss: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Method for weighting loss value.

        Args:
            loss (torch.Tensor): Loss value tensor.
            gt_mask (torch.Tensor): Ground truth masks.

        Returns:
            torch.Tensor: Weighted loss tensor value.
        """
        if self.pos_weight is not None:
            weight = self.pos_weight.repeat(
                gt_mask.shape[0], gt_mask.shape[2], gt_mask.shape[3], gt_mask.shape[4], 1
            ).permute((0, 4, 1, 2, 3))
            weight = weight.to(gt_mask.device)
            weight = weight * gt_mask + (1 - gt_mask)
            return loss * weight
        else:
            return loss

    def aggregate_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Method for loss value aggregation by loss tensor reduction.

        Args:
            loss (torch.Tensor): Loss value tensor.

        Returns:
            torch.Tensor: Reduced loss value tensor.
        """
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

    @staticmethod
    def roi_filtration(filtration_mask: torch.Tensor, data_tensor: torch.Tensor) -> torch.Tensor:
        """
        Method for filtering the data tensor based on a given filtration mask.

        Args:
            filtration_mask (torch.Tensor): A binary mask used for filtering. If None, no filtration is applied.
            data_tensor (torch.Tensor): The input data tensor to be filtered.

        Returns:
            torch.Tensor: The filtered data tensor. If filtration_mask is None, returns the original data_tensor.
        """
        if filtration_mask is not None:
            data_tensor = data_tensor * filtration_mask
        return data_tensor


class CrossEntropy(SegmentationLoss):
    def __init__(
            self,
            pos_weight=None,
            reduction='mean',
            from_logits=True,
            mode: str = 'binary',
            classes: List[int] = None,
            ignore_value: float = -1
    ) -> None:
        """
        Cross entropy segmentation loss class.

        Args:
            pos_weight (torch.Tensor, optional): A weight of positive examples. Must be a vector with length equal to the
                number of classes. Default: None
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of
                elements in the output, 'sum': the output will be summed. Default: 'mean'
            from_logits (bool): If True, assumes input is raw logits. If False, assumes input is probabilities. Default: True
            mode (str): Specifies the task type: 'binary' | 'multilabel' | 'multiclass'. Default: 'binary'
            ignore_value (float): Specifies a target value that is ignored and does not contribute to the input gradient.
                Default: -1
        """
        super().__init__(pos_weight=pos_weight, ignore_value=ignore_value, reduction=reduction)
        assert mode in ["binary", "multilabel", 'multiclass'], 'Incorrect task type!'
        self.mode = mode
        if self.mode == 'multiclass':
            if from_logits:
                self.ce = nn.CrossEntropyLoss(weight=self.pos_weight, reduction='none')
            else:
                raise NotImplementedError(
                    'CrossEntropy with from_logits=False is not implemented for multiclass mode!'
                )
        else:
            if from_logits:
                self.ce = nn.BCEWithLogitsLoss(reduction='none')
            else:
                self.ce = nn.BCELoss(reduction='none')
        if classes is not None:
            assert mode != "binary", "Masking num_classes is not supported with mode=binary"
            classes = self.__to_tensor(classes, dtype=torch.long)
        self.classes = classes

    @abstractmethod
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None):
        pass


class CustomLoss(SegmentationLoss, ABC, _Loss):
    def __init__(
            self,
            reduction: str,
            from_logits: bool,
            ignore_value: float,
            mode: str,
            classes: List[int],
            pos_weight: torch.Tensor,
            batchwise: bool = False
    ) -> None:
        """
        Abstract class for custom segmentation loss functions.
        Args:
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            from_logits: If True, assumes input is raw logits. If False, assumes input is probabilities.
            ignore_value: Specifies a target value that is ignored and does not contribute to the input gradient.
            mode: Specifies the task type: 'binary' | 'multilabel' | 'multiclass'.
            classes: List of classes that contribute in loss computation. By default, all channels are included.
        """
        super().__init__(reduction=reduction, ignore_value=ignore_value, pos_weight=pos_weight)
        assert mode in ["binary", "multi-binary", "multi-class"]
        self.mode = mode
        self.from_logits = from_logits
        if classes is not None:
            assert mode != "binary", "Masking num_classes is not supported with mode=binary"
            classes = self.__to_tensor(classes, dtype=torch.long)
        self.classes = classes
        self.batchwise = batchwise

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None):
        """
        Forward pass for computing the custom loss.

        Args:
            y_pred (torch.Tensor): The predicted segmentation map, shape (N, C, D, H, W).
            y_true (torch.Tensor): The ground truth segmentation map, shape (N, D, H, W) or (N, C, D, H, W).
            filtration_mask (torch.Tensor, optional):
            A binary mask for filtering regions of interest, shape (N, 1, D, H, W).

        Returns:
            torch.Tensor: The computed Dice loss.

        Note:
            - If `from_logits` is True, sigmoid is applied to `y_pred`.
            - The method handles binary, multi-binary, and multiclass segmentation.
            - Uncertain annotations and regions outside the filtration mask are handled as per the base class methods.
        """

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

        if self.mode == "multiclass":
            y_true = F.one_hot(y_true.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
            y_true = y_true.view(batch_size, num_classes, -1)
            y_pred = y_pred.view(batch_size, num_classes, -1)

        assert y_true.shape == y_pred.shape, \
            f"Got differing shapes for y_true ({y_true.shape}) and y_pred ({y_pred.shape})"

        y_true = self.filter_uncertain_annotation(data_tensor=y_true, gt_mask=y_true)
        y_true = self.add_weights(loss=y_true, gt_mask=y_true)
        y_true = self.roi_filtration(filtration_mask=filtration_mask, data_tensor=y_true)

        y_pred = self.filter_uncertain_annotation(data_tensor=y_pred, gt_mask=y_true)
        y_pred = self.add_weights(loss=y_pred, gt_mask=y_true)
        y_pred = self.roi_filtration(filtration_mask=filtration_mask, data_tensor=y_pred)

        loss = self.compute_loss(y_pred=y_pred, y_true=y_true.type_as(y_pred), dims=dims)

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss=loss, y_true=gt)

    @abstractmethod
    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
        pass


def to_tensor(x, dtype=None) -> torch.Tensor:
    """
    Converts input to PyTorch tensor.

    This method converts various input types (torch.Tensor, numpy.ndarray, list, tuple)
    to a PyTorch tensor. If a specific dtype is provided, the resulting tensor
    will be cast to that dtype.

    Args:
        x: Input data to be converted to tensor. Can be a torch.Tensor,
           numpy.ndarray, list, or tuple.
        dtype: Optional. The desired data type of the output tensor.

    Returns:
        torch.Tensor: The input data converted to a PyTorch tensor.

    Note:
        If the input is already a torch.Tensor and no dtype is specified,
        the input is returned unchanged.
    """
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


def focal_loss_with_logits(
    output: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    reduction: str = "mean",
    normalized: bool = False,
    reduced_threshold: Optional[float] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        output: Tensor of arbitrary shape (predictions of the model)
        target: Tensor of the same shape as input
        gamma: Focal loss power factor
        alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
            high values will give more weight to positive class.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).

    References:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type(output.type())

    logpt = F.binary_cross_entropy_with_logits(output, target, reduction="none")
    pt = torch.exp(-logpt)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt

    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)

    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss /= norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


def softmax_focal_loss_with_logits(
    output: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    reduction="mean",
    normalized=False,
    reduced_threshold: Optional[float] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Softmax version of focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        output: Tensor of shape [B, C, *] (Similar to nn.CrossEntropyLoss)
        target: Tensor of shape [B, *] (Similar to nn.CrossEntropyLoss)
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    """
    log_softmax = F.log_softmax(output, dim=1)

    loss = F.nll_loss(log_softmax, target, reduction="none")
    pt = torch.exp(-loss)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * loss

    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss = loss / norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


def soft_jaccard_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    """
    Compute Soft Jaccard score between two tensors.

    Args:
        output (torch.Tensor): A tensor of shape [B, C, *] representing the predicted probabilities.
        target (torch.Tensor): A tensor of shape [B, C, *] representing the ground truth.
        smooth (float, optional): Smoothing factor to avoid division by zero. Default: 0.0.
        eps (float, optional): Small constant to avoid numerical instability. Default: 1e-7.
        dims (tuple, optional): Dimensions to reduce over. If None, reduces over all dimensions. Default: None.

    Returns:
        torch.Tensor: Soft Jaccard score.

    Note:
        The Soft Jaccard score (also known as Intersection over Union) is a measure of the overlap between
        the prediction and the target. It ranges from 0 (no overlap) to 1 (perfect overlap).
    """
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)
    return jaccard_score


def soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    """
    Compute Soft Dice score between two tensors.

    Args:
        output (torch.Tensor): A tensor of shape [B, C, *] representing the predicted probabilities.
        target (torch.Tensor): A tensor of shape [B, C, *] representing the ground truth.
        smooth (float, optional): Smoothing factor to avoid division by zero. Default: 0.0.
        eps (float, optional): Small constant to avoid numerical instability. Default: 1e-7.
        dims (tuple, optional): Dimensions to reduce over. If None, reduces over all dimensions. Default: None.

    Returns:
        torch.Tensor: Soft Dice score.

    Note:
        The Soft Dice score is a measure of overlap between two samples. It ranges from 0 to 1,
        where a Dice score of 1 denotes perfect and complete overlap.
    """
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score


def soft_tversky_score(
    output: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    """
    Compute Soft Tversky score between two tensors.

    Args:
        output (torch.Tensor): A tensor of shape [B, C, *] representing the predicted probabilities.
        target (torch.Tensor): A tensor of shape [B, C, *] representing the ground truth.
        alpha (float): Weight constant that penalize model for FPs (False Positives)
        beta (float): Weight constant that penalize model for FNs (False Positives)
        smooth (float, optional): Smoothing factor to avoid division by zero. Default: 0.0.
        eps (float, optional): Small constant to avoid numerical instability. Default: 1e-7.
        dims (tuple, optional): Dimensions to reduce over. If None, reduces over all dimensions. Default: None.

    Returns:
        torch.Tensor: Soft Tversky score.

    Note:
        The Soft Tversky score is a measure of overlap between two samples. It ranges from 0 to 1,
        where a Tversky score of 1 denotes perfect and complete overlap.
    """
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)  # TP
        fp = torch.sum(output * (1.0 - target), dim=dims)
        fn = torch.sum((1 - output) * target, dim=dims)
    else:
        intersection = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1.0 - target))
        fn = torch.sum((1 - output) * target)

    tversky_score = (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth).clamp_min(eps)
    return tversky_score


def wing_loss(output: torch.Tensor, target: torch.Tensor, width=5, curvature=0.5, reduction="mean"):
    """Wing loss

    References:
        https://arxiv.org/pdf/1711.06753.pdf

    """
    diff_abs = (target - output).abs()
    loss = diff_abs.clone()

    idx_smaller = diff_abs < width
    idx_bigger = diff_abs >= width

    loss[idx_smaller] = width * torch.log(1 + diff_abs[idx_smaller] / curvature)

    C = width - width * math.log(1 + width / curvature)
    loss[idx_bigger] = loss[idx_bigger] - C

    if reduction == "sum":
        loss = loss.sum()

    if reduction == "mean":
        loss = loss.mean()

    return loss


def label_smoothed_nll_loss(
    lprobs: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
    ignore_index=None,
    dim=-1,
) -> torch.Tensor:
    """
    NLL loss with label smoothing

    References:
        https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py

    Args:
        lprobs (torch.Tensor): Log-probabilities of predictions (e.g after log_softmax)
        target (torch.Tensor): Target tensor
        epsilon (float): Smoothing parameter, default 0.1
        ignore_index (int, optional): Pad token, default None
        dim (int, optional): Dimension to reduce, default -1

    Returns:
        torch.Tensor: loss, scalar by default
    """

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, value=0)
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)

    eps_i = epsilon / lprobs.size(dim)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss
