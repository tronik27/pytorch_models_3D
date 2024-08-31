import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from torch.nn.modules.loss import _Loss

from base_modules import SegmentationLoss


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
             - **y_pred** - torch.Tensor of shape (N, C, D, H, W)
             - **y_true** - torch.Tensor of shape (N, D, H, W) or (N, C, D, H, W)
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
        """
        Forward pass for computing the Dice loss.

        Args:
            y_pred (torch.Tensor): The predicted segmentation map, shape (N, C, D, H, W).
            y_true (torch.Tensor): The ground truth segmentation map, shape (N, D, H, W) or (N, C, D, H, W).
            filtration_mask (torch.Tensor, optional): A binary mask for filtering regions of interest, shape (N, 1, D, H, W).

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

        assert y_true.shape == y_pred.shape, f"Got differing shapes for y_true ({y_true.shape}) and y_pred ({y_pred.shape})"

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
        Compute the Dice loss between predicted and true segmentation masks.

        Args:
            y_pred (torch.Tensor): Predicted segmentation mask.
            y_true (torch.Tensor): Ground truth segmentation mask.
            dims (int or Tuple[int, int]): Dimensions along which to compute the Dice loss.

        Returns:
            torch.Tensor: Computed Dice loss.

        Note:
            - The method handles smoothing to avoid division by zero.
            - It can compute log loss if specified during initialization.
            - The loss is clipped to valid ranges and NaN values are handled.
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
        """
        Convert input to PyTorch tensor.

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
        """
        Compute the Adaptive TvMF Dice Loss.

        This method calculates the loss using a cosine similarity-based approach,
        which is then adjusted by the kappa parameter to penalize false positives.

        Args:
            y_pred (torch.Tensor): The predicted segmentation map.
            y_true (torch.Tensor): The ground truth segmentation map.
            dims (int): The dimension along which to compute the loss.

        Returns:
            torch.Tensor: The computed loss value.
        """
        y_pred = nn.functional.normalize(y_pred, p=2, dim=dims)
        y_true = nn.functional.normalize(y_true, p=2, dim=dims)
        cosine = torch.sum(y_pred * y_true, dim=dims)
        intersect = (1. + cosine).div(1. + (1. - cosine).mul(self.kappa)) - 1.
        loss = (1 - intersect )**2.0
        return loss
