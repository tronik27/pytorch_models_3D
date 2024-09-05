import torch
import torch.nn as nn
from typing import List, Tuple

from base_modules import CustomLoss


class DiceLoss(CustomLoss):

    def __init__(
            self,
            mode,
            reduction: str = 'mean',
            pos_weight: torch.Tensor = None,
            classes=None,
            log_loss=False,
            from_logits=True,
            smooth=0.0,
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
        super().__init__(
            from_logits=from_logits,
            pos_weight=pos_weight,
            ignore_value=ignore_value,
            reduction=reduction,
            classes=classes,
            mode=mode,
            batchwise=batchwise
        )
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

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
            mode=mode, classes=classes, log_loss=log_loss, from_logits=from_logits,
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
        loss = (1 - intersect)**2.0
        return loss
