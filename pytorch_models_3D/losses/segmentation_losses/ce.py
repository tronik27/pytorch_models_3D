from typing import Tuple, Optional, List
import torch
from torch import nn
import torch.nn.functional as F

from base_modules import CrossEntropy, CustomLoss, label_smoothed_nll_loss


class SegmentationCE(CrossEntropy):
    """Cross entropy segmentation loss class."""
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Method for loss value calculation.

        Args:
            y_pred (torch.Tensor): Predicted segmentation map, shape (N, C, D, H, W).
            y_true (torch.Tensor): Ground truth segmentation map, shape (N, D, H, W) or (N, C, D, H, W).
            filtration_mask (torch.Tensor, optional): A binary mask for filtering regions of interest,
                shape (N, 1, D, H, W).

        Returns:
            torch.Tensor: Computed segmentation loss value.
        """
        loss = self.ce(y_pred, y_true)
        loss = self.filter_uncertain_annotation(data_tensor=loss, gt_mask=y_true)
        loss = self.add_weights(loss=loss, gt_mask=y_true)
        loss = self.roi_filtration(data_tensor=loss, filtration_mask=filtration_mask)
        if self.classes is not None:
            loss = loss[self.classes]
        return self.aggregate_loss(loss=loss)


class Focal(CrossEntropy):

    def __init__(
            self,
            pos_weight=None,
            reduction='mean',
            from_logits=True,
            mode: str = 'binary',
            ignore_value=-1,
            gamma=2,
            classes: List[int] = None
    ) -> None:
        """
        Focal cross entropy segmentation loss class.

        This class implements the focal loss, which is designed to address class imbalance issues
        in segmentation tasks by down-weighting well-classified examples.

        Args:
            pos_weight (torch.Tensor, optional): A weight of positive examples. Must be a vector with length
                equal to the number of classes. Default: None
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                'none': no reduction will be applied, 'mean': the sum of the output will be divided by
                the number of elements in the output, 'sum': the output will be summed. Default: 'mean'
            from_logits (bool): If True, assumes input is raw logits. If False, assumes input is probabilities.
                Default: True
            mode (str): Specifies the task type: 'binary' | 'multilabel' | 'multiclass'. Default: 'binary'
            ignore_value (float): Specifies a target value that is ignored and does not contribute to the
                input gradient. Default: -1
            gamma (float): Focusing parameter for focal loss. Higher gamma increases focus on hard examples.
                Default: 2

        Note:
            The focal loss is defined as: FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
            where pt is the model's estimated probability for the target class.
        """
        super().__init__(
            reduction=reduction,
            pos_weight=pos_weight,
            mode=mode,
            ignore_value=ignore_value,
            from_logits=from_logits,
            classes=classes
        )
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate the focal loss value.

        This method computes the focal loss, which is designed to address class imbalance
        by down-weighting well-classified examples.

        Args:
            y_pred (torch.Tensor): The model's predicted output. Shape should be (N, C, ...) where N is
                                   the batch size and C is the number of classes.
            y_true (torch.Tensor): The ground truth labels. Shape should match y_pred or be (N, ...) for
                                   class indices.
            filtration_mask (torch.Tensor, optional): Mask for filtering regions of interest. Should have
                                                      the same spatial dimensions as y_pred and y_true.

        Returns:
            torch.Tensor: The computed focal loss.

        Note:
            - The focal loss is defined as: FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
              where pt is the model's estimated probability for the target class.
            - This implementation applies additional processing steps such as filtering
              uncertain annotations and region of interest filtration.
        """
        cent_loss = self.ce(y_pred, y_true)
        pt = torch.exp(-cent_loss)
        loss = (1 - pt) ** self.gamma * cent_loss
        loss = self.filter_uncertain_annotation(data_tensor=loss, gt_mask=y_true)
        loss = self.add_weights(loss=loss, gt_mask=y_true)
        loss = self.roi_filtration(data_tensor=loss, filtration_mask=filtration_mask)
        if self.classes is not None:
            loss = loss[self.classes]
        return self.aggregate_loss(loss=loss)


class WeightedCE(CustomLoss, CrossEntropy):

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, dims: int or Tuple[int, int]) -> torch.Tensor:
        """
        Compute the weighted Cross-Entropy and IoU losses.

        Args:
            y_pred (torch.Tensor): Predicted segmentation map.
            y_true (torch.Tensor): Ground truth segmentation map.
            dims (int or Tuple[int, int]): Dimensions along which to compute the loss.

        Returns:
            torch.Tensor: Computed weighted Cross-Entropy and IoU losses.

        Note:
            - The weighting is applied to CE using IoU.
            - This loss is particularly useful for segmentation tasks with imbalanced classes or small objects.
        """
        weight = 1 + 5 * torch.abs(nn.functional.avg_pool3d(y_true, kernel_size=31, stride=1, padding=15) - y_true)
        weighted_bce = self.__ce(y_true=y_true, y_pred=y_pred, weight=weight, dims=dims)
        weighted_iou = self.__iou(y_true=y_true, y_pred=y_pred, weight=weight, dims=dims)
        loss = weighted_iou + weighted_bce
        return loss

    def __ce(
            self, y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor, dims: int or Tuple[int, int]
    ) -> torch.Tensor:
        """
        Compute the weighted Cross Entropy (CE) loss.

        Args:
            y_pred (torch.Tensor): Predicted values, shape (N, C, *).
            y_true (torch.Tensor): Ground truth values, shape (N, C, *).
            weight (torch.Tensor): Weight tensor for each sample, shape (N, C, *).
            dims (int or Tuple[int, ...]): Dimensions to reduce when computing the loss.

        Returns:
            torch.Tensor: Weighted BCE loss.

        Note:
            This method calculates the CE loss and applies the provided weights.
            The weighted loss is then summed over the specified dimensions and
            normalized by the sum of weights.
        """
        bce = self.ce(y_pred, y_true)
        weighted_bce = (weight * bce).sum(dim=dims) / weight.sum(dim=dims)
        return weighted_bce

    def __iou(
            self, y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor, dims: int or Tuple[int, int]
    ) -> torch.Tensor:
        """
        Compute the weighted Intersection over Union (IoU) loss.

        Args:
            y_pred (torch.Tensor): Predicted values, shape (N, C, *).
            y_true (torch.Tensor): Ground truth values, shape (N, C, *).
            weight (torch.Tensor): Weight tensor for each sample, shape (N, C, *).
            dims (int or Tuple[int, ...]): Dimensions to reduce when computing the loss.

        Returns:
            torch.Tensor: Weighted IoU loss.

        Note:
            This method calculates the IoU loss and applies the provided weights.
            The predicted values are passed through a sigmoid function if self.from_logits is True.
            The weighted loss is computed as
            1 - (weighted intersection + 1) / (weighted union - weighted intersection + 1).
        """
        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)
        y_pred = torch.sigmoid(y_pred)
        inter = ((y_pred * y_true) * weight).sum(dim=dims)
        union = ((y_pred + y_true) * weight).sum(dim=dims)
        weighted_iou = 1 - (inter + 1) / (union - inter + 1)
        return weighted_iou


class SoftBCEWithLogitsLoss(nn.Module):

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = -100,
        reduction: str = "mean",
        smooth_factor: Optional[float] = None,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """Drop-in replacement for torch.nn.BCEWithLogitsLoss with few additions: ignore_index and label_smoothing

        Args:
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 1] -> [0.9, 0.1, 0.9])

        Shape
             - **y_pred** - torch.Tensor of shape NxCxHxW
             - **y_true** - torch.Tensor of shape NxHxW or Nx1xHxW

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: torch.Tensor of shape (N, C, H, W)
            y_true: torch.Tensor of shape (N, H, W)  or (N, 1, H, W)

        Returns:
            loss: torch.Tensor
        """

        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (1 - self.smooth_factor)
        else:
            soft_targets = y_true

        loss = F.binary_cross_entropy_with_logits(
            y_pred,
            soft_targets,
            self.weight,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss


class SoftCrossEntropyLoss(CrossEntropy):

    def __init__(
        self,
        reduction: str = "mean",
        smooth_factor: Optional[float] = None,
        dim: int = 1,
        pos_weight: Optional[torch.Tensor] = None,
        from_logits: bool = True,
        mode: str = 'binary',
        ignore_value: float = -1,
        classes: List[int] = None
    ):
        """
        SoftCrossEntropyLoss

        Args:
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of
                elements in the output, 'sum': the output will be summed. Default: 'mean'
            smooth_factor (float): Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 1] -> [0.9, 0.1, 0.9])
            dim (int): Dimension to apply the loss to. Default: 1
            pos_weight (torch.Tensor): A weight of positive examples. Must be a vector with length equal to the
                number of classes. Default: None
            from_logits (bool): If True, assumes input is raw logits. If False, assumes input is probabilities.
                Default: True
            mode (str): Specifies the task type: 'binary' | 'multilabel' | 'multiclass'. Default: 'binary'
            ignore_value (float): Specifies a target value that is ignored and does not contribute to the input
                gradient.
                Default: -1
            classes (List[int]): List of classes that contribute in loss computation. By default, all channels are
                included. Default: None
        """
        super().__init__()
        self.smooth_factor = smooth_factor
        self.dim = dim
        super().__init__(
            reduction=reduction,
            pos_weight=pos_weight,
            mode=mode,
            ignore_value=ignore_value,
            from_logits=from_logits,
            classes=classes
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate the soft cross entropy loss value.

        Args:
            y_pred (torch.Tensor): The model's predicted output. Shape should be (N, C, ...) where N is
                the batch size and C is the number of classes.
            y_true (torch.Tensor): The ground truth labels. Shape should match y_pred or be (N, ...) for
                class indices.
            filtration_mask (torch.Tensor, optional): Mask for filtering regions of interest. Should have
                the same spatial dimensions as y_pred and y_true.

        Returns:
            torch.Tensor: The computed soft cross entropy loss.

        Note:
            - The method handles binary, multi-binary, and multiclass segmentation.
            - Uncertain annotations and regions outside the filtration mask are handled as per the base class methods.
        """
        log_prob = F.log_softmax(y_pred, dim=self.dim)
        loss = label_smoothed_nll_loss(
            log_prob,
            y_true,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            dim=self.dim,
        )
        loss = self.filter_uncertain_annotation(data_tensor=loss, gt_mask=y_true)
        loss = self.add_weights(loss=loss, gt_mask=y_true)
        loss = self.roi_filtration(data_tensor=loss, filtration_mask=filtration_mask)
        if self.classes is not None:
            loss = loss[self.classes]
        return self.aggregate_loss(loss=loss)
