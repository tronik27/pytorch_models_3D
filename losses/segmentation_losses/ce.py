from typing import Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F

from base_modules import CrossEntropy, label_smoothed_nll_loss


class SegmentationCE(CrossEntropy):
    """Cross entropy segmentation loss class."""
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Method for loss value calculation.
        :param y_pred: model predicted output.
        :param y_true: ground truth.
        :param filtration_mask: area of interest masks
        """
        loss = self.ce(y_pred, y_true)
        loss = self.filter_uncertain_annotation(data_tensor=loss, gt_mask=y_true)
        loss = self.add_weights(loss=loss, gt_mask=y_true)
        loss = self.roi_filtration(data_tensor=loss, filtration_mask=filtration_mask)
        return self.aggregate_loss(loss=loss, y_true=y_true)


class Focal(CrossEntropy):

    def __init__(
            self,
            pos_weight=None,
            reduction='mean',
            from_logits=True,
            mode: str = 'binary',
            ignore_value=-1,
            gamma=2,
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
            reduction=reduction, pos_weight=pos_weight, mode=mode, ignore_value=ignore_value, from_logits=from_logits
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
        return self.aggregate_loss(loss=loss, y_true=y_true)


class WeightedCE(CrossEntropy):

    def __init__(
            self,
            reduction='mean',
            from_logits=True,
            mode: str = 'binary',
            ignore_value=-1,
            batchwise=False,
    ) -> None:
        """
        Weighted Cross-Entropy and IoU Loss.

        This loss function combines Cross-Entropy (CE) and Intersection over Union (IoU) losses,
        weighted by the ground truth mask. It's designed to handle class imbalance and improve
        segmentation accuracy, especially for small or difficult objects.

        Args:
            reduction (str): Specifies the reduction to apply to the output.
                Options: 'none' | 'mean' | 'sum'. Default: 'mean'.
            from_logits (bool): If True, assumes input is raw logits. Default: True.
            mode (str): Specifies the mode of operation.
                Options: 'binary' | 'multi-binary' | 'multiclass'. Default: 'binary'.
            ignore_value (int): Specifies a target value that is ignored and does not contribute to the input gradient.
                Default: -1.
            batchwise (bool): If True, computes the loss across the batch. Default: False.

        Note:
            - The weighting is applied to CE using IoU.
            - This loss is particularly useful for segmentation tasks with imbalanced classes or small objects.
        """
        super().__init__(
            reduction=reduction, mode=mode, ignore_value=ignore_value, from_logits=from_logits
        )
        self.batchwise = batchwise
        self.from_logits = from_logits

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, filtration_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Method for focal loss value calculation.
        Args:
            y_pred: model predicted output.
            y_true: ground truth.
            filtration_mask: area of interest masks

        Returns:

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
        
        if self.mode == "multiclass":
            y_true = F.one_hot(y_true.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
            y_true = y_true.view(batch_size, num_classes, -1)
            y_pred = y_pred.view(batch_size, num_classes, -1)

        y_true = self.filter_uncertain_annotation(data_tensor=y_true, gt_mask=y_true)
        y_true = self.roi_filtration(filtration_mask=filtration_mask, data_tensor=y_true)

        y_pred = self.filter_uncertain_annotation(data_tensor=y_pred, gt_mask=y_true)
        y_pred = self.roi_filtration(filtration_mask=filtration_mask, data_tensor=y_pred)

        weight = 1 + 5 * torch.abs(nn.functional.avg_pool3d(y_true, kernel_size=31, stride=1, padding=15) - y_true)
        weighted_bce = self.__ce(y_true=y_true, y_pred=y_pred, weight=weight, dims=dims)
        weighted_iou = self.__iou(y_true=y_true, y_pred=y_pred, weight=weight, dims=dims)
        loss = weighted_iou + weighted_bce

        return self.aggregate_loss(loss=loss, y_true=gt)

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
            The weighted loss is computed as 1 - (weighted intersection + 1) / (weighted union - weighted intersection + 1).
        """
        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)
        y_pred = torch.sigmoid(y_pred)
        inter = ((y_pred * y_true) * weight).sum(dim=dims)
        union = ((y_pred + y_true) * weight).sum(dim=dims)
        weighted_iou = 1 - (inter + 1) / (union - inter + 1)
        return weighted_iou


class SoftBCEWithLogitsLoss(nn.Module):

    __constants__ = [
        "weight",
        "pos_weight",
        "reduction",
        "ignore_index",
        "smooth_factor",
    ]

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


class SoftCrossEntropyLoss(nn.Module):

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(
        self,
        reduction: str = "mean",
        smooth_factor: Optional[float] = None,
        ignore_index: Optional[int] = -100,
        dim: int = 1,
    ):
        """Drop-in replacement for torch.nn.CrossEntropyLoss with label_smoothing

        Args:
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 0] -> [0.9, 0.05, 0.05])

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(y_pred, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            y_true,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )
