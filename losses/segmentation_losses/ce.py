from typing import Tuple
import torch
from torch import nn

from base_modules import CrossEntropy


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
            ignore_value=-1,
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

