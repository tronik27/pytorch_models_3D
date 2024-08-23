from torch import nn
import torch
from typing import List
from segmentation_losses.ce import Focal, SegmentationCE, WeightedCE
from segmentation_losses.tversky import TverskyLoss, FocalTverskyLoss
from losses.classification_losses import CELoss, FocalLoss
from segmentation_losses.dice import DiceLoss, AdaptiveTvMFDiceLoss


class CombinedClassificationLoss(nn.Module):
    def __init__(
            self,
            losses: List[str],
            mode: str,
            loss_weights: List[float] = None,
            weights: torch.Tensor = None,
            batchwise: bool = False,
            alpha: float = 0.5,
            beta: float = 0.5,
            gamma: float = 0.5,
    ) -> None:
        """
        Combined classification loss class.
        Args:
            losses:
            mode:
            loss_weights:
            weights:
            batchwise:
            alpha:
            beta:
            gamma:
        """
        super().__init__()
        if loss_weights is not None:
            assert len(loss_weights) == len(losses), \
                'Number of loss weights must be equal to number of loss functions!'
            self.loss_weights = loss_weights
        else:
            self.loss_weights = [1]*len(losses)
        self.bce = CELoss(reduction='mean', from_logits=True, pos_weight=weights, mode=mode)
        self.focal = FocalLoss(reduction='mean', from_logits=True, pos_weight=weights, gamma=gamma, mode=mode)
        self.losses = {
            'ce_loss': self.bce,
            'focal': self.focal,
        }
        self.use_losses = list()
        for loss_name in losses:
            if loss_name in self.segmentation_losses.keys():
                self.use_losses.append(loss_name)
            else:
                raise NotImplementedError('Unknown type of classification loss!')

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        loss = None
        for loss_name, loss_weight in zip(self.use_losses, self.loss_weights):
            curr_loss = self.losses[loss_name](y_true=gt, y_pred=pred) * loss_weight
            loss = curr_loss if loss is None else loss + curr_loss
        return loss


class CombinedSegmentationLoss(nn.Module):

    def __init__(
            self,
            losses: List[str] or str,
            mode: str,
            cls_losses: List[str] or str = None,
            loss_weights: List[float] = None,
            cls_loss_weights: List[float] = None,
            mask_weight: float = 1.,
            mask_pos_weight: float = 1.,
            weights: torch.Tensor = None,
            batchwise: bool = False,
            alpha: float = 0.5,
            beta: float = 0.5,
            gamma: float = 0.5,
    ) -> None:
        """
        Combined loss function class for segmentation.
        Args:
            losses: list of loss names
            mode: loss mode (binary, multibinary).
            loss_weights:
            mask_weight:
            mask_pos_weight:
            weights: positive class weights for classification loss. Must be a vector with length equal to the number
        of classes.
            batchwise:
            alpha:
            beta:
            gamma:
        """
        super(CombinedSegmentationLoss, self).__init__()
        self.heatmap_weight = torch.as_tensor(mask_weight)
        self.heatmap_pos_weights = torch.as_tensor(mask_pos_weight)
        if loss_weights is not None:
            assert len(loss_weights) == len(losses), \
                'Number of loss weights must be equal to number of loss functions!'
            self.loss_weights = loss_weights
        else:
            self.loss_weights = [1]*len(losses)
        self.segmentation_bce = SegmentationCE(
            reduction='mean', from_logits=True, pos_weight=self.heatmap_pos_weights, mode=mode
        )
        self.segmentation_dice_loss = DiceLoss(
            mode=mode,
            from_logits=True,
            batchwise=batchwise,
            smooth=1e-4,
            ignore_value=-1,
        )
        self.segmentation_tversky_loss = TverskyLoss(
            mode=mode,
            from_logits=True,
            batchwise=batchwise,
            smooth=1e-4,
            ignore_value=-1,
            alpha=alpha,
            beta=beta,
        )
        self.segmentation_focal_tversky_loss = FocalTverskyLoss(
            mode=mode,
            from_logits=True,
            batchwise=batchwise,
            smooth=1e-4,
            ignore_value=-1,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )
        self.segmentation_focal_loss = Focal(
            reduction='mean',
            from_logits=True,
            pos_weight=self.heatmap_pos_weights,
            mode=mode if not mode else mode,
        )
        self.weighted_loss = WeightedCE(
            reduction='mean',
            from_logits=True,
            mode=mode,
            batchwise=batchwise
        )
        self.tvmf_dice = AdaptiveTvMFDiceLoss(
            mode=mode,
            from_logits=True,
            batchwise=batchwise,
            smooth=1e-4,
            ignore_value=-1,
        )
        self.segmentation_losses = {
            "cross_entropy": self.segmentation_bce,
            'focal': self.segmentation_focal_loss,
            'dice': self.segmentation_dice_loss,
            'tversky': self.segmentation_tversky_loss,
            'focal_tversky': self.segmentation_focal_tversky_loss,
            'weighted': self.weighted_loss,
            'tvmf_dice': self.tvmf_dice,
        }
        self.use_losses = list()
        for loss_name in losses:
            if loss_name in self.segmentation_losses.keys():
                self.use_losses.append(loss_name)
            else:
                raise NotImplementedError('Unknown type of loss!')
        self.classification_loss = CombinedClassificationLoss(losses=cls_losses, weights=cls_loss_weights, mode=mode)

    def forward(
            self,
            classification_pred: torch.Tensor,
            pred_mask: torch.Tensor or List[torch.Tensor],
            gt: torch.Tensor,
            classification_gt: torch.Tensor,
            filtration: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Method for calculating the value of the loss function.
        :param classification_pred: predicted class labels.
        :param pred_mask: predicted class labels.
        :param gt: ground truth masks.
        :param classification_gt: ground truth labels.
        :param filtration: area of interest masks.
        :return calculated loss value
        """
        cls_loss = self.classification_loss(pred=classification_pred, y_true=classification_gt)
        pred_mask = pred_mask if isinstance(pred_mask, list) else [pred_mask]
        segmentation_loss = torch.sum(torch.stack([
            self.__segmentation_loss(pred_mask=mask, gt_mask=gt, filtration_mask=filtration) for mask in pred_mask
        ], dim=0))
        loss = cls_loss + segmentation_loss * self.heatmap_weight
        return loss

    def __segmentation_loss(
            self, pred_mask: torch.Tensor, gt_mask: torch.Tensor, filtration_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Method for calculating the value of the segmentation loss function.
        :param pred_mask: predicted segmentation masks.
        :param gt_mask: ground truth masks.
        :param filtration_mask: area of interest masks.
        :return calculated loss value
        """
        loss = None
        for loss_name, loss_weight in zip(self.use_losses, self.loss_weights):
            if loss is None:
                loss = self.segmentation_losses[loss_name](
                    y_true=gt_mask, y_pred=pred_mask, filtration_mask=filtration_mask
                ) * loss_weight
            else:
                loss += self.segmentation_losses[loss_name](
                    y_true=gt_mask, y_pred=pred_mask, filtration_mask=filtration_mask
                ) * loss_weight
        return loss
