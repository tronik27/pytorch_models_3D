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
        Combined loss function class for classification tasks.

        This class combines multiple classification loss functions, allowing for a weighted sum of different losses.

        Args:
            losses (List[str]): List of loss function names to be used.
            mode (str): The mode of operation ('binary', 'multiclass', or 'multilabel').
            loss_weights (List[float], optional): Weights for each loss function. If None, equal weights are used.
            weights (torch.Tensor, optional): Class weights for imbalanced datasets.
            batchwise (bool): If True, compute loss batchwise. Defaults to False.
            alpha (float): Parameter for Focal Tversky Loss. Defaults to 0.5.
            beta (float): Parameter for Focal Tversky Loss. Defaults to 0.5.
            gamma (float): Focal parameter for Focal Loss and Focal Tversky Loss. Defaults to 0.5.

        Attributes:
            loss_weights (List[float]): Weights for each loss function.
            bce (CELoss): Binary Cross Entropy loss instance.
            focal (FocalLoss): Focal Loss instance.
            losses (dict): Dictionary mapping loss names to their respective instances.
            use_losses (list): List of loss names to be used in forward pass.

        Raises:
            NotImplementedError: If an unknown loss type is specified.

        Note:
            The class currently supports 'ce_loss' (Cross Entropy) and 'focal' (Focal Loss).
            Additional loss functions can be added to the `losses` dictionary as needed.
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
        Combined loss function class for segmentation tasks.

        This class combines multiple loss functions for segmentation, allowing for a flexible
        and customizable approach to training segmentation models.

        Args:
            losses (List[str] or str): List of loss function names or a single loss function name.
            mode (str): Loss mode, either 'binary' or 'multibinary'.
            cls_losses (List[str] or str, optional): List of classification loss names or a single loss name.
            loss_weights (List[float], optional): Weights for each segmentation loss function. 
                Must match the length of 'losses' if provided.
            cls_loss_weights (List[float], optional): Weights for each classification loss function.
                Must match the length of 'cls_losses' if provided.
            mask_weight (float, default=1.): Weight for the segmentation mask loss.
            mask_pos_weight (float, default=1.): Positive class weight for the segmentation mask.
            weights (torch.Tensor, optional): Positive class weights for classification loss. 
                Must be a vector with length equal to the number of classes.
            batchwise (bool, default=False): If True, compute loss batchwise.
            alpha (float, default=0.5): Parameter for Tversky and Focal Tversky losses.
            beta (float, default=0.5): Parameter for Tversky and Focal Tversky losses.
            gamma (float, default=0.5): Focusing parameter for Focal and Focal Tversky losses.

        Note:
            - The class supports various segmentation losses including BCE, Dice, Tversky, and Focal losses.
            - Classification losses can be added separately using the 'cls_losses' parameter.
            - Ensure that the provided loss names are implemented in the class.
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
        Calculate the combined loss for classification and segmentation tasks.

        Args:
            classification_pred (torch.Tensor): Predicted class probabilities or logits.
            pred_mask (torch.Tensor or List[torch.Tensor]): Predicted segmentation mask(s).
            gt (torch.Tensor): Ground truth segmentation masks.
            classification_gt (torch.Tensor): Ground truth classification labels.
            filtration (torch.Tensor, optional): Mask for filtering regions of interest. Defaults to None.

        Returns:
            torch.Tensor: Combined loss value for classification and segmentation tasks.

        Note:
            - If pred_mask is a list, the segmentation loss is computed for each mask and summed.
            - The final loss is a weighted sum of classification and segmentation losses.
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
        Calculate the segmentation loss function value.

        This method computes the segmentation loss by combining multiple loss functions
        as specified in self.use_losses, with their corresponding weights.

        Args:
            pred_mask (torch.Tensor): Predicted segmentation mask.
            gt_mask (torch.Tensor): Ground truth segmentation mask.
            filtration_mask (torch.Tensor, optional): Mask for filtering regions of interest.

        Returns:
            torch.Tensor: Calculated segmentation loss value.

        Note:
            The final loss is a weighted sum of individual loss functions specified
            in self.use_losses. If only one loss function is used, its value is
            returned directly.
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
