from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from base_modules import soft_jaccard_score, to_tensor
from dice import DiceLoss

__all__ = ["JaccardLoss"]


class JaccardLoss(DiceLoss):
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-7,
        ignore_value: float = -1,
        ignore_mask: bool = False,
        batchwise: bool = False,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        """Jaccard loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes: List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(jaccard_coeff)`, otherwise `1 - jaccard_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)
            ignore_value: Label that indicates ignored pixels (does not contribute to loss)
            ignore_mask: If True, ignore certain regions in the mask
            batchwise: If True, compute loss batchwise
            pos_weight: Weights for positive examples

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, D, H, W)
             - **y_true** - torch.Tensor of shape (N, D, H, W) or (N, C, D, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super(JaccardLoss, self).__init__(
            mode=mode,
            classes=classes,
            log_loss=log_loss,
            from_logits=from_logits,
            smooth=smooth,
            ignore_value=ignore_value,
            ignore_mask=ignore_mask,
            eps=eps,
            batchwise=batchwise,
            pos_weight=pos_weight
        )

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, dims: int or Tuple[int, int]) -> torch.Tensor:
        """
        Compute the Jaccard loss between predicted and true tensors.

        This method calculates the Jaccard loss, which is based on the Jaccard similarity coefficient 
        (also known as Intersection over Union). It supports binary, multiclass, and multilabel cases.

        Args:
            y_pred (torch.Tensor): The predicted tensor.
            y_true (torch.Tensor): The ground truth tensor.
            dims (int or Tuple[int, int]): The dimensions along which to sum.

        Returns:
            torch.Tensor: The computed Jaccard loss.

        Note:
            - The loss is computed as 1 - Jaccard coefficient.
            - If log_loss is True, the negative log of the Jaccard coefficient is returned.
            - The smooth term is added to both the numerator and denominator for numerical stability.
        """
        scores = soft_jaccard_score(
            y_pred,
            y_true.type(y_pred.dtype),
            smooth=self.smooth,
            eps=self.eps,
            dims=dims,
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        return loss
