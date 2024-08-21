from typing import List, Tuple
import torch

from dice import DiceLoss


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

