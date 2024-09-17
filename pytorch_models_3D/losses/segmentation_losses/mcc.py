from typing import List, Tuple
import torch
import torch.nn.functional as F
from base_modules import CustomLoss


class MCCLoss(CustomLoss):
    def __init__(
            self,
            eps: float = 1e-5,
            ignore_value: float = -1,
            pos_weight: torch.Tensor = None,
            reduction: str = 'mean',
            batchwise: bool = False,
            from_logits: bool = True,
            mode: str = 'binary',
            classes: List[int] = None
    ):

        """
        Implementation of Matthews Correlation Coefficient (MCC) loss.

        Parameters
        ----------
        eps : float, optional
            A small epsilon for numerical stability to avoid division by zero.
            Defaults to 1e-5.
        ignore_value : float, optional
            A target value that is ignored and does not contribute to the input gradient.
            Defaults to -1.
        pos_weight : torch.Tensor, optional
            A weight tensor for positive examples. Must be a vector with length equal to the
            number of classes. If None, all weights are set to 1.
            Defaults to None.
        reduction : str, optional
            Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            Defaults to 'mean'.
        batchwise : bool, optional
            If True, computes loss batchwise.
            Defaults to False.
        from_logits : bool, optional
            If True, assumes input is raw logits.
            Defaults to True.
        mode : str, optional
            Specifies the task type: 'binary' | 'multi-binary' | 'multiclass'.
            Defaults to 'binary'.
        classes : List[int], optional
            List of classes that contribute in loss computation. By default, all channels are included.
            Defaults to None.
        """
        super().__init__(
            ignore_value=ignore_value,
            pos_weight=pos_weight,
            reduction=reduction,
            from_logits=from_logits,
            mode=mode,
            classes=classes,
            batchwise=batchwise
        )
        self.eps = eps

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
        """
        Compute the Matthews Correlation Coefficient Loss between predicted and true segmentation masks.

        Args:
            dims: (int or Tuple[int, int]): The dimensions along which to sum.
            y_pred (torch.Tensor): Predicted segmentation mask.
            y_true (torch.Tensor): Ground truth segmentation mask.

        Returns:
            torch.Tensor: Computed Matthews Correlation Coefficient Loss.

        Note:
            - The loss is computed as 1 - MCC
            - The method handles smoothing to avoid division by zero.
            - It can compute loss for binary, multiclass and multilabel cases.
        """
        if self.mode == 'binary':
            tp = torch.sum(torch.mul(y_pred, y_true)) + self.eps
            tn = torch.sum(torch.mul((1 - y_pred), (1 - y_true))) + self.eps
            fp = torch.sum(torch.mul(y_pred, (1 - y_true))) + self.eps
            fn = torch.sum(torch.mul((1 - y_pred), y_true)) + self.eps

            numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
            denominator = torch.sqrt(torch.add(tp, fp) * torch.add(tp, fn) * torch.add(tn, fp) * torch.add(tn, fn))

            mcc = torch.div(numerator.sum(dim=dims), denominator.sum(dim=dims))
            loss = 1.0 - mcc

        else:
            losses = list()
            for i in range(y_pred.size(1)):
                y_pred_i = y_pred[:, i:i + 1, ...]
                y_true_i = y_true[:, i:i + 1, ...]

                tp = torch.sum(torch.mul(y_pred_i, y_true_i)) + self.eps
                tn = torch.sum(torch.mul((1 - y_pred_i), (1 - y_true_i))) + self.eps
                fp = torch.sum(torch.mul(y_pred_i, (1 - y_true_i))) + self.eps
                fn = torch.sum(torch.mul((1 - y_pred_i), y_true_i)) + self.eps

                numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
                denominator = torch.sqrt(torch.add(tp, fp) * torch.add(tp, fn) * torch.add(tn, fp) * torch.add(tn, fn))

                mcc = torch.div(numerator.sum(dim=dims), denominator.sum(dim=dims))
                losses.append(1.0 - mcc)

            loss = torch.mean(torch.stack(losses))

        return loss
