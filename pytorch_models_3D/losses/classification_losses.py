from torch import nn
import torch
from abc import ABC, abstractmethod


class ClassificationLoss(nn.Module, ABC):

    def __init__(self, ignore_value: float = -1, reduction: str = 'None', pos_weight: torch.Tensor = None):
        super().__init__()
        self.ignore_value = ignore_value
        self.reduction = reduction
        self.pos_weight = pos_weight

    @abstractmethod
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        pass

    def ignore(self, y_true, y_pred):
        """
        Method for ignoring uncertain annotated masks.
        :param y_pred: model predicted output.
        :param y_true: ground truth.
        :return filtered loss tensor value.
        """
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        indices = y_true != self.ignore_value
        return y_pred[indices], y_true[indices]

    def aggregate_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Method for loss value aggregation by loss tensor reduction.
        :param loss: loss value tensor.
        return reduced loss value tensor.
        """
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class CrossEntropy(ClassificationLoss):
    def __init__(
            self, pos_weight=None, reduction='mean', from_logits=True, mode: str = 'binary', ignore_value: float = -1
    ) -> None:
        """
        Cross entropy segmentation loss class.
        """
        super().__init__(pos_weight=pos_weight, ignore_value=ignore_value, reduction=reduction)
        assert mode in ["binary", "multi-binary", "multiclass"], 'Incorrect task type!'
        self.mode = mode
        if self.mode == "multiclass":
            if from_logits:
                self.ce_loss = nn.CrossEntropyLoss(reduction='none')
            else:
                raise NotImplementedError
        else:
            if from_logits:
                self.ce_loss = nn.BCEWithLogitsLoss(reduction='none')
            else:
                self.ce_loss = nn.BCELoss(reduction='none')

    @abstractmethod
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        pass


class CELoss(CrossEntropy):

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Method for loss value calculation.
        :param y_pred: model predicted output.
        :param y_true: ground truth.
        """
        y_pred, y_true = self.ignore(y_pred=y_pred, y_true=y_true)
        loss = self.ce_loss(y_pred, y_true)
        loss = self.aggregate_loss(loss=loss)
        return loss


class FocalLoss(CrossEntropy):

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

    def forward(self, y_pred, y_true):
        """
        Method for loss value calculation.
        :param y_pred: model predicted output.
        :param y_true: ground truth.
        """
        y_true, y_pred = self.ignore(y_true, y_pred)
        cent_loss = self.ce_loss(y_pred, y_true)
        pt = torch.exp(-cent_loss)
        focal_loss = (1 - pt) ** self.gamma * cent_loss
        focal_loss = self.aggregate_loss(loss=focal_loss)
        return focal_loss


# class LabelSmoothingCrossEntropy(ClassificationLoss):
#     def __init__(self, weight=None, epsilon=0.1, reduce=None):
#         super().__init__()
#         self.epsilon = epsilon
#         self.reduction = reduce
#         self.nll_loss = nn.NLLLoss(weight=weight, reduce=reduce)
#
#     def forward(self, preds, target):
#         n = preds.size()[-1]
#         log_preds = func.log_softmax(preds, max_reduction=-1)
#         loss = self.aggregate_loss(-log_preds.sum(max_reduction=-1), self.reduction)
#         nll = self.nll_loss(log_preds, target)
#         return self.linear_combination(loss / n, nll)
#
#     def linear_combination(self, x, y_true):
#         return self.epsilon * x + (1 - self.epsilon) * y_true
