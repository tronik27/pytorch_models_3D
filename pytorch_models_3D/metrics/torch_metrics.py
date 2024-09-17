from typing import List, Optional, Union
from torchmetrics import Metric, AveragePrecision, AUROC
import torch

from torch import Tensor

from torchmetrics.functional.classification.average_precision import _multilabel_precision_recall_curve_update
from torchmetrics.functional.classification.auroc import _multilabel_precision_recall_curve_update


class IoU(Metric):
    """IoU score"""

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        thresholds: Union[float, List[float]] = 0.5,
        thresholds_step: Optional[Union[int, List[int]]] = None,
        average: str = "mean",
        ignore_empty: bool = False,
        zero_division: float = 0.0,
    ):
        """
        Args:
            thresholds (Union[float, List[float]], optional): classification thresholds binary or multi-binary.\
            Defaults to 0.5.
            thresholds_step (Union[int, List[int]], optional): classification thresholds
            step binary or multi-binary, used when not None.\
            Defaults to None.
            average (str): average strategy, available modes ["mean", "none"]. Defaults to "mean".
            ignore_empty (bool): whether to ignore empty target masks [True, False]. Defaults to False.
            zero_division (float): use value when intersection and union are equal to zero. Defaults to 0.
        """
        super().__init__()

        self.thresholds = thresholds
        if thresholds_step is not None:
            self.thresholds_step = torch.linspace(0, 1.0, thresholds_step + 2)[1:-1]
            self.thresholds_step = torch.nn.Parameter(self.thresholds_step)
            self.num_steps = thresholds_step
        else:
            self.thresholds_step = thresholds_step
        self.average = average
        self.ignore_empty = ignore_empty
        self.zero_division = zero_division

        self.add_state("sample_iou", default=list(), dist_reduce_fx="cat")
        self.add_state("negative", default=list(), dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Function collects per sample iou and bool value is sample negative or not"""

        B, C = preds.shape[:2]
        preds = preds.view(B, C, -1)
        target = target.view(B, C, -1)
        negative = torch.clip(target.view(B, -1).sum(axis=-1), 0, 1).type(torch.bool)

        if self.thresholds_step is None:
            preds = preds.transpose(1, 2) > self.thresholds
            preds = preds.transpose(1, 2)
        else:
            preds = preds.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, self.num_steps)
            preds = preds > self.thresholds_step.unsqueeze(-1).expand(-1, C).T
            preds = preds.permute(0, 3, 2, 1)
            target = target.unsqueeze(1).expand(-1, self.num_steps, -1, -1)

        intersection = (preds * target).sum(dim=-1)
        cardinality = (preds + target).sum(dim=-1)
        union = cardinality - intersection
        scores = intersection / (union + 1e-6)
        scores[union.sum(dim=-1) == 0] = self.zero_division

        self.sample_iou.append(scores)
        self.negative.append(negative)

    def compute(self):
        """Computing IoU with respect to initialization parameters"""

        negatives = self.negative
        sample_iou = self.sample_iou

        if self.average == "none":
            return sample_iou

        if self.thresholds_step is None:
            if self.ignore_empty:
                return sample_iou[negatives].mean()
            else:
                return sample_iou.mean()

        if self.ignore_empty:
            sample_iou = sample_iou[negatives].transpose(0, 2).mean(dim=-1)
            best_thds = sample_iou.argmax(dim=-1)
            sample_iou = torch.as_tensor([sample_iou[i, thd] for i, thd in enumerate(best_thds)]).mean()
            return sample_iou
        else:
            sample_iou = sample_iou.transpose(0, 2).mean(dim=-1)
            best_thds = sample_iou.argmax(dim=-1)
            sample_iou = torch.as_tensor([sample_iou[i, thd] for i, thd in enumerate(best_thds)]).mean()
            return sample_iou


class Dice(Metric):
    """Dice score"""

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        thresholds: Union[float, List[float]] = 0.5,
        thresholds_step: Optional[Union[int, List[int]]] = None,
        average: str = "mean",
        ignore_empty: bool = False,
        zero_division: float = 0.0,
    ):
        """
        Args:
            thresholds (Union[float, List[float]], optional): classification thresholds binary or multi-binary.\
            Defaults to 0.5.
            thresholds_step (Union[int, List[int]], optional): classification thresholds
            step binary or multi-binary, used when not None.\
            Defaults to None.
            average (str): average strategy, avalible modes ["mean", "none"]. Defaults to "mean".
            ignore_empty (bool): whether to ignore empty target masks [True, False]. Defaults to False.
            zero_division (float): use value when intersetion andd union are equal to zero. Defaults to 0.
        """
        super().__init__()

        self.thresholds = thresholds
        if thresholds_step is not None:
            self.thresholds_step = torch.linspace(0, 1.0, thresholds_step + 2)[1:-1]
            self.thresholds_step = torch.nn.Parameter(self.thresholds_step)
            self.num_steps = thresholds_step
        else:
            self.thresholds_step = thresholds_step
        self.average = average
        self.ignore_empty = ignore_empty
        self.zero_division = zero_division

        self.add_state("sample_iou", default=[], dist_reduce_fx="cat")
        self.add_state("negative", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Function collects per sample Dice and bool value is sample negative or not"""

        B, C = preds.shape[:2]
        preds = preds.view(B, C, -1)
        target = target.view(B, C, -1)
        negative = torch.clip(target.view(B, -1).sum(axis=-1), 0, 1).type(torch.bool)

        if self.thresholds_step is None:
            preds = preds.transpose(1, 2) > self.thresholds
            preds = preds.transpose(1, 2)
        else:
            preds = preds.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, self.num_steps)
            preds = preds > self.thresholds_step.unsqueeze(-1).expand(-1, C).T
            preds = preds.permute(0, 3, 2, 1)
            target = target.unsqueeze(1).expand(-1, self.num_steps, -1, -1)

        intersection = (preds * target).sum(dim=-1)
        cardinality = (preds + target).sum(dim=-1)
        union = cardinality

        scores = 2 * intersection / (union + intersection + 1e-6)
        scores[union.sum(dim=-1) == 0] = self.zero_division

        self.sample_iou.append(scores)
        self.negative.append(negative)

    def compute(self):
        """Computing Dice with respect to initiialization parameters"""

        negatives = self.negative
        sample_iou = self.sample_iou

        if self.average == "none":
            return sample_iou

        if self.thresholds_step is None:
            if self.ignore_empty:
                return sample_iou[negatives].mean()
            else:
                return sample_iou.mean()

        if self.ignore_empty:
            sample_iou = sample_iou[negatives].transpose(0, 2).mean(dim=-1)
            best_thds = sample_iou.argmax(dim=-1)
            sample_iou = torch.as_tensor([sample_iou[i, thd] for i, thd in enumerate(best_thds)]).mean()
            return sample_iou
        else:
            sample_iou = sample_iou.transpose(0, 2).mean(dim=-1)
            best_thds = sample_iou.argmax(dim=-1)
            sample_iou = torch.as_tensor([sample_iou[i, thd] for i, thd in enumerate(best_thds)]).mean()
            return sample_iou


class AP(AveragePrecision):

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target, num_classes, pos_label = _multilabel_precision_recall_curve_update(
            preds.cpu(), target.cpu(), self.num_classes, self.pos_label, self.average
        )
        self.preds.append(preds)
        self.target.append(target)
        self.num_classes = num_classes
        self.pos_label = pos_label


class ROCAUC(AUROC):

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth labels
        """
        preds, target, mode = _multilabel_precision_recall_curve_update(preds.cpu(), target.cpu())

        self.preds.append(preds)
        self.target.append(target)

        if self.mode and self.mode != mode:
            raise ValueError(
                "The mode of data (binary, multi-label, multi-class) should be constant, but changed"
                f" between batches from {self.mode} to {mode}"
            )
        self.mode = mode
