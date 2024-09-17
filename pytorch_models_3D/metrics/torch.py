from torchmetrics import Metric
import torch
from typing import Union, List


class IoU(Metric):
    """IoU score"""

    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, thresholds: Union[float, List[float]] = 0.5):
        """
        Args:
            thresholds (Union[float, List[float]], optional): classification threshold binary or multi-binary.\
            Defaults to 0.5.
        """
        super().__init__()
        self.thresholds = thresholds

        self.add_state("number", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("sum", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Function updates sum of IoUs per sample and number of samples processed"""
        B, C = preds.shape[:2]
        preds = preds.view(B, C, -1)
        target = target.view(B, C, -1)

        preds = preds.transpose(1, 2) > self.thresholds
        preds = preds.transpose(1, 2)

        intersection = (preds * target).sum(dim=-1)
        cardinality = (preds + target).sum(dim=-1)
        union = cardinality - intersection

        scores = 2 * intersection / (union + 1e-6)
        scores = scores.mean(dim=-1)

        self.sum += scores.sum()
        self.number += B

    def compute(self):
        """Computing score as mean IoU through all processed samples"""
        return self.sum.float() / self.number
