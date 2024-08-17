import torch
from torch.nn.modules.loss import _Loss
import numpy as np
import torch.nn.functional as F
from typing import Any
from research_sdk.losses.functional import soft_dice_score


def to_tensor(x: Any, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x


class ObjectDiceLoss(_Loss):
    def __init__(
        self,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        ignore_mask=False,
        batchwise=False,
        alpha=0.5,
        beta=0.5,
        gamma=1,
        eps=1e-7,
    ):
        super(ObjectDiceLoss, self).__init__()
        self.classes = classes
        self.from_logit = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index
        self.ignore_mask = ignore_mask
        self.bacthwise = batchwise
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y_pred: torch.tensor, y_true: torch.tensor, filtration_mask: torch.Tensor = None):
        assert y_true.size(0) == y_pred.size(0)
        if self.from_logit:
            y_pred = torch.sigmoid(y_pred)

        bs = y_true.size(0)
        num_classes = y_pred.size(1)

        # print(f"(B, C, H, W) y_pred: {y_pred.shape}, y_true: {y_true.shape}")

        y_true = y_true.view(bs, num_classes, -1)
        y_pred = y_pred.view(bs, num_classes, -1)

        y_true = self.roi_filtration(filtration_mask=filtration_mask, data_tensor=y_true)
        y_pred = self.roi_filtration(filtration_mask=filtration_mask, data_tensor=y_pred)

        # print(f"(B, C, HxW) y_pred: {y_pred.shape}, y_true: {y_true.shape}")

        ignore = y_true == -1

        # get y_true background target
        y_bg = y_true.clone()
        y_bg[y_bg > 0] = 1.0
        y_bg[ignore] = 1.0
        y_bg = 1 - y_bg

        # get y_true objects target
        y_fg = y_true.clone()
        y_fg[ignore] = 0.0
        y_objects = F.one_hot(y_fg.long()).permute(3, 0, 1, 2)[1:]

        # print(f"y_bg: {y_bg.shape}, y_objects: {y_objects.shape}
        dims = 2
        if self.bacthwise:
            dims = (0, 2)

        if y_objects.size(0) == 0:
            loss = soft_dice_score(y_pred, y_pred, self.smooth, self.eps, dims)
            loss = loss.mean()
        else:
            loss = self.compute_score(
                y_pred, y_bg.type_as(y_pred), y_objects.type_as(y_pred), self.alpha,
                self.beta, self.gamma, smooth=self.smooth, eps=self.eps
            )
        return loss

    @staticmethod
    def compute_score(output, target, target_obj, alpha, beta, gamma, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_dice_score_object(output, target, target_obj, alpha, beta, gamma, smooth, eps, dims)

    @staticmethod
    def roi_filtration(filtration_mask: torch.Tensor, data_tensor: torch.Tensor) -> torch.Tensor:
        if filtration_mask is not None:
            data_tensor = data_tensor * filtration_mask
        return data_tensor


def soft_dice_score_object(
        output: torch.Tensor,
        target_bg: torch.Tensor,
        target_obj: torch.Tensor,
        alpha,
        beta,
        gamma,
        smooth: float = 0.0,
        eps: float = 1e-7,
        dims=None
) -> torch.Tensor:
    assert output.size() == target_bg.size()
    tp = output * target_obj
    fn = (1 - output) * target_obj
    fp = output * target_bg

    # print(f"tp:{tp.shape}, fn: {fn.shape}, fp:{fp.shape}")

    tp = torch.sum(tp, dim=3)
    fn = torch.sum(fn, dim=3)
    fp = torch.sum(fp, dim=2)
    #
    fp_mask = fp < 0
    if fp_mask.sum() > 0:
        print(f"fp: {fp[fp_mask]}")

    fn_mask = fn < 0
    if fn_mask.sum() > 0:
        print(f"fp: {fp[fp_mask]}")

    # print(f"Summed tp:{tp.shape}, fn: {fn.shape}, fp:{fp.shape}")

    dice_score = (tp + smooth) / (tp + fp * alpha + fn * beta + smooth).clamp_min(eps)

    # print(f"dice_score: {dice_score.shape}")

    dice_loss = (1 - dice_score).permute(2, 0, 1)
    # dice_loss = torch.pow(1 + dice_loss, gamma)
    tp = tp.permute(2, 0, 1)

    # print(f"dice_loss: {loss.shape}, tp: {tp.shape}")

    loss_per_class = []
    for class_loss, class_tp in zip(dice_loss, tp):
        mask = class_tp > 0
        if mask.sum() > 0:
            loss_per_class.append(class_loss[mask].mean())
        else:
            loss_per_class.append(class_loss.mean())
        # print(f"objects in class: {(class_tp > 0).sum()}")

    loss_per_class = torch.stack(loss_per_class)
    loss = loss_per_class.mean()
    return loss
