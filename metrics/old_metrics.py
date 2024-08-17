import torch
from research_sdk.metrics import BaseLogger, metrics as mt, tensor2numpy
from research_sdk.losses.functional import soft_jaccard_score


class TrainLogger(BaseLogger):
    def __init__(self, config, float_precision=3):
        metrics = [
            mt.RocAUC(name="roc", mode="multi-binary"),
            mt.AP(name="ap", mode="multi-binary"),
        ]
        super(TrainLogger, self).__init__(metrics, float_precision)
        self.y = None
        self.y_hat = None
        self.config = config

    def update(self, target, output):
        target = tensor2numpy(target)
        output = tensor2numpy(output)

        self.y = target
        self.y_hat = output
        self.metrics['roc'].compute(self.y, self.y_hat, reduction="none")
        self.metrics['ap'].compute(self.y, self.y_hat, reduction="none")

    def reset(self):
        super(TrainLogger, self).reset()
        self.y = None
        self.y_hat = None


class TestLogger(BaseLogger):
    def __init__(self, config, float_precision=3):
        metrics = [
            mt.BalancedAccuracy(name="acc", mode="binary", progress_bar=True),
            mt.RocAUC(name="roc", mode="binary"),
            mt.AP(name="ap", mode="binary"),
            mt.RunningLoss(name="loss", progress_bar=True, reduction="momentum")
        ]
        super(TestLogger, self).__init__(metrics, float_precision)
        self.y = None
        self.y_hat = None
        self.h_hat = None
        self.config = config

    def update(self, target, output, loss):
        target = tensor2numpy(target)
        output = tensor2numpy(output)

        self.y = target
        self.y_hat = output

        self.metrics['acc'].compute(self.y, (self.y_hat > 0.5), reduction="none")
        self.metrics['roc'].compute(self.y, self.y_hat, reduction="none")
        self.metrics['ap'].compute(self.y, self.y_hat, reduction="none")
        self.metrics['loss'].compute(loss)

    def reset(self):
        super(TestLogger, self).reset()
        self.y = None
        self.y_hat = None
        self.h_hat = None


def jaccard_score(mask, mask_hat):
    mask_hat = (torch.sigmoid(mask_hat) > 0.5).float()
    scores = soft_jaccard_score(mask_hat, mask, smooth=1.0, dims=(2, 3))
    ignore_mask = torch.amax(mask, dim=(2, 3)) > 0
    return scores, ignore_mask


def aggregate_jacc(jacc, ignore):
    ajacc = []
    jacc = jacc.transpose(0, 1)
    ignore = ignore.transpose(0, 1)
    for cls_jacc, cls_ignore in zip(jacc, ignore):
        ajacc.append(cls_jacc[cls_ignore].mean())

    ajacc = torch.stack(ajacc)
    return ajacc.detach().cpu().numpy()


def print_jaccard(jacc, classnames):
    print("jacc score per class:")
    for i, cls in enumerate(classnames):
        print(cls + f": {jacc[i]:.4}")
