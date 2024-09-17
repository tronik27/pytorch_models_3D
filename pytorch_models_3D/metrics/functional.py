import numpy as np
from functools import wraps
import sklearn.metrics as sklm
from typing import Tuple


__all__ = [
    "_ignore",
    "ignore",
    "balanced_accuracy_score",
    "roc_auc_score",
    "average_precision_score",
]


def _ignore(y: np.ndarray, y_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Filter samples with ground truth set to -1

    Args:
        y (np.ndarray): 1-d array of ground truth values
        y_hat (np.ndarray): 1-d array of predictions

    Returns:
        Tuple[np.ndarray, np.ndarray]: output `y_filt` and `y_hat_filt` params

        - y_filt (np.ndarray): 1-d array of filtered ground truth values
        - y_hat_filt (np.ndarray): 1-d array of filtered predictions
    """
    y = y.reshape(-1)
    y_hat = y_hat.reshape(-1)
    indices = y > -1
    y_filt, y_hat_filt = y[indices], y_hat[indices]
    return y_filt, y_hat_filt


def ignore(func):
    """
    Function wrapper to ignore samples where gt set to -1
    """

    @wraps(func)
    def wrapped_function(y, y_hat, *args, **kwargs):
        y, y_hat = _ignore(y, y_hat)
        return func(y, y_hat, *args, **kwargs)

    return wrapped_function


@ignore
def balanced_accuracy_score(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Balanced accuracy score counter

    Args:
        y (np.ndarray): 1-d array of ground truth values
        y_hat (np.ndarray): 1-d array of predictions

    Returns:
        float: balanced accuracy score in [0...1]
    """
    confusion = sklm.confusion_matrix(y, y_hat)
    norm_confusion = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-8)
    return np.mean(np.diag(norm_confusion))


@ignore
def roc_auc_score(y: np.ndarray, y_hat: np.ndarray) -> float:
    """ROC AUC score counter

    Args:
        y (np.ndarray): 1-d array of ground truth values
        y_hat (np.ndarray): 1-d array of predictions

    Returns:
        float: roc auc score in [0...1]
    """
    if (y == 0).all() or (y == 1).all():
        return 0.0

    return sklm.roc_auc_score(y, y_hat)


@ignore
def average_precision_score(y: np.ndarray, y_hat: np.ndarray) -> float:
    """AP score counter

    Args:
        y (np.ndarray): 1-d array of ground truth values
        y_hat (np.ndarray): 1-d array of predictions

    Returns:
        float: average precision score in [0...1]
    """
    if (y == 0).all() or (y == 1).all():
        return 0.0

    precision, recall, _ = sklm.precision_recall_curve(y.astype(int), y_hat)
    for j in range(len(precision)):
        precision[j] = np.max(precision[: j + 1])
    average_precision = -np.sum(np.diff(recall) * np.array(precision)[:-1])
    return average_precision
