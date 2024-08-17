import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from typing import List, Tuple
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns

import opinion.metrics.functional as F


__all__ = []


def fig2numpy(fig: Figure) -> np.ndarray:
    """Convert matplotlib figure to numpy image

    Args:
        fig (Figure): matplotlib figure with plot

    Returns:
        np.ndarray: plot converted to numpy array
    """
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    plot = data.reshape((int(h), int(w), -1))
    return plot


def plot_confusion_matrix(y: np.ndarray, y_hat: np.ndarray, classes: List[str] = None) -> Tuple[np.ndarray, float]:
    """Plot confusion matrix for binary and multi-class classification

    Args:
        y (np.ndarray): 1-d array of ground truth values
        y_hat (np.ndarray): 1-d array of predictions
        classes (List[str], optional): array of classnames. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: output `plot` and `score` params

        - plot (np.ndarray): confusion matrix's plot
        - score (np.ndarray): balanced accuracy score
    """
    confusion = confusion_matrix(y, y_hat)
    norm_confusion = 100 * confusion / (confusion.sum(axis=1, keepdims=True) + 1e-8)
    score = F.balanced_accuracy_score(y, y_hat)

    annotation = []
    for i in range(confusion.shape[0]):
        labels = []
        for value, norm in zip(confusion[i], norm_confusion[i]):
            labels.append("{:d} ({:.2f}%)".format(value, np.round(norm, 2)))
        annotation.append(labels)

    annotation = np.array(annotation)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax = sns.heatmap(
        norm_confusion,
        ax=ax,
        annot=annotation,
        fmt="",
        xticklabels=classes,
        cmap=plt.cm.Blues,
        linecolor="black",
        linewidths=0.25,
        cbar=False,
        annot_kws={"fontsize": "xx-large"},
    )

    if classes is None:
        classes = map(str, np.arange(confusion.shape[0]))

    ax.set_title(f"Balanced accuracy: {score:.4f}", fontsize=18)
    ax.set_yticklabels(classes, va="center", rotation=90, position=(0, 0.28))
    ax.set_xticklabels(classes, va="center", rotation=0, position=(0, -0.01))
    ax.tick_params(labelsize=18)
    ax.grid(linestyle="--", alpha=0.35)

    plot = fig2numpy(fig)
    return plot, score


def plot_roc(
    y: np.ndarray,
    y_hat: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Plot roc for binary classification

    Args:
        y (np.ndarray): 1-d array of ground truth values
        y_hat (np.ndarray): 1-d array of predictions

    Returns:
        Tuple[np.ndarray, np.ndarray]: output `plot` and `score` params

        - plot (np.ndarray): roc's plot
        - score (np.ndarray): roc auc score
    """
    score = F.roc_auc_score(y, y_hat)
    fpr, tpr, _ = roc_curve(y, y_hat)

    fig = plt.figure(figsize=(20, 20), dpi=101)
    ax = plt.subplot()

    ax.plot(
        100.0 * fpr,
        100.0 * tpr,
        label="ROC curve (area = {0:0.4f})".format(score),
        marker=".",
        lw=5,
        aa=True,
        alpha=0.9,
    )
    ax.plot([0, 100], [0, 100], "k--", lw=2)
    ax.set_xlim([-1, 101])
    ax.set_ylim([-1, 101])
    ax.set_xlabel("False Positive Rate, %", fontsize=24)
    ax.set_ylabel("True Positive Rate, %", fontsize=24)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, 101, 5), minor=False)
    ax.set_xticks(range(0, 101, 1), minor=True)
    ax.set_yticks(range(0, 101, 5), minor=False)
    ax.set_yticks(range(0, 101, 1), minor=True)
    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.tick_params(axis="both", which="minor", labelsize=24)
    ax.set_title("ROC", fontsize=24)
    ax.legend(loc="lower right", fontsize=24)
    ax.grid(which="major", alpha=1.0, linewidth=2, linestyle="--")
    ax.grid(which="minor", alpha=0.5, linestyle="--")

    plot = fig2numpy(fig)
    return plot, score


def plot_precision_recall(
    y: np.ndarray,
    y_hat: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Plot precision recall for binary classification

    Args:
        y (np.ndarray): 1-d array of ground truth values
        y_hat (np.ndarray): 1-d array of predictions

    Returns:
        Tuple[np.ndarray, np.ndarray]: output `plot` and `score` params

        - plot (np.ndarray): pr's plot
        - score (np.ndarray): average precision score
    """
    precision, recall, _ = precision_recall_curve(y.astype(int), y_hat)
    for j in range(len(precision)):
        precision[j] = np.max(precision[: j + 1])
    score = F.average_precision_score(y, y_hat)

    color = "slateblue"
    y_values = [100.0 * x for x in precision]
    x_values = [100.0 * x for x in recall]
    fig = plt.figure(figsize=(20, 20), dpi=101)
    ax = plt.subplot()
    ax.set_xlim([-1, 101])
    ax.set_ylim([-1, 101])
    ax.set_xlabel("Recall, %", fontsize=24)
    ax.set_ylabel("Precision, %", fontsize=24)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, 101, 5), minor=False)
    ax.set_xticks(range(0, 101, 1), minor=True)
    ax.set_yticks(range(0, 101, 5), minor=False)
    ax.set_yticks(range(0, 101, 1), minor=True)
    ax.tick_params(axis="both", which="major", labelsize=24)
    ax.tick_params(axis="both", which="minor", labelsize=24)
    ax.grid(which="major", alpha=1.0, linewidth=2, linestyle="--")
    ax.grid(which="minor", alpha=0.5, linestyle="--")
    ax.set_title("Precision-Recall", fontsize=24)
    plt.plot(
        x_values,
        y_values,
        marker=".",
        color=color,
        aa=True,
        alpha=0.9,
        linewidth=5,
        label=f"AP: {score:.04f}",
    )
    plt.legend(loc="lower center", fontsize=24)
    plot = fig2numpy(fig)
    return plot, score
