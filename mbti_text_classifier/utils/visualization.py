from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(save_path)
    return fig


def save_class_distribution(y_train, y_val, class_names, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_counts = pd.Series(y_train).value_counts().reindex(class_names, fill_value=0)
    val_counts = pd.Series(y_val).value_counts().reindex(class_names, fill_value=0)

    x_positions = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x_positions - width / 2, train_counts, width, label="Train")
    ax.bar(x_positions + width / 2, val_counts, width, label="Validation")

    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path)
    return fig


def save_training_curves(metrics_history, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "train_loss" in metrics_history and "val_loss" in metrics_history:
        ax = axes[0]
        ax.plot(metrics_history["train_loss"], label="Train Loss", marker="o")
        ax.plot(metrics_history["val_loss"], label="Val Loss", marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if "train_accuracy" in metrics_history and "val_accuracy" in metrics_history:
        ax = axes[1]
        ax.plot(metrics_history["train_accuracy"], label="Train Acc", marker="o")
        ax.plot(metrics_history["val_accuracy"], label="Val Acc", marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training and Validation Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path)
    return fig
