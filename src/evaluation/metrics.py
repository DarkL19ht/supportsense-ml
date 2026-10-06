from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def calculate_classification_metrics(
    y_true,
    y_pred,
):
    """
    Calculate multiclass classification metrics.

    Returns accuracy and macro-averaged precision,
    recall, and F1.
    """

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }