from src.evaluation.metrics import (
    calculate_classification_metrics,
)


def test_perfect_predictions():
    y_true = [
        "a",
        "b",
        "c",
    ]

    y_pred = [
        "a",
        "b",
        "c",
    ]

    metrics = (
        calculate_classification_metrics(
            y_true,
            y_pred,
        )
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["macro_f1"] == 1.0