from pathlib import Path

import joblib


def load_model(
    model_path: Path,
):
    """
    Load a trained model from disk.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(
        model_path
    )


def predict_intents(
    model,
    texts,
):
    """
    Predict intents for a collection of text queries.
    """

    return model.predict(
        texts
    )