# Build a reusable prediction function
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "supportsense_sklearn.joblib"
)


def load_model(model_path=DEFAULT_MODEL_PATH):
    """
    Load the trained SupportSense sklearn pipeline.
    """
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = joblib.load(model_path)

    if not isinstance(model, Pipeline):
        raise TypeError(
            "Expected a saved sklearn Pipeline. "
            "Check the model artifact structure."
        )

    return model


def predict_intent(text, model):
    """
    Predict a BANKING77 intent from a customer message.

    Returns the intent label as a string.
    """
    if not isinstance(text, str):
        raise TypeError(
            "Customer message must be a string."
        )

    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError(
            "Please enter a customer message."
        )

    if len(cleaned_text) > 2000:
        raise ValueError(
            "Customer message must be 2000 characters or fewer."
        )

    prediction = model.predict(
        [cleaned_text]
    )[0]

    return str(prediction)