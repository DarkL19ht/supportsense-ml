# Adding automated tests

import pytest

from src.models.app_predict import (
    load_model,
    predict_intent,
)


@pytest.fixture(scope="module")
def trained_model():
    return load_model()


def test_model_loads(trained_model):
    assert hasattr(
        trained_model,
        "predict",
    )


def test_prediction_returns_string(trained_model):
    result = predict_intent(
        "My card has not arrived",
        trained_model,
    )

    assert isinstance(
        result,
        str,
    )

    assert len(result) > 0


def test_prediction_is_valid_class(trained_model):
    result = predict_intent(
        "I forgot my card PIN",
        trained_model,
    )

    valid_classes = {
        str(label)
        for label in trained_model.classes_
    }

    assert result in valid_classes


def test_empty_input_rejected(trained_model):
    with pytest.raises(
        ValueError,
        match="Please enter",
    ):
        predict_intent(
            "",
            trained_model,
        )


def test_whitespace_input_rejected(trained_model):
    with pytest.raises(ValueError):
        predict_intent(
            "   ",
            trained_model,
        )


def test_non_string_input_rejected(trained_model):
    with pytest.raises(TypeError):
        predict_intent(
            None,
            trained_model,
        )


def test_very_long_input_rejected(trained_model):
    with pytest.raises(ValueError):
        predict_intent(
            "a" * 2001,
            trained_model,
        )


def test_missing_model_file():
    with pytest.raises(FileNotFoundError):
        load_model(
            "models/does_not_exist.joblib"
        )

def test_model_prediction_is_repeatable(trained_model):
    message = "My card has not arrived yet"

    first_prediction = predict_intent(
        message,
        trained_model,
    )

    second_prediction = predict_intent(
        message,
        trained_model,
    )

    assert first_prediction == second_prediction