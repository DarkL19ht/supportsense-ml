from sklearn.pipeline import Pipeline

from src.models.train import (
    build_model,
)


def test_build_model_returns_pipeline():
    model = build_model(
        ngram_range=(1, 2),
        min_df=2,
        c_value=1.0,
    )

    assert isinstance(
        model,
        Pipeline,
    )


def test_model_contains_expected_steps():
    model = build_model(
        ngram_range=(1, 2),
        min_df=2,
        c_value=1.0,
    )

    assert "tfidf" in model.named_steps
    assert "classifier" in model.named_steps