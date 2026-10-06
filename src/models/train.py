from pathlib import Path

import joblib

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_model(
    ngram_range,
    min_df,
    c_value,
):
    """
    Build a TF-IDF + Linear SVM classification pipeline.
    """

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=ngram_range,
                min_df=min_df,
            ),
        ),
        (
            "classifier",
            LinearSVC(
                C=c_value,
            ),
        ),
    ])

    return model


def train_model(
    model,
    texts,
    labels,
):
    """
    Train a classification model.
    """

    model.fit(
        texts,
        labels,
    )

    return model


def save_model(
    model,
    output_path: Path,
):
    """
    Save a trained model to disk.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        output_path,
    )