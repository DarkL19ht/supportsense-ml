from pathlib import Path

import pandas as pd


def load_banking77_data(
    data_dir: Path,
):
    """
    Load the BANKING77 train and test CSV files.

    Parameters
    ----------
    data_dir:
        Directory containing train.csv and test.csv.

    Returns
    -------
    tuple
        train_df and test_df.
    """

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found: {test_path}"
        )

    train_df = pd.read_csv(
        train_path
    )

    test_df = pd.read_csv(
        test_path
    )

    return train_df, test_df