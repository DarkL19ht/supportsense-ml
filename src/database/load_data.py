from pathlib import Path

import pandas as pd

from connection import get_engine


TRAIN_PATH = Path("data/raw/train.csv")
TEST_PATH = Path("data/raw/test.csv")


def load_csv_files():
    """Read the BANKING77 train and test CSV files."""

    print("Reading CSV files...")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    return train_df, test_df


def prepare_data(train_df, test_df):
    """Prepare the datasets before loading into PostgreSQL."""

    train_df = train_df.copy()
    test_df = test_df.copy()

    # Remember which original dataset each row came from.
    train_df["dataset_split"] = "train"
    test_df["dataset_split"] = "test"

    # Calculate the number of words in each customer query.
    train_df["word_count"] = (
        train_df["text"]
        .astype(str)
        .str.split()
        .str.len()
    )

    test_df["word_count"] = (
        test_df["text"]
        .astype(str)
        .str.split()
        .str.len()
    )

    # Combine train and test into one DataFrame.
    combined_df = pd.concat(
        [train_df, test_df],
        ignore_index=True
    )

    print(f"Combined rows: {len(combined_df)}")

    return combined_df


def upload_to_postgres(df):
    """Upload the prepared data to PostgreSQL."""

    print("Connecting to PostgreSQL...")

    engine = get_engine()

    print("Uploading data...")

    df.to_sql(
        name="support_queries",
        con=engine,
        if_exists="replace",
        index=False,
    )

    print("Upload complete!")


def main():
    train_df, test_df = load_csv_files()

    combined_df = prepare_data(
        train_df,
        test_df
    )

    print("\nPreview:")
    print(combined_df.head())

    upload_to_postgres(combined_df)


if __name__ == "__main__":
    main()