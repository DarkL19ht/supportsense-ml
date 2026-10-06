from pathlib import Path
import urllib.request


TRAIN_URL = (
    "https://raw.githubusercontent.com/"
    "PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/train.csv"
)

TEST_URL = (
    "https://raw.githubusercontent.com/"
    "PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/test.csv"
)

RAW_DATA_DIR = Path("data/raw")


def download_file(url: str, destination: Path) -> None:
    """Download a file if it does not already exist."""

    if destination.exists():
        print(f"{destination} already exists.")
        return

    print(f"Downloading {destination.name}...")
    urllib.request.urlretrieve(url, destination)
    print("Complete.")


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    download_file(
        TRAIN_URL,
        RAW_DATA_DIR / "train.csv"
    )

    download_file(
        TEST_URL,
        RAW_DATA_DIR / "test.csv"
    )


if __name__ == "__main__":
    main()