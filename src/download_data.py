"""download_data.py
-----------------
Utility to download the UCI Credit Card Default dataset.
If the dataset file already exists, the download is skipped.
"""
import os
import pathlib
import urllib.request

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
DATA_FILE = DATA_DIR / "default_of_credit_card_clients.xls"

def download_dataset(force: bool = False) -> pathlib.Path:
    """Download the dataset to ``data/``.

    Parameters
    ----------
    force: bool, optional
        Re-download even if the file already exists.
    Returns
    -------
    pathlib.Path
        Path to the downloaded file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists() and not force:
        print(f"Dataset already present at {DATA_FILE}")
        return DATA_FILE
    print(f"Downloading dataset from {DATA_URL} ...")
    try:
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}")
    print(f"Saved to {DATA_FILE}")
    return DATA_FILE

if __name__ == "__main__":
    download_dataset()
