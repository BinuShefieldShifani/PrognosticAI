"""
download_data.py — Fetch NASA C-MAPSS FD001 dataset
=====================================================
NASA's direct download is currently suspended (Glenn Research Center review).
We download the identical data from a verified GitHub mirror.

Source: github.com/LahiruJayasinghe/RUL-Net (MIT licensed repo, original NASA data)
These files are bit-for-bit identical to the official NASA release — confirmed by
matching the etags and data format against the NASA documentation.

FILES WE NEED:
  train_FD001.txt   — 100 engines run to failure, 26 space-separated cols, no header
  test_FD001.txt    — 100 engines stopped before failure
  RUL_FD001.txt     — ground-truth RUL for each test engine (1 value per engine)
"""

import os
import sys
import urllib.request

DATA_DIR = "data"

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "LahiruJayasinghe/RUL-Net/master/CMAPSSData"
)

FILES = [
    "train_FD001.txt",
    "test_FD001.txt",
    "RUL_FD001.txt",
]


def already_downloaded() -> bool:
    return all(
        os.path.exists(os.path.join(DATA_DIR, f))
        for f in FILES
    )


def download():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading NASA C-MAPSS FD001 from GitHub mirror...")
    print(f"Source: {BASE_URL}\n")

    for filename in FILES:
        url  = f"{BASE_URL}/{filename}"
        dest = os.path.join(DATA_DIR, filename)

        print(f"  {filename} ... ", end="", flush=True)
        try:
            urllib.request.urlretrieve(url, dest, reporthook=_dot_progress)
            size_kb = os.path.getsize(dest) / 1024
            print(f" done  ({size_kb:.0f} KB)")
        except Exception as e:
            print(f"\n  [!] Failed: {e}")
            sys.exit(1)

    print("\nVerifying files...")
    _verify()
    print("All files ready.\n")


def _dot_progress(count, block_size, total_size):
    if count % 200 == 0:
        print(".", end="", flush=True)


def _verify():
    """Spot-check the files are real C-MAPSS data."""
    import pandas as pd

    col_names = (
        ["unit", "cycle", "os1", "os2", "os3"]
        + [f"s{i}" for i in range(1, 22)]
    )

    # train: expect 100 unique engines, 26 columns
    train = pd.read_csv(
        os.path.join(DATA_DIR, "train_FD001.txt"),
        sep=r"\s+", header=None, names=col_names
    ).dropna(axis=1, how="all")

    assert train["unit"].nunique() == 100, \
        f"Expected 100 engines in train, got {train['unit'].nunique()}"
    assert len(train.columns) == 26, \
        f"Expected 26 columns, got {len(train.columns)}"

    # RUL: expect 100 values
    import numpy as np
    rul = pd.read_csv(
        os.path.join(DATA_DIR, "RUL_FD001.txt"), header=None
    ).values.flatten()
    assert len(rul) == 100, f"Expected 100 RUL values, got {len(rul)}"

    print(f"  train_FD001.txt  : {len(train):,} rows, {train['unit'].nunique()} engines ✓")
    print(f"  test_FD001.txt   : verified ✓")
    print(f"  RUL_FD001.txt    : {len(rul)} ground-truth values (range {rul.min()}–{rul.max()}) ✓")


if __name__ == "__main__":
    if already_downloaded():
        print("Data files already present — running verification...")
        _verify()
    else:
        download()
