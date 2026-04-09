from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATA_FILE = RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ZIP_FILE = RAW_DIR / "telco-customer-churn.zip"

# Public mirror of the same popular Kaggle dataset file
FALLBACK_CSV_URLS = [
    # IBM public raw CSV for the same Telco churn dataset
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    # Community mirror containing the original Kaggle filename
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def has_kaggle_credentials() -> bool:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    return kaggle_json.exists() and os.access(kaggle_json, os.R_OK)


def try_kaggle_download() -> bool:
    if shutil.which("kaggle") is None:
        return False

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "blastchar/telco-customer-churn",
        "-p",
        str(RAW_DIR),
        "-f",
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if DATA_FILE.exists():
            return True
        if ZIP_FILE.exists():
            with zipfile.ZipFile(ZIP_FILE, "r") as zf:
                zf.extractall(RAW_DIR)
            return DATA_FILE.exists()
        return False
    except subprocess.CalledProcessError:
        return False


def download_from_fallback() -> bool:
    for url in FALLBACK_CSV_URLS:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            DATA_FILE.write_bytes(resp.content)
            if DATA_FILE.exists() and DATA_FILE.stat().st_size > 0:
                return True
        except requests.RequestException:
            continue
    return False


def main() -> int:
    ensure_dirs()

    if DATA_FILE.exists() and DATA_FILE.stat().st_size > 0:
        print(f"Dataset already present: {DATA_FILE}")
        return 0

    kaggle_ok = False
    if has_kaggle_credentials():
        print("Trying Kaggle API download...")
        kaggle_ok = try_kaggle_download()

    if kaggle_ok:
        print(f"Downloaded dataset via Kaggle API: {DATA_FILE}")
        return 0

    print("Kaggle download unavailable/failed. Trying fallback source...")
    fallback_ok = download_from_fallback()
    if fallback_ok:
        print(f"Downloaded dataset via fallback URL: {DATA_FILE}")
        return 0

    print("Failed to download dataset from both Kaggle and fallback source.")
    print("Please configure Kaggle API credentials and retry.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
