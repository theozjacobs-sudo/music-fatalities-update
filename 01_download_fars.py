"""
Download FARS (Fatality Analysis Reporting System) data from NHTSA for 2017-2022.
FARS provides population-based data on all fatal crashes on public roads in the U.S.
"""

import os
import zipfile
import io
import requests
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# FARS data is available as CSV zip files from NHTSA
# URL pattern: https://static.nhtsa.gov/nhtsa/downloads/FARS/{year}/National/FARS{year}NationalCSV.zip
FARS_BASE_URL = "https://static.nhtsa.gov/nhtsa/downloads/FARS"

YEARS = range(2017, 2023)  # 2017 through 2022

def download_fars_year(year, max_retries=4):
    """Download and extract FARS data for a given year."""
    url = f"{FARS_BASE_URL}/{year}/National/FARS{year}NationalCSV.zip"
    year_dir = os.path.join(DATA_DIR, f"fars_{year}")

    # Check if already downloaded
    if os.path.exists(year_dir) and len(os.listdir(year_dir)) > 0:
        print(f"  FARS {year} already downloaded, skipping.")
        return True

    os.makedirs(year_dir, exist_ok=True)

    for attempt in range(max_retries):
        try:
            print(f"  Downloading FARS {year} from {url}...")
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                zf.extractall(year_dir)

            print(f"  FARS {year} downloaded and extracted ({len(os.listdir(year_dir))} files).")
            return True
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"  Attempt {attempt+1} failed for FARS {year}: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  FAILED to download FARS {year} after {max_retries} attempts.")
                return False


def main():
    print("Downloading FARS data (2017-2022)...")
    for year in YEARS:
        print(f"\nYear {year}:")
        download_fars_year(year)

    # List what we have
    print("\n\nDownloaded data summary:")
    for year in YEARS:
        year_dir = os.path.join(DATA_DIR, f"fars_{year}")
        if os.path.exists(year_dir):
            files = os.listdir(year_dir)
            print(f"  {year}: {len(files)} files")
            for f in sorted(files):
                size = os.path.getsize(os.path.join(year_dir, f))
                print(f"    {f} ({size:,} bytes)")
        else:
            print(f"  {year}: NOT DOWNLOADED")


if __name__ == "__main__":
    main()
