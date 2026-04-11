"""
Script tải dữ liệu mẫu từ PhysioNet (không cần đăng ký tài khoản).
Tải 3 file nhỏ nhất để test: n16 (29MB), n12 (51MB), sdb1 (138MB).
"""

import urllib.request
import os
from pathlib import Path

BASE_URL = "https://physionet.org/files/capslpdb/1.0.0"
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Danh sách file cần tải (chọn file nhỏ nhất để bắt đầu)
FILES = [
    # (filename, mô tả)
    ("n16.edf",  "Healthy subject - 29MB"),
    ("n16.txt",  "Annotations for n16"),
    ("n12.edf",  "Healthy subject - 51MB"),
    ("n12.txt",  "Annotations for n12"),
    ("sdb1.edf", "Sleep-disordered breathing - 138MB"),
    ("sdb1.txt", "Annotations for sdb1"),
    ("ins1.edf", "Insomnia - 252MB"),
    ("ins1.txt", "Annotations for ins1"),
]


def download_file(filename: str, description: str):
    dest = DATA_DIR / filename
    if dest.exists():
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"  [SKIP] {filename} already exists ({size_mb:.1f} MB)")
        return

    url = f"{BASE_URL}/{filename}"
    print(f"  Downloading {filename} ({description})...")

    def progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size * 100 // total_size, 100)
            mb_done = count * block_size / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            print(f"\r    {pct}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"\r    Done — {size_mb:.1f} MB saved to {dest}")


if __name__ == "__main__":
    print("=== Downloading CAP Sleep DB sample files ===\n")
    for fname, desc in FILES:
        download_file(fname, desc)
    print("\nAll files downloaded to data/raw/")
    print("Files:")
    for f in sorted(DATA_DIR.glob("*")):
        print(f"  {f.name:30s}  {f.stat().st_size/1024/1024:.1f} MB")
