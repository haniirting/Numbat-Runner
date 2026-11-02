#!/usr/bin/env python3
"""
Numbat full repository downloader.

Downloads every file from https://huggingface.co/burner22/numbat
to a local folder using Hugging Face's official client.

IMPORTANT:
- The repository has been reported to contain about 7.72 TB of weights.
- You need substantially more free disk space than that (8 TB is a bare minimum;
  9 TB or more is safer).
- This script does NOT run the model.
- It does not execute the repository's custom Python code.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ID = "burner22/numbat"
DEFAULT_FOLDER = Path("D:/Numbat-Model") if sys.platform == "win32" else Path.home() / "Numbat-Model"
MIN_FREE_BYTES = 8_000_000_000_000  # 8 TB decimal; safety floor, not a guarantee


def ensure_huggingface_hub() -> None:
    if importlib.util.find_spec("huggingface_hub") is None:
        print("Installing the required package: huggingface_hub")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "huggingface_hub"]
        )


def main() -> int:
    print("=" * 68)
    print("NUMBAT — FULL DOWNLOAD")
    print("=" * 68)
    print(f"Repository: https://huggingface.co/{REPO_ID}")
    print("This downloads the repository files, including all model weight shards.")
    print("It does NOT run the model or verify that the repository is authentic.")
    print()

    target_text = input(f"Download folder [{DEFAULT_FOLDER}]: ").strip()
    target = Path(target_text).expanduser() if target_text else DEFAULT_FOLDER
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)

    free = shutil.disk_usage(target).free
    print(f"\nTarget: {target}")
    print(f"Free space: {free / 1_000_000_000:.2f} GB")

    if free < MIN_FREE_BYTES:
        print("\nSTOP: not enough free space for a safe full download.")
        print("The model repository is reported to be about 7.72 TB.")
        print("Use a drive with at least 8 TB free; 9 TB+ is recommended.")
        return 2

    ensure_huggingface_hub()
    from huggingface_hub import snapshot_download

    print("\nChecking the remote file list and estimated download size...")
    try:
        plan = snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(target),
            max_workers=4,
            dry_run=True,
        )
    except Exception as exc:
        print(f"\nCould not inspect the Hugging Face repository: {exc}")
        return 1

    total = 0
    for item in plan:
        size = getattr(item, "file_size", None)
        if size is None:
            size = getattr(item, "size", 0)
        total += int(size or 0)

    print(f"Files to download: {len(plan)}")
    print(f"Estimated size: {total / 1_000_000_000:.2f} GB")
    print(f"Free space now: {shutil.disk_usage(target).free / 1_000_000_000:.2f} GB")

    free = shutil.disk_usage(target).free
    if total and free < total + 100_000_000_000:
        print("\nSTOP: not enough free space for the reported files plus 100 GB reserve.")
        return 2

    print("\nThis may take a very long time and consume several terabytes of bandwidth.")
    print("Nothing has been downloaded yet.")
    confirmation = input(
        f'To start, type exactly DOWNLOAD {total // 1_000_000_000} GB: '
    ).strip()
    expected = f"DOWNLOAD {total // 1_000_000_000} GB"
    if confirmation != expected:
        print("Cancelled. No model files were downloaded.")
        return 0

    print("\nStarting download. Keep this computer and drive powered on.")
    try:
        result = snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(target),
            max_workers=4,
        )
    except KeyboardInterrupt:
        print("\nStopped by user. Re-run this script to resume; cached files are reused.")
        return 130
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        print("You can re-run this script; Hugging Face's local download cache may resume files.")
        return 1

    print("\nDownload completed.")
    print(f"Local folder: {result}")
    print("Note: downloaded files are not proof that the model is genuine or safe to execute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
