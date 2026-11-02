
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "burner22/numbat"

def main():
    print("NUMBAT DYNAMIC LOADER")
    print("=" * 50)

    # Пока загружаем только конфигурацию.
    # Весовые shards подключим после проверки их распределения.
    files = ["config.json"]

    temp_dir = Path(tempfile.mkdtemp(prefix="numbat_"))

    try:
        print(f"Temporary directory: {temp_dir}")

        for filename in files:
            print(f"\nDownloading: {filename}")

            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=temp_dir / "model",
            )

            print(f"Downloaded: {Path(path).name}")

        print("\nTemporary model files are ready.")
        print("Model inference is not connected yet.")

    finally:
        print("\nCleaning temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Temporary files deleted.")

if __name__ == "__main__":
    main()
