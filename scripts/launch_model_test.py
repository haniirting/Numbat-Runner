import json
import os
import shutil
import sys
import traceback
import subprocess
from datetime import datetime, timezone

import requests

MODEL_ID = "burner22/numbat"
API_URL = f"https://huggingface.co/api/models/{MODEL_ID}?blobs=true"
LOG_PATH = "numbat_launch.log"


def log(message):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def inspect_repository():
    log("Connecting to Hugging Face...")

    response = requests.get(API_URL, timeout=60)
    response.raise_for_status()

    info = response.json()
    files = info.get("siblings", [])

    total = 0
    largest = []

    for item in files:
        size = item.get("size")
        lfs = item.get("lfs") or {}

        if not isinstance(size, int):
            size = lfs.get("size")

        if isinstance(size, int):
            total += size
            largest.append(
                (size, item.get("rfilename", "?"))
            )

    largest.sort(reverse=True)

    log(f"Model files: {len(files)}")
    log(f"Total known size: {total / 1024**3:.2f} GiB")

    for size, name in largest[:10]:
        log(f"{size:,} bytes | {name}")

    return total


def main():
    open(LOG_PATH, "w", encoding="utf-8").close()

    log("=" * 60)
    log("NUMBAT REAL MODEL LAUNCH ATTEMPT")
    log("=" * 60)

    disk = shutil.disk_usage(".")

    log(f"Python: {sys.version}")
    log(f"CPU cores: {os.cpu_count()}")
    log(f"Disk free: {disk.free / 1024**3:.2f} GiB")

    try:
        total_bytes = inspect_repository()

        free_bytes = shutil.disk_usage(".").free
        safe_budget = int(free_bytes * 0.80)

        if total_bytes <= 0:
            raise RuntimeError(
                "Could not determine model size."
            )

        if total_bytes > safe_budget:
            log("MODEL_LOAD_BLOCKED_BY_RUNNER_DISK_LIMIT")
            log("Model is larger than available safe storage.")
            log("No model weights downloaded.")
            log("No remote model code executed.")
            return 2

        log("Storage check passed.")

        # Install inference dependencies only after
        # confirming that the model can fit on disk.
        log("Installing inference dependencies...")

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "transformers",
                "accelerate",
                "safetensors",
            ],
            check=True,
        )

        import torch

        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
        )

        log(f"PyTorch: {torch.__version__}")
        log(f"CUDA: {torch.cuda.is_available()}")

        log("Loading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )

        log("TOKENIZER LOADED")

        log("Loading actual model weights...")
        log("Remote repository model code may execute.")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="cpu",
        )

        log("MODEL LOADED SUCCESSFULLY")

        prompt = "Hello! Reply with one short sentence."

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        )

        log("Starting inference...")

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
            )

        answer = tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )

        log("=" * 60)
        log("INFERENCE SUCCESS")
        log("GENERATED RESPONSE:")
        log(answer)
        log("=" * 60)

        return 0

    except Exception:
        log("MODEL LAUNCH FAILED")
        log(traceback.format_exc())

        return 1


if __name__ == "__main__":
    sys.exit(main())
