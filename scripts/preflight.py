
import json
import platform
import shutil
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone

MODEL_ID = "burner22/numbat"
API_URL = f"https://huggingface.co/api/models/{MODEL_ID}?blobs=true"


def get_disk_info():
    disk = shutil.disk_usage(".")
    return {
        "total_gb": round(disk.total / (1024 ** 3), 2),
        "free_gb": round(disk.free / (1024 ** 3), 2),
        "used_gb": round(disk.used / (1024 ** 3), 2),
    }


def get_model_info():
    request = urllib.request.Request(
        API_URL,
        headers={"User-Agent": "Numbat-Runner-Preflight/1.0"}
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))

        siblings = data.get("siblings", [])

        return {
            "available": True,
            "model_id": data.get("id", MODEL_ID),
            "files_count": len(siblings),
            "private": data.get("private"),
            "pipeline_tag": data.get("pipeline_tag"),
        }

    except urllib.error.HTTPError as error:
        return {
            "available": False,
            "error": f"HTTP {error.code}"
        }
    except Exception as error:
        return {
            "available": False,
            "error": str(error)
        }


def main():
    report = {
        "project": "Numbat Runner",
        "check_time": datetime.now(timezone.utc).isoformat(),
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
            "architecture": platform.machine(),
            "cpu_count": __import__("os").cpu_count(),
        },
        "disk": get_disk_info(),
        "model": get_model_info(),
        "download_started": False,
        "model_code_executed": False,
    }

    print("=" * 55)
    print("          NUMBAT RUNNER - PREFLIGHT")
    print("=" * 55)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("=" * 55)
    print("PREFLIGHT FINISHED")
    print("No model weights downloaded.")
    print("No remote model code executed.")


if __name__ == "__main__":
    main()
