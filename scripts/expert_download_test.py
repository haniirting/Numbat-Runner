
import json
import struct
import hashlib
import tempfile
from pathlib import Path

import requests

REPO = "burner22/numbat"
SHARD = "model-00060-of-00772.safetensors"
TENSOR = "model.layers.10.mlp.experts.100.gate_proj.weight"

URL = f"https://huggingface.co/{REPO}/resolve/main/{SHARD}"


def get_range(start, end):
    response = requests.get(
        URL,
        headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity"
        },
        timeout=180
    )

    if response.status_code != 206:
        raise RuntimeError(f"HTTP {response.status_code}")

    expected = end - start + 1

    if not response.headers.get("Content-Range", "").startswith(
        f"bytes {start}-{end}/"
    ):
        raise RuntimeError("Invalid Content-Range")

    if len(response.content) != expected:
        raise RuntimeError("Incomplete download")

    return response.content


def main():
    print("NUMBAT REAL WEIGHT DOWNLOAD TEST")
    print("=" * 55)

    # Read Safetensors header
    first = get_range(0, 7)
    header_size = struct.unpack("<Q", first)[0]

    if not 0 < header_size <= 64 * 1024 * 1024:
        raise RuntimeError("Invalid header")

    raw = get_range(8, 8 + header_size - 1)
    header = json.loads(raw.decode("utf-8"))

    if TENSOR not in header:
        raise RuntimeError("Tensor not found in this shard")

    info = header[TENSOR]
    start, end = info["data_offsets"]
    size = end - start

    print("Tensor:", TENSOR)
    print("Dtype:", info["dtype"])
    print("Shape:", info["shape"])
    print("Expected bytes:", size)

    absolute_start = 8 + header_size + start
    absolute_end = absolute_start + size - 1

    # Download only the selected matrix
    data = get_range(absolute_start, absolute_end)

    if len(data) != size:
        raise RuntimeError("Size mismatch")

    checksum = hashlib.sha256(data).hexdigest()

    # Temporary file, automatically deleted
    with tempfile.TemporaryDirectory(prefix="numbat_") as folder:
        path = Path(folder) / "expert_weight.bin"
        path.write_bytes(data)

        print("Downloaded:", path.stat().st_size, "bytes")
        print("SHA256:", checksum)
        print("Temporary file created: YES")

    print("Temporary file deleted: YES")
    print("Full shard downloaded: NO")
    print("REAL WEIGHT DOWNLOAD SUCCESS")


if __name__ == "__main__":
    main()
