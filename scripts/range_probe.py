
import json
import struct
import requests
from urllib.parse import quote

REPO = "burner22/numbat"
SHARD = "model-00060-of-00772.safetensors"

BASE = (
    f"https://huggingface.co/{REPO}/resolve/main/"
    f"{quote(SHARD)}"
)

MAX_HEADER = 64 * 1024 * 1024
MAX_TENSOR = 64 * 1024 * 1024


def get_range(start, end):
    response = requests.get(
        BASE,
        headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": "Numbat-Runner/1.0"
        },
        stream=True,
        timeout=120
    )

    if response.status_code != 206:
        response.close()
        raise RuntimeError(
            f"Range requests unavailable: HTTP {response.status_code}"
        )

    expected = end - start + 1
    content_range = response.headers.get("Content-Range", "")

    if not content_range.startswith(f"bytes {start}-{end}/"):
        response.close()
        raise RuntimeError(f"Unexpected Content-Range: {content_range}")

    data = bytearray()

    try:
        for chunk in response.iter_content(65536):
            data.extend(chunk)
            if len(data) > expected:
                raise RuntimeError("Too much data received")
    finally:
        response.close()

    if len(data) != expected:
        raise RuntimeError(
            f"Incomplete download: {len(data)} / {expected}"
        )

    return bytes(data)


def main():
    print("NUMBAT TENSOR RANGE PROBE")
    print("=" * 55)
    print("Shard:", SHARD)

    first = get_range(0, 7)
    header_size = struct.unpack("<Q", first)[0]

    if not 0 < header_size <= MAX_HEADER:
        raise RuntimeError("Invalid Safetensors header")

    header_bytes = get_range(8, 8 + header_size - 1)
    header = json.loads(header_bytes.decode("utf-8"))

    tensors = []

    for name, info in header.items():
        if name == "__metadata__":
            continue

        start, end = info["data_offsets"]
        size = end - start
        shape = info.get("shape", [])

        if size > 0 and shape:
            tensors.append((size, name, info))

    print("Total tensors:", len(header) - 1)
    print("Non-scalar tensors:", len(tensors))

    tensors.sort(key=lambda item: item[0])

    print("\nSmallest tensors:")
    for size, name, info in tensors[:15]:
        print(
            f"{size:>12} bytes | "
            f"{info['dtype']} | "
            f"{info['shape']} | {name}"
        )

    candidates = [
        item for item in tensors
        if item[0] <= MAX_TENSOR
    ]

    if not candidates:
        print("\nNo tensor up to 64 MiB found.")
        print("No tensor data downloaded.")
        return

    size, name, info = candidates[0]

    print("\nSelected tensor:", name)
    print("Dtype:", info["dtype"])
    print("Shape:", info["shape"])
    print("Size:", size, "bytes")

    data_start = 8 + header_size + info["data_offsets"][0]
    data_end = data_start + size - 1

    tensor_data = get_range(data_start, data_end)

    print("\nDownloaded bytes:", len(tensor_data))
    print("Full shard downloaded: NO")
    print("Tensor data stored temporarily in RAM.")
    print("RANGE PROBE SUCCESS")


if __name__ == "__main__":
    main()
