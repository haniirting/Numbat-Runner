
import json
import struct
import requests
from urllib.parse import quote

REPO = "burner22/numbat"
SHARD = "model-00060-of-00772.safetensors"
EXPERT_ID = 100

URL = (
    f"https://huggingface.co/{REPO}/resolve/main/"
    f"{quote(SHARD)}"
)

def get_range(start, end):
    r = requests.get(
        URL,
        headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity"
        },
        timeout=120
    )

    if r.status_code != 206:
        raise RuntimeError(f"HTTP {r.status_code}")

    expected = end - start + 1

    if not r.headers.get("Content-Range", "").startswith(
        f"bytes {start}-{end}/"
    ):
        raise RuntimeError("Invalid Content-Range")

    if len(r.content) != expected:
        raise RuntimeError("Incomplete range")

    return r.content


def main():
    print("NUMBAT EXPERT INSPECTOR")
    print("=" * 55)

    first = get_range(0, 7)
    header_size = struct.unpack("<Q", first)[0]

    if not 0 < header_size <= 64 * 1024 * 1024:
        raise RuntimeError("Invalid header size")

    raw = get_range(8, 8 + header_size - 1)
    header = json.loads(raw.decode("utf-8"))

    expert_marker = f"experts.{EXPERT_ID}."

    expert_tensors = []
    all_sizes = []

    for name, info in header.items():
        if name == "__metadata__":
            continue

        start, end = info["data_offsets"]
        size = end - start

        all_sizes.append((size, name, info))

        if expert_marker in name:
            expert_tensors.append((size, name, info))

    print(f"Shard: {SHARD}")
    print(f"Expert: {EXPERT_ID}")
    print(f"Expert tensors found: {len(expert_tensors)}")

    print("\nEXPERT TENSORS:")

    for size, name, info in sorted(
        expert_tensors, key=lambda x: x[0]
    ):
        print(
            f"{size:>14} bytes | "
            f"{info['dtype']} | "
            f"{info['shape']} | {name}"
        )

    print("\nLARGEST TENSORS IN SHARD:")

    for size, name, info in sorted(
        all_sizes, key=lambda x: x[0], reverse=True
    )[:15]:
        print(
            f"{size:>14} bytes | "
            f"{info['dtype']} | "
            f"{info['shape']} | {name}"
        )

    print("\nINSPECTION COMPLETE")
    print("Only Safetensors metadata was downloaded.")
    print("No weight data was downloaded.")


if __name__ == "__main__":
    main()
