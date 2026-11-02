
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
MIN_TENSOR = 1024
MAX_TENSOR = 1024 * 1024


def get_range(start, end):
    response = requests.get(
        BASE,
        headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": "Numbat-Runner/1.0"
        },
        stream=True,
        timeout=90,
        allow_redirects=True
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
        raise RuntimeError(
            f"Unexpected Content-Range: {content_range}"
        )

    data = bytearray()

    try:
        for chunk in response.iter_content(65536):
            data.extend(chunk)

            if len(data) > expected:
                raise RuntimeError("Server returned too much data")
    finally:
        response.close()

    if len(data) != expected:
        raise RuntimeError(
            f"Incomplete range: {len(data)} / {expected}"
        )

    return bytes(data)


def main():
    print("NUMBAT RANGE PROBE")
    print("=" * 50)
    print("Shard:", SHARD)

    # 1. Читаем длину заголовка Safetensors
    first = get_range(0, 7)
    header_size = struct.unpack("<Q", first)[0]

    if not 0 < header_size <= MAX_HEADER:
        raise RuntimeError(
            f"Invalid header size: {header_size}"
        )

    print("Header size:", header_size)

    # 2. Загружаем только заголовок
    header_bytes = get_range(
        8,
        8 + header_size - 1
    )

    header = json.loads(header_bytes.decode("utf-8"))

    # 3. Ищем настоящий небольшой тензор
    tensors = []

    for name, info in header.items():
        if name == "__metadata__":
            continue

        start, end = info["data_offsets"]
        size = end - start

        if (
            MIN_TENSOR <= size <= MAX_TENSOR
            and info.get("shape")
        ):
            tensors.append((size, name, info))

    print("Tensors in shard:", len(header) - 1)
    print("Suitable tensors:", len(tensors))

    if not tensors:
        print("No suitable tensor found.")
        return

    # Берём самый маленький подходящий тензор
    size, name, info = min(
        tensors,
        key=lambda item: item[0]
    )

    print("\nSelected tensor:", name)
    print("Dtype:", info["dtype"])
    print("Shape:", info["shape"])
    print("Size:", size, "bytes")

    # 4. Вычисляем абсолютные смещения
    data_start = (
        8
        + header_size
        + info["data_offsets"][0]
    )

    data_end = data_start + size - 1

    # 5. Загружаем только данные этого тензора
    tensor_data = get_range(data_start, data_end)

    print("\nDownloaded bytes:", len(tensor_data))
    print("Full shard downloaded: NO")
    print("Tensor stored temporarily in RAM.")
    print("RANGE PROBE SUCCESS")


if __name__ == "__main__":
    main()
