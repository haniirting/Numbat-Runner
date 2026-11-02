
import json
import math
import struct
import hashlib
import requests

REPO = "burner22/numbat"
SHARD = "model-00060-of-00772.safetensors"
PREFIX = "model.layers.10.mlp.experts.100.gate_proj"

TENSORS = {
    "weight": PREFIX + ".weight",
    "weight_scale": PREFIX + ".weight_scale",
    "weight_scale_2": PREFIX + ".weight_scale_2",
    "input_scale": PREFIX + ".input_scale",
}

URL = f"https://huggingface.co/{REPO}/resolve/main/{SHARD}"


def get_range(start, end):
    r = requests.get(
        URL,
        headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
        },
        timeout=180,
    )

    if r.status_code != 206:
        raise RuntimeError(f"HTTP {r.status_code}")

    expected = f"bytes {start}-{end}/"
    if not r.headers.get("Content-Range", "").startswith(expected):
        raise RuntimeError("Invalid Content-Range")

    if len(r.content) != end - start + 1:
        raise RuntimeError("Incomplete download")

    return r.content


def read_header():
    first = get_range(0, 7)
    size = struct.unpack("<Q", first)[0]

    if not 0 < size <= 64 * 1024 * 1024:
        raise RuntimeError("Invalid Safetensors header")

    raw = get_range(8, 8 + size - 1)
    return json.loads(raw.decode("utf-8")), 8 + size


def download_tensor(header, data_start, name):
    if name not in header:
        raise RuntimeError(f"Missing tensor: {name}")

    info = header[name]
    start, end = info["data_offsets"]
    size = end - start

    if not 0 < size <= 32 * 1024 * 1024:
        raise RuntimeError(f"Unexpected tensor size: {size}")

    data = get_range(data_start + start, data_start + end - 1)

    if len(data) != size:
        raise RuntimeError("Tensor size mismatch")

    return info, data


def e2m1(n):
    values = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    value = values[n & 7]
    return -value if n & 8 else value


def e4m3fn(byte):
    sign = -1.0 if byte & 128 else 1.0
    exponent = (byte >> 3) & 15
    mantissa = byte & 7

    if exponent == 0:
        value = mantissa * (2.0 ** -9)
    elif exponent == 15 and mantissa == 7:
        return float("nan")
    else:
        value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))

    return sign * value


def main():
    print("NUMBAT NVFP4 WEIGHT DECODER INSPECTOR")
    print("=" * 55)

    header, data_start = read_header()
    loaded = {}

    for key, name in TENSORS.items():
        info, data = download_tensor(header, data_start, name)
        loaded[key] = (info, data)

        print()
        print(key)
        print("Dtype:", info["dtype"])
        print("Shape:", info["shape"])
        print("Bytes:", len(data))
        print("SHA256:", hashlib.sha256(data).hexdigest())

    weight_info, weight = loaded["weight"]
    scale_info, scales = loaded["weight_scale"]
    scale2_info, scale2 = loaded["weight_scale_2"]
    input_info, input_scale_data = loaded["input_scale"]

    if weight_info["dtype"] != "U8":
        raise RuntimeError("Weight must be U8")

    if scale_info["dtype"] != "F8_E4M3":
        raise RuntimeError("Block scales must be F8_E4M3")

    if scale2_info["dtype"] != "F32":
        raise RuntimeError("Global scale must be F32")

    if input_info["dtype"] != "F32":
        raise RuntimeError("Input scale must be F32")

    rows, packed_columns = weight_info["shape"]
    scale_rows, scale_columns = scale_info["shape"]

    logical_columns = packed_columns * 2

    if rows != scale_rows:
        raise RuntimeError("Row count mismatch")

    if logical_columns != scale_columns * 16:
        raise RuntimeError("NVFP4 group size mismatch")

    if len(weight) != rows * packed_columns:
        raise RuntimeError("Weight byte count mismatch")

    if len(scales) != rows * scale_columns:
        raise RuntimeError("Scale byte count mismatch")

    if len(scale2) != 4 or len(input_scale_data) != 4:
        raise RuntimeError("Invalid scalar size")

    global_scale = struct.unpack("<f", scale2)[0]
    input_scale = struct.unpack("<f", input_scale_data)[0]

    if not math.isfinite(global_scale):
        raise RuntimeError("Invalid global scale")

    if not math.isfinite(input_scale):
        raise RuntimeError("Invalid input scale")

    # Count packed E2M1 values without creating a huge decoded matrix.
    counts = [0] * 16

    for byte in weight:
        counts[byte & 15] += 1
        counts[(byte >> 4) & 15] += 1

    total_values = len(weight) * 2
    zero_values = counts[0] + counts[8]
    negative_values = sum(counts[8:])

    # Inspect E4M3 scales.
    scale_min = float("inf")
    scale_max = float("-inf")
    finite_count = 0
    nan_count = 0

    for byte in scales:
        value = e4m3fn(byte)

        if math.isfinite(value):
            finite_count += 1
            scale_min = min(scale_min, value)
            scale_max = max(scale_max, value)
        else:
            nan_count += 1

    # Decode the first group of 16 logical weight values.
    first_values = []

    for i in range(8):
        byte = weight[i]
        first_values.append(e2m1(byte & 15))
        first_values.append(e2m1((byte >> 4) & 15))

    first_block_scale = e4m3fn(scales[0])

    reconstructed = [
        value * first_block_scale * global_scale
        for value in first_values
    ]

    print()
    print("NVFP4 STRUCTURE CHECK")
    print("Packed shape:", [rows, packed_columns])
    print("Logical shape:", [rows, logical_columns])
    print("Group size: 16")
    print("Groups per row:", scale_columns)
    print("Global scale:", global_scale)
    print("Input scale:", input_scale)

    print()
    print("E2M1 DECODE")
    print("Decoded values:", total_values)
    print("Zero values:", zero_values)
    print("Negative values:", negative_values)
    print("Nibble counts:", counts)

    print()
    print("E4M3 SCALE CHECK")
    print("Finite scales:", finite_count)
    print("NaN scales:", nan_count)
    print("Minimum:", scale_min)
    print("Maximum:", scale_max)

    print()
    print("FIRST 16 RECONSTRUCTED VALUES")
    print(reconstructed)

    total_bytes = sum(len(data) for _, data in loaded.values())

    print()
    print("Total downloaded:", total_bytes, "bytes")
    print("Full shard downloaded: NO")
    print("Remote model code executed: NO")
    print("Model inference: NOT RUN")
    print("NVFP4 DECODE INSPECTION SUCCESS")


if __name__ == "__main__":
    main()
