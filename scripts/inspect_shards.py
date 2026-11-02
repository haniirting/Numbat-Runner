
import json
import re
import urllib.request
from collections import Counter

import ijson

BASE = "https://huggingface.co/burner22/numbat/resolve/main"
MAX_INDEX_BYTES = 2 * 1024 * 1024 * 1024


def fetch_config():
    request = urllib.request.Request(
        f"{BASE}/config.json",
        headers={"User-Agent": "Numbat-Runner/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def inspect_index():
    request = urllib.request.Request(
        f"{BASE}/model.safetensors.index.json",
        headers={
            "User-Agent": "Numbat-Runner/1.0",
            "Accept-Encoding": "identity"
        }
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        size = response.headers.get("Content-Length")

        if size and int(size) > MAX_INDEX_BYTES:
            raise RuntimeError(
                f"Index is too large: {int(size) / 1024**3:.2f} GB"
            )

        shards = Counter()
        experts = set()
        layers = set()
        tensor_count = 0

        for tensor_name, filename in ijson.kvitems(
            response, "weight_map"
        ):
            tensor_count += 1
            shards[filename] += 1

            layer = re.search(r"layers\.(\d+)", tensor_name)
            expert = re.search(r"experts\.(\d+)", tensor_name)

            if layer:
                layers.add(int(layer.group(1)))

            if expert:
                experts.add(int(expert.group(1)))

    return tensor_count, shards, layers, experts


def main():
    print("NUMBAT SHARD INSPECTOR")
    print("=" * 55)

    config = fetch_config()
    tensor_count, shards, layers, experts = inspect_index()

    moe = config.get("moe_config", {})

    print(f"Model type: {config.get('model_type')}")
    print(f"Configured layers: {config.get('num_hidden_layers')}")
    print(f"Mapped tensors: {tensor_count}")
    print(f"Unique shards: {len(shards)}")
    print(f"Detected layers: {len(layers)}")
    print(f"Detected experts: {len(experts)}")
    print(f"Configured experts: {moe.get('n_routed_experts')}")
    print(f"Experts per token: {moe.get('num_experts_per_tok')}")

    print("\nLargest shard groups:")
    for filename, count in shards.most_common(15):
        print(f"{filename}: {count} tensors")

    print("\nCHECK COMPLETE")
    print("No model weights downloaded.")
    print("No remote Python code executed.")


if __name__ == "__main__":
    main()
