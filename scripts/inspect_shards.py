
import json
import re
import urllib.request
from collections import Counter

BASE = "https://huggingface.co/burner22/numbat/resolve/main"

def fetch_json(filename):
    url = f"{BASE}/{filename}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Numbat-Runner/1.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(128 * 1024 * 1024 + 1)

    if len(raw) > 128 * 1024 * 1024:
        raise RuntimeError(f"{filename}: файл слишком большой")

    return json.loads(raw.decode("utf-8"))


def main():
    print("NUMBAT SHARD INSPECTOR")
    print("=" * 55)

    config = fetch_json("config.json")
    index = fetch_json("model.safetensors.index.json")

    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("В index.json отсутствует weight_map")

    shards = Counter(weight_map.values())

    experts = set()
    layers = set()

    for tensor_name in weight_map:
        layer = re.search(r"layers\.(\d+)", tensor_name)
        expert = re.search(r"experts\.(\d+)", tensor_name)

        if layer:
            layers.add(int(layer.group(1)))

        if expert:
            experts.add(int(expert.group(1)))

    print(f"Model type: {config.get('model_type')}")
    print(f"Layers in config: {config.get('num_hidden_layers')}")
    print(f"Mapped tensors: {len(weight_map)}")
    print(f"Unique weight shards: {len(shards)}")
    print(f"Detected layer IDs: {len(layers)}")
    print(f"Detected expert IDs: {len(experts)}")

    moe = config.get("moe_config", {})
    print(f"Configured experts: {moe.get('n_routed_experts')}")
    print(f"Experts per token: {moe.get('num_experts_per_tok')}")

    print("\nLargest shard groups by tensor count:")
    for filename, count in shards.most_common(15):
        print(f"{filename}: {count} tensors")

    print("\nCHECK COMPLETE")
    print("No model weights downloaded.")
    print("No remote Python code executed.")


if __name__ == "__main__":
    main()
