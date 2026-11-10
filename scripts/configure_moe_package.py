"""Write the ``config.json`` fields of a Mixture-of-Experts release compiled as one qbruntime MXQ package.

The compiler emits a ``.mxqpkg`` holding every shared / expert / ``lm_head`` MXQ plus its ``manifest.json``. This
script reads that package (``Model`` creation only, no accelerator, no ``launch``) and records in a release
``config.json``:

- ``moe_mxq_package``: the package path relative to the config (``moe_mxq_pack_index`` is removed);
- ``max_batch_size``: the KV entries of the shared MXQs (``CacheInfo.num_batches``) of a BatchLLM package, else 1;
- ``moe_shared_output_order``: the role of each shared MXQ output. The manifest names the residual (the output its
  residual add consumes), the router gate is the output with ``num_experts`` channels, and the remaining output is the
  expert input; the quantization scales (reciprocal at the shared/expert boundary) cross-check that choice;
- ``npu_prefill_chunk_size``: the shared MXQs' per-sequence token width, unless the config already has one;
- with ``--num-devices N``: ``shared_target_cores`` (core 0 of each device), ``expert_target_cores`` (every core
  of every device) and ``lm_head_target_cores`` (core 0 of the last device), the round-robin placement the
  backend expects.

Usage:
    python scripts/configure_moe_package.py release/mxq/model.mxqpkg --config release/config.json --num-devices 4
    python scripts/configure_moe_package.py model.mxqpkg --config release/config.json \\
        --package-path mxq/model.mxqpkg --base-config upstream/config.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Any, Optional

from transformers_mblt.utils.moe_utils import (
    MobilintMixtureOfExpertsPackageLayout,
    read_moe_package_manifest,
)

_EXPERT_IN_SCALE_TOLERANCE = 1e-3
ROLES = ("residual", "expert_in", "gate")
_MOBILINT_FIELDS = {
    "architectures": ["MobilintQwen3MoEForCausalLM"],
    "auto_map": {
        "AutoConfig": "proxy_qwen3_moe.MobilintQwen3MoEConfig",
        "AutoModel": "proxy_qwen3_moe.MobilintQwen3MoEForCausalLM",
        "AutoModelForCausalLM": "proxy_qwen3_moe.MobilintQwen3MoEForCausalLM",
    },
    "model_type": "mobilint-qwen3_moe",
}


def _scale_pair_matches(out_scale: Any, in_scale: Any) -> bool:
    if not (getattr(out_scale, "is_uniform", False) and getattr(in_scale, "is_uniform", False)):
        return False
    return abs(float(out_scale.scale) * float(in_scale.scale) - 1.0) <= _EXPERT_IN_SCALE_TOLERANCE


def derive_package_fields(package_path: str, num_layers: int, num_experts: int) -> dict[str, Any]:
    """Return ``max_batch_size``, ``moe_shared_output_order`` and the per-sequence width read from the package."""
    import qbruntime

    qbruntime.set_log_level(qbruntime.LogLevel.Warn)
    package = qbruntime.ModelPackage(package_path)
    manifest = read_moe_package_manifest(package)
    layout = MobilintMixtureOfExpertsPackageLayout.from_manifest(manifest, num_layers, num_experts)

    orders: list[list[str]] = []
    max_batch_size = 1
    item_width: Optional[int] = None
    for layer_idx in range(num_layers):
        shared = package.create_model(layout.shared[layer_idx])
        expert = package.create_model(layout.experts[layer_idx][0])
        try:
            shapes = shared.get_model_output_shape()
            out_scales = shared.get_output_scale()
            in_scale = expert.get_input_scale()[0]
            if layer_idx == 0:
                item_width = int(shared.get_input_buffer_info()[0].max_width)
                if layout.batch:
                    cache_infos = list(shared.get_cache_infos())
                    if not cache_infos:
                        raise ValueError("the BatchLLM package's shared MXQ reports no KV cache")
                    max_batch_size = min(int(info.num_batches) for info in cache_infos)
        finally:
            shared.dispose()
            expert.dispose()
        gates = [idx for idx, shape in enumerate(shapes) if int(shape[-1]) == num_experts]
        if len(shapes) != 3 or len(gates) != 1:
            raise ValueError(f"layer {layer_idx}: cannot identify the router gate among output shapes {shapes}.")
        gate = gates[0]
        residual = layout.residual_output[layer_idx]
        if residual == gate:
            raise ValueError(f"layer {layer_idx}: the manifest puts the residual on the router gate output {gate}.")
        expert_in = ({0, 1, 2} - {gate, residual}).pop()
        matches = [idx for idx in (residual, expert_in) if _scale_pair_matches(out_scales[idx], in_scale)]
        if matches and expert_in not in matches:
            raise ValueError(
                f"layer {layer_idx}: the manifest puts the residual at output {residual}, but the MXQ scales make "
                f"output {matches} the expert input."
            )
        roles = [""] * 3
        roles[residual], roles[expert_in], roles[gate] = ROLES
        orders.append(roles)

    default = collections.Counter(tuple(order) for order in orders).most_common(1)[0][0]
    compact: dict[str, list[str]] = {"default": list(default)}
    compact.update({str(layer_idx): order for layer_idx, order in enumerate(orders) if tuple(order) != default})
    return {
        "max_batch_size": max_batch_size,
        "moe_shared_output_order": compact,
        "npu_prefill_chunk_size": item_width,
        "batch": layout.batch,
        "num_entries": len(layout.entries()),
    }


def target_core_fields(num_devices: int) -> dict[str, Any]:
    """Round-robin placement over ``num_devices`` boards: shared layers on core 0 of each, experts everywhere."""
    if num_devices < 1:
        raise ValueError("--num-devices must be >= 1")
    return {
        "shared_target_device": "aries-rb",
        "shared_core_mode": "single",
        "shared_target_cores": [f"{dev}:0:0" for dev in range(num_devices)],
        "expert_target_device": "aries-rb",
        "expert_core_mode": "single",
        "expert_target_cores": [
            f"{dev}:{cluster}:{core}" for dev in range(num_devices) for cluster in range(2) for core in range(4)
        ],
        "lm_head_target_device": "aries-rb",
        "lm_head_core_mode": "single",
        "lm_head_target_cores": [f"{num_devices - 1}:0:0"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package", help="the compiled .mxqpkg")
    parser.add_argument("--config", required=True, help="release config.json to write (created from --base-config)")
    parser.add_argument("--base-config", help="upstream config.json to start a new release config from")
    parser.add_argument("--package-path", help="moe_mxq_package value (default: PACKAGE relative to the config)")
    parser.add_argument("--num-devices", type=int, help="also write round-robin target cores for this many boards")
    parser.add_argument("--num-layers", type=int, help="override config num_hidden_layers (debug packages)")
    args = parser.parse_args()

    if os.path.isfile(args.config):
        with open(args.config, encoding="utf-8") as file:
            config = json.load(file)
    elif args.base_config:
        with open(args.base_config, encoding="utf-8") as file:
            config = json.load(file)
        config.update(_MOBILINT_FIELDS)
    else:
        print(f"{args.config} does not exist; pass --base-config to create it", file=sys.stderr)
        return 1
    if args.num_layers is not None:
        config["num_hidden_layers"] = int(args.num_layers)
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config["num_experts"])

    try:
        derived = derive_package_fields(args.package, num_layers, num_experts)
    except (ValueError, KeyError) as exc:
        print(f"cannot configure the package: {exc}", file=sys.stderr)
        return 1

    config_dir = os.path.dirname(os.path.abspath(args.config))
    package_path = args.package_path or os.path.relpath(os.path.abspath(args.package), config_dir).replace(os.sep, "/")
    config["moe_mxq_package"] = package_path
    config.pop("moe_mxq_pack_index", None)
    config["max_batch_size"] = derived["max_batch_size"]
    config["moe_shared_output_order"] = derived["moe_shared_output_order"]
    if not config.get("npu_prefill_chunk_size") and derived["npu_prefill_chunk_size"]:
        config["npu_prefill_chunk_size"] = {"single": int(derived["npu_prefill_chunk_size"])}
    if args.num_devices is not None:
        config.update(target_core_fields(args.num_devices))
    config.setdefault("num_expert_workers", 24)

    with open(args.config, "w", encoding="utf-8", newline="\n") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
        file.write("\n")
    kind = "BatchLLM" if derived["batch"] else "single-sequence"
    order = json.dumps(derived["moe_shared_output_order"])
    print(
        f"[config] {args.config}: {kind} package {package_path} with {derived['num_entries']} entries, "
        f"max_batch_size={derived['max_batch_size']}, moe_shared_output_order={order}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
