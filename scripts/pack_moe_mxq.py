"""Package a Mixture-of-Experts release: pack its MXQs and derive the shared MXQ output order.

The Hugging Face Hub rate-limits API requests per repository file, so a release of thousands of small MXQs downloads
slowly or fails with ``429 Too Many Requests``. A packed release has one pack per layer plus an index JSON; the
loader (``MobilintMixtureOfExpertsNPUBackend``) materializes each member only between ``Model()`` and ``launch()``.

Pack ``l`` holds ``shared_template.format(layer=l)`` and then ``expert_template.format(layer=l, expert=e)`` for every
expert in order, each member starting on an ``--alignment`` boundary (zero padding, no compression, no header).
Every member is verified byte for byte against its source after writing unless ``--no-verify`` is given.

``--config`` also writes ``moe_mxq_pack_index`` and ``moe_shared_output_order`` into a release ``config.json``. The
order (which shared output is the residual, the expert input and the router gate) is read from the MXQs themselves:
each shared and first-expert MXQ is created with ``qbruntime.Model`` (no accelerator, no ``launch``), the gate is the
output with ``num_experts`` channels, and the expert input is the output whose quantization scale is the reciprocal of
the expert input scale. ``--manifest`` cross-checks the residual against the compiler's ``manifest.json``.

Usage:
    python scripts/pack_moe_mxq.py SRC_DIR OUT_DIR --num-layers 48 --num-experts 128 \\
        --manifest SRC_DIR/manifest.json --config RELEASE/config.json
    python scripts/pack_moe_mxq.py SRC_DIR OUT_DIR --num-layers 48 --num-experts 128 --skip-pack --config config.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import sys
import time
from typing import Optional

from transformers_mblt.utils.moe_utils import MobilintMixtureOfExpertsPackIndex

_COPY_CHUNK = 16 * 1024 * 1024
_EXPERT_IN_SCALE_TOLERANCE = 1e-3
ROLES = ("residual", "expert_in", "gate")


def _member_paths(src_dir: str, layer_idx: int, num_experts: int, shared: str, expert: str) -> list[str]:
    return [os.path.join(src_dir, shared.format(layer=layer_idx))] + [
        os.path.join(src_dir, expert.format(layer=layer_idx, expert=expert_idx)) for expert_idx in range(num_experts)
    ]


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(_COPY_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_range(file, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    file.seek(offset)
    remaining = size
    while remaining:
        chunk = file.read(min(_COPY_CHUNK, remaining))
        if not chunk:
            raise EOFError(f"pack ends {remaining} bytes early")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def pack_layer(paths: list[str], pack_path: str, alignment: int) -> list[int]:
    """Write one layer pack and return the exact member sizes."""
    sizes = []
    tmp_path = pack_path + ".partial"
    with open(tmp_path, "wb") as out:
        for member_idx, path in enumerate(paths):
            with open(path, "rb") as src:
                shutil.copyfileobj(src, out, _COPY_CHUNK)
            size = os.path.getsize(path)
            sizes.append(size)
            if member_idx != len(paths) - 1:
                out.write(b"\0" * (-size % alignment))
    os.replace(tmp_path, pack_path)
    return sizes


def _manifest_residual_indices(manifest_path: str, num_layers: int) -> list[int]:
    """Return, per layer, the shared output the compiler's residual add consumes."""
    with open(manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)
    blocks = sorted(manifest["blocks"], key=lambda block: block["idx"])
    if len(blocks) != num_layers:
        raise ValueError(f"{manifest_path} lists {len(blocks)} blocks, expected {num_layers}.")
    residuals = []
    for block in blocks:
        outputs = block["partitions"]["shared"]["outputs"]
        consumed = [name for name in block["cpuOps"]["residual"]["inputs"] if name in outputs]
        if len(consumed) != 1:
            raise ValueError(f"{manifest_path}: block {block['idx']} residual add consumes {consumed} of {outputs}.")
        residuals.append(outputs.index(consumed[0]))
    return residuals


def derive_shared_output_order(
    shared_paths: list[str], expert_paths: list[str], num_experts: int, manifest: Optional[str] = None
) -> list[list[str]]:
    """Return the role of every shared MXQ output, per layer, read from the MXQs (``Model()`` only, no launch)."""
    import qbruntime
    from qbruntime import Model, ModelConfig

    qbruntime.set_log_level(qbruntime.LogLevel.Warn)
    residual_from_manifest = _manifest_residual_indices(manifest, len(shared_paths)) if manifest else None
    orders = []
    for layer_idx, (shared_path, expert_path) in enumerate(zip(shared_paths, expert_paths)):
        shared, expert = Model(shared_path, ModelConfig()), Model(expert_path, ModelConfig())
        try:
            shapes = shared.get_model_output_shape()
            out_scales = shared.get_output_scale()
            in_scale = expert.get_input_scale()[0]
        finally:
            shared.dispose()
            expert.dispose()
        gates = [idx for idx, shape in enumerate(shapes) if int(shape[-1]) == num_experts]
        if len(shapes) != 3 or len(gates) != 1:
            raise ValueError(f"layer {layer_idx}: cannot identify the router gate among output shapes {shapes}.")
        gate = gates[0]
        candidates = [
            idx
            for idx, scale in enumerate(out_scales)
            if idx != gate
            and scale.is_uniform
            and in_scale.is_uniform
            and abs(float(scale.scale) * float(in_scale.scale) - 1.0) <= _EXPERT_IN_SCALE_TOLERANCE
        ]
        if residual_from_manifest is not None:
            # The manifest names the residual; the scales must agree that the remaining output is the expert input.
            residual = residual_from_manifest[layer_idx]
            if residual == gate:
                raise ValueError(f"layer {layer_idx}: {manifest} puts the residual on the router gate output {gate}.")
            expert_in = ({0, 1, 2} - {gate, residual}).pop()
            if expert_in not in candidates:
                raise ValueError(
                    f"layer {layer_idx}: {manifest} puts the residual at output {residual}, but the MXQ scales do not "
                    f"make output {expert_in} the expert input (scale matches: {candidates})."
                )
        elif len(candidates) == 1:
            expert_in = candidates[0]
            residual = ({0, 1, 2} - {gate, expert_in}).pop()
        else:
            found = "no shared output scale" if not candidates else f"outputs {candidates} all"
            raise ValueError(
                f"layer {layer_idx}: {found} match the reciprocal of the expert input scale, so the expert input is "
                "ambiguous; pass --manifest to take the residual from the compiler manifest."
            )
        roles = [""] * 3
        roles[residual], roles[expert_in], roles[gate] = ROLES
        orders.append(roles)
    return orders


def compact_output_order(orders: list[list[str]]) -> dict[str, list[str]]:
    """Return the ``moe_shared_output_order`` mapping: the most common order as ``default`` plus per-layer overrides."""
    default = collections.Counter(tuple(order) for order in orders).most_common(1)[0][0]
    compact = {"default": list(default)}
    compact.update({str(layer_idx): order for layer_idx, order in enumerate(orders) if tuple(order) != default})
    return compact


def update_config(path: str, fields: dict) -> None:
    with open(path, encoding="utf-8") as file:
        config = json.load(file)
    config.update(fields)
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
        file.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src_dir", help="directory holding the individual shared/expert MXQs")
    parser.add_argument("out_dir", help="directory to write the packs and the index into")
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--shared-template", default="block_{layer}_shared.mxq")
    parser.add_argument("--expert-template", default="block_{layer}_expert_{expert}.mxq")
    parser.add_argument("--pack-template", default="block_{layer}.mxqpack")
    parser.add_argument("--index-name", default="index.json")
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument("--no-verify", action="store_true", help="skip the byte-for-byte check of every member")
    parser.add_argument("--skip-pack", action="store_true", help="only derive the output order (needs --config)")
    parser.add_argument("--manifest", help="compiler manifest.json to cross-check the derived output order against")
    parser.add_argument("--config", help="release config.json to write moe_mxq_pack_index / moe_shared_output_order")
    parser.add_argument(
        "--config-index-path",
        help="moe_mxq_pack_index value to write (default: OUT_DIR/INDEX_NAME relative to the config's directory)",
    )
    args = parser.parse_args()
    if args.skip_pack and not args.config:
        parser.error("--skip-pack only makes sense with --config")

    started = time.perf_counter()
    member_paths = [
        _member_paths(args.src_dir, layer_idx, args.num_experts, args.shared_template, args.expert_template)
        for layer_idx in range(args.num_layers)
    ]
    missing = [path for paths in member_paths for path in paths if not os.path.isfile(path)]
    if missing:
        print(f"missing {len(missing)} MXQs, e.g. {missing[0]}", file=sys.stderr)
        return 1

    index_path = os.path.join(args.out_dir, args.index_name)
    if not args.skip_pack:
        os.makedirs(args.out_dir, exist_ok=True)
        all_sizes = []
        for layer_idx, paths in enumerate(member_paths):
            pack_path = os.path.join(args.out_dir, args.pack_template.format(layer=layer_idx))
            all_sizes.append(pack_layer(paths, pack_path, args.alignment))
            print(f"[pack] layer {layer_idx}: {len(paths)} MXQs -> {pack_path} ({os.path.getsize(pack_path)} bytes)")

        index = MobilintMixtureOfExpertsPackIndex(
            pack=args.pack_template,
            alignment=args.alignment,
            sizes=tuple(tuple(row) for row in all_sizes),
        )
        for layer_idx in range(args.num_layers):
            pack_path = os.path.join(args.out_dir, args.pack_template.format(layer=layer_idx))
            if os.path.getsize(pack_path) != index.pack_size(layer_idx):
                print(f"{pack_path} size does not match its index", file=sys.stderr)
                return 1

        if not args.no_verify:
            for layer_idx, paths in enumerate(member_paths):
                pack_path = os.path.join(args.out_dir, args.pack_template.format(layer=layer_idx))
                with open(pack_path, "rb") as pack:
                    for path, offset, size in zip(paths, index.offsets(layer_idx), index.sizes[layer_idx]):
                        if _sha256_range(pack, offset, size) != _sha256_file(path):
                            print(f"{pack_path}: member {path} does not match its source", file=sys.stderr)
                            return 1
            print(f"[verify] {sum(len(row) for row in all_sizes)} members match their sources")

        with open(index_path, "w", encoding="utf-8", newline="\n") as file:
            file.write(index.to_json())
        if MobilintMixtureOfExpertsPackIndex.load(index_path) != index:
            print(f"{index_path} does not round-trip", file=sys.stderr)
            return 1
        print(f"[index] {index_path} ({os.path.getsize(index_path)} bytes)")

    if args.config:
        try:
            orders = derive_shared_output_order(
                [paths[0] for paths in member_paths],
                [paths[1] for paths in member_paths],
                args.num_experts,
                manifest=args.manifest,
            )
        except ValueError as exc:
            print(f"cannot derive moe_shared_output_order: {exc}", file=sys.stderr)
            return 1
        fields = {"moe_shared_output_order": compact_output_order(orders)}
        if not args.skip_pack:
            config_dir = os.path.dirname(os.path.abspath(args.config))
            default_index = os.path.relpath(os.path.abspath(index_path), config_dir).replace(os.sep, "/")
            fields["moe_mxq_pack_index"] = args.config_index_path or default_index
        update_config(args.config, fields)
        print(f"[config] {args.config}: {json.dumps(fields)}")

    print(f"[done] {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
