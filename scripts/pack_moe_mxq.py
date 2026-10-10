"""Pack a Mixture-of-Experts release's per-layer shared/expert MXQs into one uncompressed ``.mxqpack`` per layer.

The Hugging Face Hub rate-limits API requests per repository file, so a release of thousands of small MXQs downloads
slowly or fails with ``429 Too Many Requests``. A packed release has one pack per layer plus an index JSON; the
loader (``MobilintMixtureOfExpertsNPUBackend``) materializes each member only between ``Model()`` and ``launch()``.

Pack ``l`` holds ``shared_template.format(layer=l)`` and then ``expert_template.format(layer=l, expert=e)`` for every
expert in order, each member starting on an ``--alignment`` boundary (zero padding, no compression, no header).
Every member is verified byte for byte against its source after writing unless ``--no-verify`` is given.

Usage:
    python scripts/pack_moe_mxq.py SRC_DIR OUT_DIR --num-layers 48 --num-experts 128
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time

from transformers_mblt.utils.moe_utils import MobilintMixtureOfExpertsPackIndex

_COPY_CHUNK = 16 * 1024 * 1024


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
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_sizes = []
    started = time.perf_counter()
    for layer_idx in range(args.num_layers):
        paths = _member_paths(args.src_dir, layer_idx, args.num_experts, args.shared_template, args.expert_template)
        missing = [path for path in paths if not os.path.isfile(path)]
        if missing:
            print(f"missing {len(missing)} MXQs for layer {layer_idx}, e.g. {missing[0]}", file=sys.stderr)
            return 1
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
        for layer_idx in range(args.num_layers):
            paths = _member_paths(args.src_dir, layer_idx, args.num_experts, args.shared_template, args.expert_template)
            pack_path = os.path.join(args.out_dir, args.pack_template.format(layer=layer_idx))
            with open(pack_path, "rb") as pack:
                for path, offset, size in zip(paths, index.offsets(layer_idx), index.sizes[layer_idx]):
                    if _sha256_range(pack, offset, size) != _sha256_file(path):
                        print(f"{pack_path}: member {path} does not match its source", file=sys.stderr)
                        return 1
        print(f"[verify] {sum(len(row) for row in all_sizes)} members match their sources")

    index_path = os.path.join(args.out_dir, args.index_name)
    with open(index_path, "w", encoding="utf-8", newline="\n") as file:
        file.write(index.to_json())
    if MobilintMixtureOfExpertsPackIndex.load(index_path) != index:
        print(f"{index_path} does not round-trip", file=sys.stderr)
        return 1
    print(f"[index] {index_path} ({os.path.getsize(index_path)} bytes); done in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
