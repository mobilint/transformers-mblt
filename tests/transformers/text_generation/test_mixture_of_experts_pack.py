"""Tests for the packed MoE MXQ format: index, pack script, and member materialization (no NPU)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from transformers_mblt.utils.moe_utils import MobilintMixtureOfExpertsPackIndex, _MxqPackReader

REPO_ROOT = Path(__file__).resolve().parents[3]
NUM_LAYERS, NUM_EXPERTS = 2, 3


def _write_release(src: Path) -> dict[tuple[int, int], bytes]:
    """Write fake per-layer MXQs of uneven sizes; return ``{(layer, member): bytes}``."""
    members = {}
    for layer in range(NUM_LAYERS):
        payloads = [os.urandom(5000 + layer)] + [os.urandom(4096 * expert + 7) for expert in range(NUM_EXPERTS)]
        (src / f"block_{layer}_shared.mxq").write_bytes(payloads[0])
        for expert in range(NUM_EXPERTS):
            (src / f"block_{layer}_expert_{expert}.mxq").write_bytes(payloads[1 + expert])
        members.update({(layer, idx): payload for idx, payload in enumerate(payloads)})
    return members


@pytest.fixture()
def packed_release(tmp_path: Path) -> tuple[Path, dict[tuple[int, int], bytes]]:
    src, out = tmp_path / "src", tmp_path / "mxq"
    src.mkdir()
    members = _write_release(src)
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "pack_moe_mxq.py"),
            str(src),
            str(out),
            "--num-layers",
            str(NUM_LAYERS),
            "--num-experts",
            str(NUM_EXPERTS),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))},
    )
    return out, members


def test_index_offsets_are_aligned_and_match_the_pack(packed_release) -> None:
    out, members = packed_release
    index = MobilintMixtureOfExpertsPackIndex.load(str(out / "index.json"))

    index.validate(NUM_LAYERS, NUM_EXPERTS)
    assert index.alignment == 4096
    for layer in range(NUM_LAYERS):
        assert all(offset % 4096 == 0 for offset in index.offsets(layer))
        assert index.sizes[layer] == tuple(len(members[(layer, idx)]) for idx in range(1 + NUM_EXPERTS))
        pack = (out / f"block_{layer}.mxqpack").read_bytes()
        assert len(pack) == index.pack_size(layer)
        for idx, offset in enumerate(index.offsets(layer)):
            assert pack[offset : offset + index.sizes[layer][idx]] == members[(layer, idx)]
    # One row per layer keeps the index readable.
    assert (out / "index.json").read_text().count("\n") == NUM_LAYERS + 8


@pytest.mark.parametrize("use_memfd", [True, False])
def test_reader_materializes_exact_members_and_releases_them(packed_release, use_memfd: bool) -> None:
    if use_memfd and not (sys.platform.startswith("linux") and hasattr(os, "memfd_create")):
        pytest.skip("memfd is Linux-only")
    out, members = packed_release
    index_path = str(out / "index.json")
    reader = _MxqPackReader(index_path, MobilintMixtureOfExpertsPackIndex.load(index_path))
    reader._use_memfd = use_memfd
    try:
        for (layer, idx), payload in members.items():
            path, release = reader.materialize(layer, idx, f"block_{layer}_{idx}.mxq")
            with open(path, "rb") as file:
                assert file.read() == payload
            release()
            assert not os.path.exists(path)
    finally:
        reader.close()
    # Nothing materialized is left next to the packs.
    assert sorted(p.name for p in out.iterdir()) == ["block_0.mxqpack", "block_1.mxqpack", "index.json"]


def test_index_rejects_wrong_shape_and_format(packed_release, tmp_path: Path) -> None:
    out, _ = packed_release
    index = MobilintMixtureOfExpertsPackIndex.load(str(out / "index.json"))
    with pytest.raises(ValueError, match="layers"):
        index.validate(NUM_LAYERS + 1, NUM_EXPERTS)
    with pytest.raises(ValueError, match="members"):
        index.validate(NUM_LAYERS, NUM_EXPERTS + 1)
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "other", "version": 1, "alignment": 4096, "pack": "x", "sizes": []}')
    with pytest.raises(ValueError, match="mobilint-moe-mxqpack"):
        MobilintMixtureOfExpertsPackIndex.load(str(bad))
