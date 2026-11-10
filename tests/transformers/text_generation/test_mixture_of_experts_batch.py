"""Unit tests for the BatchLLM MXQ package path of Mixture-of-Experts releases (no NPU).

Covers the slot-aware ``MobilintMixtureOfExpertsCache``, the batched forward (``_moe_forward_batch``) against a
per-sequence reference, the per-step token limits, the generation mixin's capacity checks, and package loading
through a fake ``qbruntime.ModelPackage``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import pytest
import torch

from transformers_mblt.models.qwen3_moe.configuration_qwen3_moe import MobilintQwen3MoEConfig
from transformers_mblt.utils import moe_utils
from transformers_mblt.utils.cache_utils import MobilintMixtureOfExpertsCache
from transformers_mblt.utils.generation_utils import MobilintMixtureOfExpertsGenerationMixin
from transformers_mblt.utils.modeling_utils import MobilintMixtureOfExpertsModelMixin, MobilintModelMixin
from transformers_mblt.utils.moe_utils import (
    MobilintMixtureOfExpertsNPUBackend,
    MobilintMixtureOfExpertsPackageLayout,
)

_ALL_CORES = [f"{dev}:{cluster}:{core}" for dev in range(2) for cluster in range(2) for core in range(4)]


def _config(**overrides: Any) -> MobilintQwen3MoEConfig:
    kwargs: dict[str, Any] = dict(
        vocab_size=7,
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=6,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        moe_mxq_package="mxq/release.mxqpkg",
        max_batch_size=4,
        shared_target_cores=["0:0:0", "1:0:0"],
        expert_target_cores=_ALL_CORES,
        lm_head_target_cores=["1:0:0"],
        npu_prefill_chunk_size={"single": 3},
    )
    kwargs.update(overrides)
    return MobilintQwen3MoEConfig(**kwargs)


# ---- Fake BatchLLM backend ------------------------------------------------------------------------------------


class _SharedMxq:
    """A shared MXQ handle double (the KV itself lives in :class:`_FakeBatchBackend`)."""


class _FakeBatchBackend:
    """Stand-in for the BatchLLM primitives of :class:`MobilintMixtureOfExpertsNPUBackend`.

    The "attention" of ``shared[l]`` is a causal prefix mean over each sequence's tokens (the KV slot holds the
    running sum and count of the sequence, so a later step continues the same mean): ``residual = x + mean``,
    ``gate = mean @ gate_w[l]``, ``expert_in = mean``. Expert ``e`` returns ``expert_in * (l + 1) * (e + 1) / 10``.
    The last layer is post-slice: it keeps each sequence's last token only. The lm_head is ``x @ lm_w``.
    """

    core_mode = "single"

    def __init__(self, num_layers: int, hidden: int, num_experts: int, vocab: int, k: int = 4) -> None:
        rng = np.random.default_rng(0)
        self.num_layers = num_layers
        self.hidden = hidden
        self.gate_w = rng.standard_normal((num_layers, hidden, num_experts)).astype(np.float32)
        self.lm_w = rng.standard_normal((hidden, vocab)).astype(np.float32)
        self.is_batch = True
        self.k_per_model = self.max_batch_size = k
        self.max_item_tokens = 3
        self.max_step_tokens = 6
        # Plain instances (not ``object()``): the residency owner table hangs its slot map on the model objects.
        self.shared_models = [_SharedMxq() for _ in range(num_layers)]
        self.layers = [
            type("Layer", (), {"shared": model, "last_only": i == num_layers - 1})()
            for i, model in enumerate(self.shared_models)
        ]
        # kv[(layer, slot)] = (sum, count) of the tokens the slot has seen, per cache_size prefix.
        self.kv: dict[tuple[int, int], list[np.ndarray]] = {}
        self.steps: list[list[tuple[int, int, int]]] = []  # per shared call: (sequence_length, cache_size, cache_id)
        self.npu_clock = moe_utils._NpuClock()

    def is_post_slice_layer(self, layer_idx: int) -> bool:
        return layer_idx == self.num_layers - 1

    def _expert(self, layer_idx: int, expert_idx: int, x: np.ndarray) -> np.ndarray:
        return x * (layer_idx + 1) * (expert_idx + 1) / 10.0

    def shared_forward_batch(self, layer_idx: int, hidden_states: np.ndarray, params: Sequence[Any]):
        if layer_idx == 0:
            self.steps.append([(p.sequence_length, p.cache_size, p.cache_id) for p in params])
        total = sum(p.sequence_length for p in params)
        assert hidden_states.shape == (total, self.hidden), (hidden_states.shape, total)
        assert total <= self.max_step_tokens and all(p.sequence_length <= self.max_item_tokens for p in params)
        means = []
        start = 0
        for p in params:
            tokens = hidden_states[start : start + p.sequence_length]
            start += p.sequence_length
            history = self.kv.setdefault((layer_idx, p.cache_id), [])
            del history[p.cache_size :]  # a write from cache_size discards whatever followed
            assert len(history) == p.cache_size, "KV slot must hold exactly cache_size tokens"
            for token in tokens:
                history.append(token.copy())
                means.append(np.mean(history, axis=0))
        mean = np.stack(means)
        residual = hidden_states + mean
        gate = mean @ self.gate_w[layer_idx]
        expert_in = mean.copy()
        if self.is_post_slice_layer(layer_idx):
            last = np.cumsum([p.sequence_length for p in params]) - 1
            residual, gate, expert_in = residual[last], gate[last], expert_in[last]
        return residual.astype(np.float32), gate.astype(np.float32), expert_in.astype(np.float32)

    def experts_forward_routed(self, layer_idx: int, inputs: dict[int, np.ndarray]) -> Iterator[tuple[int, np.ndarray]]:
        for expert_idx in reversed(list(inputs)):  # completion order differs from routing order
            yield expert_idx, self._expert(layer_idx, expert_idx, inputs[expert_idx])

    def lm_head_forward_rows(self, x: np.ndarray) -> np.ndarray:
        return x @ self.lm_w


def _harness(config: MobilintQwen3MoEConfig, backend: _FakeBatchBackend) -> Any:
    namespace: dict[str, Any] = {}
    for cls in reversed(MobilintMixtureOfExpertsModelMixin.__mro__):
        if not issubclass(cls, MobilintModelMixin):
            continue
        namespace.update({k: v for k, v in vars(cls).items() if not k.startswith("__")})
    harness = type("MoEBatchHarness", (), namespace)()
    harness.config = config
    harness.npu_backend = backend
    harness.reset_npu_timing()
    return harness


def _reference_last_logits(
    config: MobilintQwen3MoEConfig, backend: _FakeBatchBackend, tokens: np.ndarray
) -> np.ndarray:
    """Last-token logits of one sequence ``[seq, hidden]`` computed with torch, independently of the batching."""
    x = torch.from_numpy(tokens.astype(np.float32))
    for layer_idx in range(config.num_hidden_layers):
        mean = torch.cumsum(x, dim=0) / torch.arange(1, x.shape[0] + 1).unsqueeze(1)
        residual = x + mean
        gate = mean @ torch.from_numpy(backend.gate_w[layer_idx])
        out = residual.clone()
        for position in range(x.shape[0]):
            probs = torch.softmax(gate[position], dim=-1)
            weights, experts = torch.topk(probs, config.num_experts_per_tok)
            weights = weights / weights.sum()
            for weight, expert in zip(weights.tolist(), experts.tolist()):
                out[position] += weight * torch.from_numpy(backend._expert(layer_idx, expert, mean[position].numpy()))
        x = out
    return (x[-1] @ torch.from_numpy(backend.lm_w)).numpy()


@pytest.mark.parametrize("lengths", [[5], [5, 2, 4], [1, 1], [7, 3, 6, 2]])
def test_batched_forward_matches_per_sequence_reference(lengths: list[int]) -> None:
    config = _config()
    backend = _FakeBatchBackend(config.num_hidden_layers, config.hidden_size, config.num_experts, config.vocab_size)
    model = _harness(config, backend)
    rng = np.random.default_rng(1)
    sequences = [rng.standard_normal((length, config.hidden_size)).astype(np.float32) for length in lengths]
    width = max(lengths)
    # Left-padded batch, as generate pads it.
    embeds = torch.zeros((len(lengths), width, config.hidden_size))
    mask = torch.zeros((len(lengths), width), dtype=torch.long)
    for row, tokens in enumerate(sequences):
        embeds[row, width - len(tokens) :] = torch.from_numpy(tokens)
        mask[row, width - len(tokens) :] = 1
    cache = MobilintMixtureOfExpertsCache(backend.shared_models, batch_size=4)  # type: ignore[arg-type]

    logits = model.moe_forward(embeds, cache, torch.arange(width), attention_mask=mask, count_npu_time=True)

    assert logits.shape == (len(lengths), 1, config.vocab_size)
    for row, tokens in enumerate(sequences):
        np.testing.assert_allclose(
            logits[row, 0].numpy(), _reference_last_logits(config, backend, tokens), rtol=1e-4, atol=1e-4
        )
    assert [cache.get_seq_length(row) for row in range(len(lengths))] == lengths
    assert model.get_npu_timing()["prefill_calls"] == 1
    # Every step honours both compiled widths: <= 3 tokens per sequence and <= 6 tokens over all sequences.
    for step in backend.steps:
        assert sum(n for n, _, _ in step) <= backend.max_step_tokens
        assert all(n <= backend.max_item_tokens for n, _, _ in step)
    # Rows run on the KV slots of their cache rows and continue from their own lengths.
    assert {cache_id for step in backend.steps for _, _, cache_id in step} == set(range(len(lengths)))


def test_batched_decode_continues_each_row_from_its_own_length() -> None:
    config = _config()
    backend = _FakeBatchBackend(config.num_hidden_layers, config.hidden_size, config.num_experts, config.vocab_size)
    model = _harness(config, backend)
    rng = np.random.default_rng(2)
    sequences = [rng.standard_normal((n, config.hidden_size)).astype(np.float32) for n in (4, 2)]
    cache = MobilintMixtureOfExpertsCache(backend.shared_models, batch_size=4)  # type: ignore[arg-type]
    prompts = torch.zeros((2, 4, config.hidden_size))
    mask = torch.zeros((2, 4), dtype=torch.long)
    for row, tokens in enumerate(sequences):
        prompts[row, 4 - len(tokens) :] = torch.from_numpy(tokens)
        mask[row, 4 - len(tokens) :] = 1
    model.moe_forward(prompts, cache, torch.arange(4), attention_mask=mask)

    # generate() passes the mask of the whole sequence with one new token per row.
    new_tokens = rng.standard_normal((2, 1, config.hidden_size)).astype(np.float32)
    decode_mask = torch.cat([mask, torch.ones((2, 1), dtype=torch.long)], dim=1)
    logits = model.moe_forward(
        torch.from_numpy(new_tokens), cache, torch.tensor([4]), attention_mask=decode_mask, count_npu_time=True
    )

    for row, tokens in enumerate(sequences):
        full = np.concatenate([tokens, new_tokens[row]])
        np.testing.assert_allclose(
            logits[row, 0].numpy(), _reference_last_logits(config, backend, full), rtol=1e-4, atol=1e-4
        )
    assert [cache.get_seq_length(row) for row in range(2)] == [5, 3]
    assert backend.steps[-1] == [(1, 4, 0), (1, 2, 1)]
    assert model.get_npu_timing()["decode_calls"] == 1


def test_batched_forward_uses_the_cache_rows_slots() -> None:
    config = _config()
    backend = _FakeBatchBackend(config.num_hidden_layers, config.hidden_size, config.num_experts, config.vocab_size)
    model = _harness(config, backend)
    models = backend.shared_models
    # Two request caches on slots 3 and 1, joined for one step (a serving engine's batch).
    first = MobilintMixtureOfExpertsCache(models, batch_size=4, slots=[3])  # type: ignore[list-item]
    second = MobilintMixtureOfExpertsCache(models, batch_size=4, slots=[1])  # type: ignore[list-item]
    joined = MobilintMixtureOfExpertsCache.from_rows([first, second])
    assert joined.joined_rows and joined.batch_size == 2 and [joined.slot_of(r) for r in range(2)] == [(0, 3), (0, 1)]

    embeds = torch.from_numpy(np.random.default_rng(3).standard_normal((2, 2, config.hidden_size)).astype(np.float32))
    model.moe_forward(embeds, joined, torch.arange(2))

    assert backend.steps == [[(2, 0, 3), (2, 0, 1)]]
    assert (first.get_seq_length(), second.get_seq_length()) == (2, 2)  # the request caches advanced by their rows


def test_batched_forward_rejects_too_many_rows_and_non_last_logits() -> None:
    config = _config()
    backend = _FakeBatchBackend(config.num_hidden_layers, config.hidden_size, config.num_experts, config.vocab_size)
    model = _harness(config, backend)
    with pytest.raises(ValueError, match="at most 4 sequences"):
        model.moe_forward(torch.zeros((5, 1, config.hidden_size)), None, torch.arange(1))
    with pytest.raises(NotImplementedError, match="last-token logits"):
        model.moe_forward(torch.zeros((1, 3, config.hidden_size)), None, torch.arange(3), logits_to_keep=0)
    with pytest.raises(ValueError, match="empty sequences"):
        model.moe_forward(
            torch.zeros((2, 2, config.hidden_size)),
            None,
            torch.arange(2),
            attention_mask=torch.tensor([[1, 1], [0, 0]]),
        )


# ---- Cache ----------------------------------------------------------------------------------------------------


def test_cache_rows_track_their_own_lengths_and_slots() -> None:
    models = [object(), object()]
    cache = MobilintMixtureOfExpertsCache(models, batch_size=4)  # type: ignore[list-item]
    assert cache.batch_size == cache.max_batch_size == cache.k_per_model == 4 and cache.n_models == 1
    assert [cache.slot_of(row) for row in range(4)] == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert cache.group_by_model([2, 0]) == {0: [(2, 2), (0, 0)]}

    cache.update_seen_tokens({0: 5, 2: 3})
    cache.update_cache_position(torch.arange(2), index=1)
    assert [cache.get_seq_length(row) for row in range(4)] == [5, 2, 3, 0]
    assert [layer.get_seq_length() for layer in cache.rows[0]] == [5, 5]
    cache.crop(4)
    assert [cache.get_seq_length(row) for row in range(4)] == [4, 2, 3, 0]
    cache.fake_prefill(7)
    assert [cache.get_seq_length(row) for row in range(4)] == [7, 7, 7, 7]
    cache.fake_prefill({1: 2})
    assert cache.get_seq_length(1) == 2
    with pytest.raises(ValueError):
        cache.set_seq_length({4: 1})

    copied = cache.copy()
    assert [copied.slot_of(row) for row in range(4)] == [cache.slot_of(row) for row in range(4)]
    assert [copied.get_seq_length(row) for row in range(4)] == [7, 2, 7, 7]
    cache.reset()
    assert [cache.get_seq_length(row) for row in range(4)] == [0] * 4

    single = MobilintMixtureOfExpertsCache(models)  # type: ignore[list-item]
    assert single.batch_size == single.k_per_model == 1 and single.shared_layers is single.rows[0]
    with pytest.raises(ValueError):
        MobilintMixtureOfExpertsCache(models, batch_size=2, slots=[1, 1])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        MobilintMixtureOfExpertsCache(models, batch_size=2, slots=[2])  # type: ignore[list-item]


def test_cache_from_rows_requires_one_row_caches_of_one_release() -> None:
    models = [object(), object()]
    a = MobilintMixtureOfExpertsCache(models, batch_size=4, slots=[0])  # type: ignore[list-item]
    b = MobilintMixtureOfExpertsCache(models, batch_size=4, slots=[0])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="distinct slots"):
        MobilintMixtureOfExpertsCache.from_rows([a, b])
    other = MobilintMixtureOfExpertsCache([object(), object()], batch_size=4, slots=[1])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="one release"):
        MobilintMixtureOfExpertsCache.from_rows([a, other])
    wide = MobilintMixtureOfExpertsCache(models, batch_size=4)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="one-row"):
        MobilintMixtureOfExpertsCache.from_rows([wide])


# ---- Generation mixin -----------------------------------------------------------------------------------------


class _Generator:
    _moe_capacity = MobilintMixtureOfExpertsGenerationMixin._moe_capacity
    _require_batch_size = MobilintMixtureOfExpertsGenerationMixin._require_batch_size
    _new_moe_cache = MobilintMixtureOfExpertsGenerationMixin._new_moe_cache
    _cache_matches_backend = MobilintMixtureOfExpertsGenerationMixin._cache_matches_backend
    _get_cache = MobilintMixtureOfExpertsGenerationMixin._get_cache
    build_mobilint_cache = MobilintMixtureOfExpertsGenerationMixin.build_mobilint_cache

    def __init__(self, capacity: int) -> None:
        self.npu_backend = type("Backend", (), {"max_batch_size": capacity, "k_per_model": capacity})()
        self.models = [object(), object()]

    def get_shared_mxq_models(self) -> list:
        return self.models


def test_generation_mixin_sizes_caches_to_the_compiled_capacity() -> None:
    generator = _Generator(4)
    cache = generator._get_cache("mobilint", 3, 0)
    assert cache.batch_size == 4 and cache.k_per_model == 4  # row j on slot j for any batch up to the capacity
    assert generator._get_cache("mobilint", 4, 0) is cache
    with pytest.raises(ValueError, match="at most 4 sequences"):
        generator._get_cache("mobilint", 5, 0)
    request = generator.build_mobilint_cache(1, slots=[2])
    assert request.batch_size == 1 and request.slot_of(0) == (0, 2) and request.k_per_model == 4
    assert generator.build_mobilint_cache(2).batch_size == 2

    single = _Generator(1)
    with pytest.raises(ValueError, match="one sequence at a time"):
        single._get_cache("mobilint", 2, 0)
    assert single._get_cache("mobilint", 1, 0).k_per_model == 1


# ---- Package loading --------------------------------------------------------------------------------------------


def _manifest(num_layers: int, num_experts: int, *, batch: bool) -> dict[str, Any]:
    blocks = []
    for layer in range(num_layers):
        outputs = [f"add_{layer}", f"norm_{layer}", f"gate_{layer}"]
        if layer % 2:
            outputs = [outputs[1], outputs[0], outputs[2]]  # the residual is not always output 0
        blocks.append(
            {
                "idx": layer,
                "type": "moe",
                "cpuOps": {"residual": {"inputs": [f"add_{layer}", "experts_out"], "output": "next"}},
                "partitions": {
                    "shared": {"file": f"block_{layer}_shared.mxq", "outputs": outputs},
                    "experts": [{"idx": e, "file": f"block_{layer}_expert_{e}.mxq"} for e in range(num_experts)],
                },
            }
        )
    return {"schemaVersion": 1, "batch": batch, "blocks": blocks, "lmHead": {"file": "lm_head.mxq"}}


def test_package_layout_reads_entries_and_residual_roles_from_the_manifest() -> None:
    layout = MobilintMixtureOfExpertsPackageLayout.from_manifest(_manifest(2, 3, batch=True), 2, 3)
    assert layout.shared == ("block_0_shared.mxq", "block_1_shared.mxq")
    assert layout.experts[1] == ("block_1_expert_0.mxq", "block_1_expert_1.mxq", "block_1_expert_2.mxq")
    assert layout.lm_head == "lm_head.mxq" and layout.batch
    assert layout.residual_output == (0, 1)
    assert len(layout.entries()) == 2 * 4 + 1
    with pytest.raises(ValueError, match="blocks"):
        MobilintMixtureOfExpertsPackageLayout.from_manifest(_manifest(2, 3, batch=True), 3, 3)
    with pytest.raises(ValueError, match="experts"):
        MobilintMixtureOfExpertsPackageLayout.from_manifest(_manifest(2, 3, batch=True), 2, 4)


class _Info:
    def __init__(self, num_batches: int) -> None:
        self.num_batches = num_batches
        self.size = 16


class _BufferInfo:
    def __init__(self, max_width: int) -> None:
        self.max_width = max_width


class _Scale:
    def __init__(self, scale: float) -> None:
        self.is_uniform = True
        self.scale = scale


class _PackagedModel:
    launched: list[tuple[str, Optional[int]]] = []

    def __init__(self, entry: str, slots: Optional[int]) -> None:
        self.entry = entry
        self.slots = slots
        self.disposed = False
        self.is_shared = "shared" in entry

    def launch(self, acc: Any) -> None:
        _PackagedModel.launched.append((self.entry, self.slots))

    def dispose(self) -> None:
        self.disposed = True

    def get_model_output_shape(self):
        if self.is_shared:
            return (
                [(1, -1, 4), (1, -1, 4), (1, -1, 3)]
                if self.entry.startswith("block_0")
                else [(1, -1, 4), (1, -1, 4), (1, -1, 3)]
            )
        return [(1, -1, 4)]

    def get_output_scale(self):
        if self.entry == "block_0_shared.mxq":
            return [_Scale(0.5), _Scale(0.25), _Scale(0.1)]  # output 1 is the expert input
        if self.is_shared:
            return [_Scale(0.25), _Scale(0.5), _Scale(0.1)]  # output 0 is the expert input (block 1 order)
        return [_Scale(1.0)]

    def get_input_scale(self):
        return [_Scale(4.0)]

    def get_cache_infos(self):
        return [_Info(4), _Info(4)] if self.is_shared else []

    def get_input_buffer_info(self):
        return [_BufferInfo(128 if self.is_shared else 512)]


class _FakeConfig:
    def __init__(self) -> None:
        self.slots: Optional[int] = None

    def set_single_core_mode(self, *args: Any) -> None:
        pass

    def set_activation_slots(self, slots: int) -> None:
        self.slots = slots


class _FakePackage:
    opened: list[str] = []

    def __init__(self, path: str) -> None:
        self.path = path
        _FakePackage.opened.append(path)
        self.manifest = _manifest(2, 3, batch=True)

    def get_path(self) -> str:
        return self.path

    def get_entry_names(self) -> list[str]:
        layout = MobilintMixtureOfExpertsPackageLayout.from_manifest(self.manifest, 2, 3)
        return ["manifest.json", *layout.entries()]

    def load_entry(self, name: str) -> bytes:
        assert name == "manifest.json"
        return json.dumps(self.manifest).encode()

    def create_model(self, entry: str, config: Any = None) -> _PackagedModel:
        return _PackagedModel(entry, None if config is None else config.slots)


def test_backend_loads_a_batch_package_and_probes_its_capacity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "mxq").mkdir()
    (tmp_path / "mxq/release.mxqpkg").write_bytes(b"pkg")
    _PackagedModel.launched = []
    _FakePackage.opened = []
    monkeypatch.setattr(moe_utils.qbruntime, "ModelPackage", _FakePackage, raising=False)
    monkeypatch.setattr(moe_utils, "Accelerator", lambda *args: object())
    monkeypatch.setattr(moe_utils, "ModelConfig", _FakeConfig)
    config = _config(num_hidden_layers=2, num_experts=3, max_batch_size=16, moe_shared_output_order=None)
    backend = MobilintMixtureOfExpertsNPUBackend(
        shared_backend=config.shared_npu_backend,
        expert_backend=config.expert_npu_backend,
        lm_head_backend=config.lm_head_npu_backend,
        num_layers=2,
        num_experts=3,
        num_experts_per_tok=2,
        hidden_size=4,
        vocab_size=7,
        num_expert_workers=2,
        shared_output_indices=[None, None],
        name_or_path=str(tmp_path),
        mxq_package="mxq/release.mxqpkg",
        max_batch_size=16,
    )
    backend.create()
    assert [os.path.normpath(path) for path in _FakePackage.opened] == [str(tmp_path / "mxq" / "release.mxqpkg")]
    assert backend.is_batch
    assert backend.shared_mxq_path(1) == "block_1_shared.mxq" and backend.lm_head_mxq_path() == "lm_head.mxq"

    backend.launch()
    # The compiled capacity wins over config.max_batch_size, and the per-step widths come from the first models.
    assert (backend.k_per_model, backend.max_batch_size) == (4, 4)
    assert (backend.max_item_tokens, backend.max_step_tokens) == (128, 512)
    slots = dict(_PackagedModel.launched)
    assert slots["block_0_shared.mxq"] == 1 and slots["lm_head.mxq"] == 1
    assert slots["block_0_expert_0.mxq"] == 0 and slots["block_1_expert_0.mxq"] == 1  # post-slice experts need a slot
    assert [(layer.residual_idx, layer.expert_in_idx, layer.gate_idx) for layer in backend.layers] == [
        (0, 1, 2),
        (1, 0, 2),
    ]
    assert [layer.last_only for layer in backend.layers] == [False, True]
    assert len(_PackagedModel.launched) == 2 * 4 + 1
    backend.dispose()


def test_backend_rejects_a_manifest_that_contradicts_the_configured_residual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "mxq").mkdir()
    (tmp_path / "mxq/release.mxqpkg").write_bytes(b"pkg")
    monkeypatch.setattr(moe_utils.qbruntime, "ModelPackage", _FakePackage, raising=False)
    config = _config(
        num_hidden_layers=2, num_experts=3, moe_shared_output_order={"default": ["gate", "residual", "expert_in"]}
    )
    backend = MobilintMixtureOfExpertsNPUBackend(
        shared_backend=config.shared_npu_backend,
        expert_backend=config.expert_npu_backend,
        lm_head_backend=config.lm_head_npu_backend,
        num_layers=2,
        num_experts=3,
        num_experts_per_tok=2,
        hidden_size=4,
        vocab_size=7,
        num_expert_workers=2,
        shared_output_indices=[config.get_moe_shared_output_indices(i) for i in range(2)],
        name_or_path=str(tmp_path),
        mxq_package="mxq/release.mxqpkg",
    )
    with pytest.raises(ValueError, match="residual add consumes output 0"):
        backend.create()


def test_backend_refuses_both_pack_index_and_package() -> None:
    config = _config()
    with pytest.raises(ValueError, match="either packed"):
        MobilintMixtureOfExpertsNPUBackend(
            shared_backend=config.shared_npu_backend,
            expert_backend=config.expert_npu_backend,
            lm_head_backend=config.lm_head_npu_backend,
            num_layers=3,
            num_experts=6,
            num_experts_per_tok=2,
            hidden_size=4,
            vocab_size=7,
            num_expert_workers=2,
            shared_output_indices=[None] * 3,
            mxq_pack_index="mxq/index.json",
            mxq_package="mxq/release.mxqpkg",
        )
