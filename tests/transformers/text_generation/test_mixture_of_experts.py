"""Unit tests for the Mixture-of-Experts config, cache, backend topology, and forward loop (no NPU)."""

from __future__ import annotations

from typing import Any, Iterator, Optional

import numpy as np
import pytest
import torch

from transformers_mblt.models.qwen3_moe.configuration_qwen3_moe import MobilintQwen3MoEConfig
from transformers_mblt.utils import moe_utils
from transformers_mblt.utils.cache_utils import MobilintMixtureOfExpertsCache
from transformers_mblt.utils.modeling_utils import MobilintMixtureOfExpertsModelMixin, MobilintModelMixin
from transformers_mblt.utils.moe_utils import MobilintMixtureOfExpertsNPUBackend

_ALL_CORES = [f"{dev}:{cluster}:{core}" for dev in range(3) for cluster in range(2) for core in range(4)]


def _small_config(**overrides: Any) -> MobilintQwen3MoEConfig:
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
        shared_mxq_path="mxq/block_{layer}_shared.mxq",
        expert_mxq_path="mxq/block_{layer}_expert_{expert}.mxq",
        lm_head_mxq_path="mxq/lm_head.mxq",
        shared_target_cores=["0:0:0", "1:0:0", "2:0:0"],
        expert_target_cores=_ALL_CORES,
        lm_head_target_cores=["2:0:0"],
        moe_shared_output_order={
            "default": ["residual", "expert_in", "gate"],
            "2": ["expert_in", "residual", "gate"],
        },
    )
    kwargs.update(overrides)
    return MobilintQwen3MoEConfig(**kwargs)


# ---- Config ---------------------------------------------------------------------------------------------------


def test_config_roundtrip_preserves_role_backends_and_output_order() -> None:
    config = _small_config(npu_prefill_chunk_size={"single": 64})

    restored = MobilintQwen3MoEConfig.from_dict(config.to_dict())

    assert restored.shared_target_cores == ["0:0:0", "1:0:0", "2:0:0"]
    assert restored.expert_target_cores == _ALL_CORES
    assert restored.lm_head_target_cores == ["2:0:0"]
    assert restored.expert_mxq_path == "mxq/block_{layer}_expert_{expert}.mxq"
    assert {restored.shared_core_mode, restored.expert_core_mode, restored.lm_head_core_mode} == {"single"}
    assert restored.get_moe_shared_output_indices(0) == (0, 1, 2)
    assert restored.get_moe_shared_output_indices(2) == (1, 0, 2)
    assert restored.npu_prefill_chunk_size == {"single": 64}
    assert restored.num_expert_workers == 24


def test_config_from_dict_applies_prefixed_overrides_to_one_role_only() -> None:
    config = _small_config()

    restored = MobilintQwen3MoEConfig.from_dict(config.to_dict(), lm_head_target_cores=["0:1:3"])

    assert restored.lm_head_target_cores == ["0:1:3"]
    assert restored.shared_target_cores == ["0:0:0", "1:0:0", "2:0:0"]


def test_config_defaults_place_every_role_on_device_zero() -> None:
    config = MobilintQwen3MoEConfig()

    assert config.shared_target_cores == ["0:0:0"]
    assert config.expert_target_cores == [f"0:{cluster}:{core}" for cluster in range(2) for core in range(4)]
    assert config.lm_head_target_cores == ["0:0:0"]
    assert config.moe_shared_output_order is None
    assert config.get_moe_shared_output_indices(0) is None


@pytest.mark.parametrize(
    "order",
    [
        {"default": ["residual", "gate"]},
        {"default": ["residual", "expert_in", "router"]},
        {"last": ["residual", "expert_in", "gate"]},
        ["residual", "expert_in", "gate"],
    ],
)
def test_config_rejects_malformed_shared_output_order(order: Any) -> None:
    with pytest.raises(ValueError):
        _small_config(moe_shared_output_order=order)


def test_config_rejects_batched_generation() -> None:
    config = _small_config()

    assert config.max_batch_size == 1
    with pytest.raises(ValueError, match="max_batch_size=1"):
        config.max_batch_size = 2


# ---- Backend topology -----------------------------------------------------------------------------------------


def _backend(config: MobilintQwen3MoEConfig, name_or_path: str = "") -> MobilintMixtureOfExpertsNPUBackend:
    return MobilintMixtureOfExpertsNPUBackend(
        shared_backend=config.shared_npu_backend,
        expert_backend=config.expert_npu_backend,
        lm_head_backend=config.lm_head_npu_backend,
        num_layers=config.num_hidden_layers,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        num_expert_workers=config.num_expert_workers,
        shared_output_indices=[config.get_moe_shared_output_indices(i) for i in range(config.num_hidden_layers)],
        name_or_path=name_or_path,
    )


def test_backend_places_roles_round_robin_over_their_devices() -> None:
    backend = _backend(_small_config())

    assert [backend.shared_device_of(layer) for layer in range(4)] == [0, 1, 2, 0]
    assert [backend.expert_device_of(expert) for expert in range(6)] == [0, 1, 2, 0, 1, 2]
    assert backend.lm_head_device == 2
    backend._validate_topology()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"lm_head_target_cores": ["0:0:0", "1:0:0"]}, "exactly one device"),
        ({"expert_core_mode": "multi", "expert_target_clusters": ["0:0"]}, "single"),
        ({"lm_head_target_device": "regulus-ra", "lm_head_target_cores": ["0:0:0"]}, "same board"),
    ],
)
def test_backend_rejects_unsupported_topologies(overrides: dict[str, Any], match: str) -> None:
    config = _small_config()
    for key, value in overrides.items():
        setattr(config, key, value)

    with pytest.raises(ValueError, match=match):
        _backend(config)._validate_topology()


def test_backend_create_resolves_local_release_and_reports_missing_files(tmp_path) -> None:
    config = _small_config()
    (tmp_path / "mxq").mkdir()
    (tmp_path / "mxq/lm_head.mxq").write_bytes(b"")
    for layer in range(config.num_hidden_layers):
        (tmp_path / f"mxq/block_{layer}_shared.mxq").write_bytes(b"")
        for expert in range(config.num_experts):
            (tmp_path / f"mxq/block_{layer}_expert_{expert}.mxq").write_bytes(b"")
    (tmp_path / "mxq/block_1_expert_3.mxq").unlink()

    backend = _backend(config, name_or_path=str(tmp_path))
    # Every expected file is checked, so a release with one hole (e.g. an interrupted download) is rejected.
    with pytest.raises(FileNotFoundError, match="block_1_expert_3"):
        backend.create()

    (tmp_path / "mxq/block_1_expert_3.mxq").write_bytes(b"")
    backend.create()
    assert backend.model_dir == str(tmp_path)
    assert backend.expert_mxq_path(1, 4) == "mxq/block_1_expert_4.mxq"


# ---- Cache ----------------------------------------------------------------------------------------------------


def test_cache_advances_every_shared_layer_in_lockstep() -> None:
    models = [object(), object(), object()]
    cache = MobilintMixtureOfExpertsCache(models)  # type: ignore[arg-type]

    cache.update_cache_position(torch.arange(5))
    assert [layer.get_seq_length() for layer in cache.shared_layers] == [5, 5, 5]

    # Callers written against MobilintCache rewind through layers[0]; the next update resyncs the rest.
    cache.layers[0]._seen_tokens = 2
    cache.update_cache_position(torch.arange(1))
    assert [layer.get_seq_length() for layer in cache.shared_layers] == [3, 3, 3]

    cache.crop(1)
    assert cache.get_seq_length() == 1
    cache.set_seq_length({0: 4})
    assert [layer.get_seq_length() for layer in cache.shared_layers] == [4, 4, 4]
    cache.fake_prefill(7)
    assert cache.get_seq_length() == 7
    with pytest.raises(ValueError):
        cache.set_seq_length({1: 4})
    assert cache.batch_size == 1


# ---- Forward loop ---------------------------------------------------------------------------------------------


class _FakeMoEBackend:
    """Deterministic stand-in for the NPU primitives of :class:`MobilintMixtureOfExpertsNPUBackend`.

    ``shared[l]`` returns ``residual = x``, ``gate = x @ gate_w[l]`` and ``expert_in = x``; expert ``e`` of layer
    ``l`` returns ``expert_in * (l + 1) * (e + 1) / 10``; the lm_head returns ``x @ lm_w``. A ``last_only`` final
    layer emits only the last chunk position.
    """

    core_mode = "single"

    def __init__(self, num_layers: int, hidden: int, num_experts: int, vocab: int, last_only: bool) -> None:
        rng = np.random.default_rng(0)
        self.num_layers = num_layers
        self.gate_w = rng.standard_normal((num_layers, hidden, num_experts)).astype(np.float32)
        self.lm_w = rng.standard_normal((hidden, vocab)).astype(np.float32)
        self.layers = [type("Layer", (), {"last_only": last_only and i == num_layers - 1})() for i in range(num_layers)]
        self.calls: list[tuple[str, int, int, int]] = []
        self._token_expert_in: Optional[np.ndarray] = None
        self.npu_clock = moe_utils._NpuClock()

    def _expert(self, layer_idx: int, expert_idx: int, x: np.ndarray) -> np.ndarray:
        return x * (layer_idx + 1) * (expert_idx + 1) / 10.0

    def shared_forward_token(self, layer_idx: int, x: np.ndarray, cache_size: int):
        self.calls.append(("token", layer_idx, cache_size, 1))
        self._token_expert_in = x.copy()
        return x.copy(), x @ self.gate_w[layer_idx]

    def experts_forward_token(self, layer_idx: int, experts: list[int]) -> Iterator[tuple[int, np.ndarray]]:
        assert self._token_expert_in is not None
        for expert_idx in reversed(experts):  # completion order differs from routing order
            yield expert_idx, self._expert(layer_idx, expert_idx, self._token_expert_in[0, 0])

    def shared_forward_chunk(self, layer_idx: int, x: np.ndarray, cache_size: int):
        self.calls.append(("chunk", layer_idx, cache_size, int(x.shape[1])))
        out = x[:, -1:, :] if self.layers[layer_idx].last_only else x
        return out.copy(), out @ self.gate_w[layer_idx], out.copy()

    def experts_forward_chunk(self, layer_idx: int, inputs: dict[int, np.ndarray]):
        for expert_idx, tokens in inputs.items():
            yield expert_idx, self._expert(layer_idx, expert_idx, tokens)

    def lm_head_forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.lm_w


def _harness(config: MobilintQwen3MoEConfig, backend: _FakeMoEBackend) -> Any:
    """Build an object carrying the MoE mixin methods without constructing an ``nn.Module``."""
    namespace: dict[str, Any] = {}
    for cls in reversed(MobilintMixtureOfExpertsModelMixin.__mro__):
        if not issubclass(cls, MobilintModelMixin):
            continue
        namespace.update({k: v for k, v in vars(cls).items() if not k.startswith("__")})
    harness = type("MoEHarness", (), namespace)()
    harness.config = config
    harness.npu_backend = backend
    harness.reset_npu_timing()
    return harness


def _reference_logits(config: MobilintQwen3MoEConfig, backend: _FakeMoEBackend, embeds: np.ndarray) -> np.ndarray:
    """Per-position logits computed independently of the chunk/token dispatch (torch routing)."""
    rows = []
    for position in range(embeds.shape[1]):
        x = torch.from_numpy(embeds[0, position].copy())
        for layer_idx in range(config.num_hidden_layers):
            probs = torch.softmax(x @ torch.from_numpy(backend.gate_w[layer_idx]), dim=-1)
            weights, experts = torch.topk(probs, config.num_experts_per_tok)
            weights = weights / weights.sum()
            out = x.clone()
            for weight, expert in zip(weights.tolist(), experts.tolist()):
                out += weight * torch.from_numpy(backend._expert(layer_idx, expert, x.numpy()))
            x = out
        rows.append((x @ torch.from_numpy(backend.lm_w)).numpy())
    return np.stack(rows)


@pytest.mark.parametrize("last_only", [False, True])
@pytest.mark.parametrize("logits_to_keep", [1, 0, 3, torch.tensor([4, 0, 4])])
def test_moe_forward_matches_reference_for_every_dispatch(last_only: bool, logits_to_keep: Any) -> None:
    config = _small_config(npu_prefill_chunk_size={"single": 2})
    backend = _FakeMoEBackend(
        config.num_hidden_layers, config.hidden_size, config.num_experts, config.vocab_size, last_only
    )
    model = _harness(config, backend)
    embeds = np.random.default_rng(1).standard_normal((1, 5, config.hidden_size)).astype(np.float32)
    cache = MobilintMixtureOfExpertsCache([object()] * config.num_hidden_layers)  # type: ignore[list-item]

    # Only a last-only release keeping several positions of one chunk falls back to the slow token path.
    slow_path = last_only and isinstance(logits_to_keep, int) and logits_to_keep in (0, 3)
    with pytest.warns(UserWarning) if slow_path else _no_warning():
        logits = model.moe_forward(
            torch.from_numpy(embeds), cache, torch.arange(5), logits_to_keep=logits_to_keep, count_npu_time=True
        )

    positions = model._output_positions_for_logits_to_keep(logits_to_keep, 5)
    expected = _reference_logits(config, backend, embeds)[positions]
    np.testing.assert_allclose(logits[0].numpy(), expected, rtol=1e-5, atol=1e-5)
    assert cache.get_seq_length() == 5
    assert model.get_npu_timing()["prefill_calls"] == 1
    # The KV cursor handed to every shared MXQ is contiguous across chunk and token calls.
    first_layer = [(kind, cache_size, n) for kind, layer, cache_size, n in backend.calls if layer == 0]
    assert sum(n for _, _, n in first_layer) == 5
    assert [cache_size for _, cache_size, _ in first_layer] == list(
        np.cumsum([0] + [n for _, _, n in first_layer])[:-1]
    )


def test_moe_forward_chunks_full_windows_and_runs_the_tail_token_by_token() -> None:
    config = _small_config(npu_prefill_chunk_size={"single": 2})
    backend = _FakeMoEBackend(
        config.num_hidden_layers, config.hidden_size, config.num_experts, config.vocab_size, False
    )
    model = _harness(config, backend)
    embeds = torch.zeros((1, 5, config.hidden_size))

    model.moe_forward(embeds, None, torch.arange(5), logits_to_keep=1)
    assert [(kind, cache_size, n) for kind, layer, cache_size, n in backend.calls if layer == 0] == [
        ("chunk", 0, 2),
        ("chunk", 2, 2),
        ("token", 4, 1),
    ]

    backend.calls.clear()
    model.moe_forward(embeds, None, torch.arange(5), logits_to_keep=1, npu_prefill_chunk_size=5)
    assert [(kind, n) for kind, layer, _, n in backend.calls if layer == 0] == [("chunk", 5)]


def test_moe_forward_decode_step_records_decode_timing() -> None:
    config = _small_config()
    backend = _FakeMoEBackend(config.num_hidden_layers, config.hidden_size, config.num_experts, config.vocab_size, True)
    model = _harness(config, backend)
    cache = MobilintMixtureOfExpertsCache([object()] * config.num_hidden_layers)  # type: ignore[list-item]
    cache.set_seq_length(9)

    logits = model.moe_forward(torch.zeros((1, 1, config.hidden_size)), cache, torch.tensor([9]), count_npu_time=True)

    assert logits.shape == (1, 1, config.vocab_size)
    assert [call[2] for call in backend.calls if call[1] == 0] == [9]
    assert cache.get_seq_length() == 10
    assert model.get_npu_timing()["decode_calls"] == 1
    assert model.npu_time is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_moe_forward_returns_logits_in_the_input_dtype(dtype: torch.dtype) -> None:
    config = _small_config()
    backend = _FakeMoEBackend(
        config.num_hidden_layers, config.hidden_size, config.num_experts, config.vocab_size, False
    )
    model = _harness(config, backend)
    embeds = torch.zeros((1, 3, config.hidden_size), dtype=dtype)

    assert model.moe_forward(embeds, None, torch.arange(3), logits_to_keep=0).dtype == dtype
    assert (
        model.moe_forward(embeds, None, torch.arange(3), logits_to_keep=torch.tensor([], dtype=torch.long)).dtype
        == dtype
    )


def test_moe_forward_rejects_batched_inputs() -> None:
    config = _small_config()
    backend = _FakeMoEBackend(
        config.num_hidden_layers, config.hidden_size, config.num_experts, config.vocab_size, False
    )
    model = _harness(config, backend)

    with pytest.raises(ValueError, match="batch_size=1"):
        model.moe_forward(torch.zeros((2, 1, config.hidden_size)), None, torch.arange(1))


class _no_warning:
    def __enter__(self) -> None:
        import warnings

        self._catcher = warnings.catch_warnings()
        self._catcher.__enter__()
        warnings.simplefilter("error", UserWarning)

    def __exit__(self, *exc: object) -> None:
        self._catcher.__exit__(*exc)
