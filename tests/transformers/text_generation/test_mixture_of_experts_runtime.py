"""Runtime robustness of the MoE backend without an NPU: rollback, buffer bounds, KV ownership, and pack packaging."""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import weakref
from pathlib import Path
from typing import Any

import pytest
import qbruntime
import torch

from transformers_mblt._npu import MobilintBackendAllocError
from transformers_mblt.models.qwen3_moe.configuration_qwen3_moe import MobilintQwen3MoEConfig
from transformers_mblt.utils import moe_utils
from transformers_mblt.utils.cache_residency import MobilintCacheStateError
from transformers_mblt.utils.cache_utils import MobilintMixtureOfExpertsCache
from transformers_mblt.utils.generation_utils import MobilintMixtureOfExpertsGenerationMixin
from transformers_mblt.utils.modeling_utils import MobilintMixtureOfExpertsModelMixin
from transformers_mblt.utils.moe_utils import (
    MobilintMixtureOfExpertsNPUBackend,
    MobilintMixtureOfExpertsPackIndex,
    _MxqPackReader,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
NUM_LAYERS, NUM_EXPERTS, HIDDEN = 2, 3, 4


# ---- Fakes ----------------------------------------------------------------------------------------------------


class _Scale:
    def __init__(self, scale: float) -> None:
        self.is_uniform = True
        self.scale = scale


class FakeModel:
    """Stand-in for ``qbruntime.Model``: shapes and scales follow the shipped Qwen3-MoE layout."""

    created: list["FakeModel"] = []
    fail_acquire_after: int | None = None
    acquired = 0

    def __init__(self, path: str, config: Any = None) -> None:
        self.path = path
        self.disposed = False
        self.launched = False
        self.is_shared = "shared" in os.path.basename(path)
        FakeModel.created.append(self)

    def launch(self, acc: Any) -> None:
        self.launched = True

    def dispose(self) -> None:
        self.disposed = True

    def get_model_output_shape(self):
        if self.is_shared:
            return [(1, -1, HIDDEN), (1, -1, HIDDEN), (1, -1, NUM_EXPERTS)]
        return [(1, -1, HIDDEN)]

    def get_output_scale(self):
        if self.is_shared:
            return [_Scale(0.5), _Scale(0.25), _Scale(0.1)]  # output 1 is the expert input (0.25 * 4 == 1)
        return [_Scale(1.0)]

    def get_input_scale(self):
        return [_Scale(4.0)]

    def _acquire(self, *args: Any) -> "_Buffer":
        FakeModel.acquired += 1
        if FakeModel.fail_acquire_after is not None and FakeModel.acquired > FakeModel.fail_acquire_after:
            raise qbruntime.QbRuntimeError("Model_BadAlloc occurred.")
        buffer = _Buffer()
        LIVE_BUFFERS.add(buffer)
        return buffer

    acquire_input_npu_data = _acquire
    acquire_output_npu_data = _acquire


class _Buffer:
    """Stand-in for ``NPUData``; weakly tracked so tests can see which buffers outlive their accelerator."""


LIVE_BUFFERS: "weakref.WeakSet[_Buffer]" = weakref.WeakSet()
# Live buffer count observed each time a FakeAccelerator is destroyed (qbruntime warns when it is non-zero).
BUFFERS_AT_ACCELERATOR_DESTRUCTION: list[int] = []


class FakeAccelerator:
    def __init__(self, *args: Any) -> None:
        pass

    def __del__(self) -> None:
        BUFFERS_AT_ACCELERATOR_DESTRUCTION.append(len(LIVE_BUFFERS))


@pytest.fixture()
def fake_runtime(monkeypatch: pytest.MonkeyPatch):
    FakeModel.created = []
    BUFFERS_AT_ACCELERATOR_DESTRUCTION.clear()
    FakeModel.fail_acquire_after = None
    FakeModel.acquired = 0
    monkeypatch.setattr(moe_utils, "Model", FakeModel)
    monkeypatch.setattr(moe_utils, "Accelerator", FakeAccelerator)
    monkeypatch.setattr(moe_utils, "ModelConfig", lambda: _FakeModelConfig())
    return FakeModel


class _FakeModelConfig:
    def set_single_core_mode(self, *args: Any) -> None:
        pass

    def set_activation_slots(self, *args: Any) -> None:
        pass


def _release_dir(tmp_path: Path) -> Path:
    (tmp_path / "mxq").mkdir(parents=True)
    (tmp_path / "mxq/lm_head.mxq").write_bytes(b"lm")
    for layer in range(NUM_LAYERS):
        (tmp_path / f"mxq/block_{layer}_shared.mxq").write_bytes(b"s")
        for expert in range(NUM_EXPERTS):
            (tmp_path / f"mxq/block_{layer}_expert_{expert}.mxq").write_bytes(b"e")
    return tmp_path


def _backend(release: Path, pack_index: str | None = None) -> MobilintMixtureOfExpertsNPUBackend:
    config = MobilintQwen3MoEConfig(
        vocab_size=7,
        hidden_size=HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        shared_mxq_path="mxq/block_{layer}_shared.mxq",
        expert_mxq_path="mxq/block_{layer}_expert_{expert}.mxq",
        lm_head_mxq_path="mxq/lm_head.mxq",
        shared_target_cores=["0:0:0", "1:0:0"],
        expert_target_cores=["0:0:0", "1:0:0"],
        lm_head_target_cores=["1:0:0"],
    )
    backend = MobilintMixtureOfExpertsNPUBackend(
        shared_backend=config.shared_npu_backend,
        expert_backend=config.expert_npu_backend,
        lm_head_backend=config.lm_head_npu_backend,
        num_layers=NUM_LAYERS,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        hidden_size=HIDDEN,
        vocab_size=7,
        num_expert_workers=2,
        shared_output_indices=[None] * NUM_LAYERS,
        name_or_path=str(release),
        mxq_pack_index=pack_index,
    )
    backend.create()
    return backend


def _packed_release_dir(tmp_path: Path) -> Path:
    """Pack the fake per-layer MXQs of :func:`_release_dir` into ``mxq/block_{layer}.mxqpack`` + ``mxq/index.json``."""
    script = _pack_script()
    src = _release_dir(tmp_path / "src")
    release = tmp_path / "release"
    (release / "mxq").mkdir(parents=True)
    (release / "mxq/lm_head.mxq").write_bytes(b"lm")
    sizes = []
    for layer in range(NUM_LAYERS):
        paths = [str(src / f"mxq/block_{layer}_shared.mxq")] + [
            str(src / f"mxq/block_{layer}_expert_{expert}.mxq") for expert in range(NUM_EXPERTS)
        ]
        sizes.append(tuple(script.pack_layer(paths, str(release / f"mxq/block_{layer}.mxqpack"), 4096)))
    index = MobilintMixtureOfExpertsPackIndex(pack="block_{layer}.mxqpack", alignment=4096, sizes=tuple(sizes))
    (release / "mxq/index.json").write_text(index.to_json())
    return release


# ---- Launch rollback ------------------------------------------------------------------------------------------


def test_launch_detects_output_order_and_disposes_everything(fake_runtime, tmp_path: Path) -> None:
    backend = _backend(_release_dir(tmp_path))
    backend.launch()

    assert [(layer.residual_idx, layer.expert_in_idx, layer.gate_idx) for layer in backend.layers] == [(0, 1, 2)] * 2
    assert len(fake_runtime.created) == NUM_LAYERS * (1 + NUM_EXPERTS) + 1
    backend.dispose()
    assert all(model.disposed for model in fake_runtime.created)
    assert backend.layers == [] and backend.accs == {}
    # Every buffer is released before its accelerator goes away (qbruntime warns otherwise).
    assert BUFFERS_AT_ACCELERATOR_DESTRUCTION == [0, 0]


def test_exit_hook_releases_buffers_when_dispose_is_never_called(
    fake_runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hooks: list = []
    monkeypatch.setattr(moe_utils.atexit, "register", hooks.append)
    monkeypatch.setattr(moe_utils.atexit, "unregister", hooks.remove)
    backend = _backend(_release_dir(tmp_path))
    backend.launch()
    assert len(hooks) == 1 and len(LIVE_BUFFERS) > 0

    hooks[0]()  # what the interpreter runs at exit, before tearing down modules
    assert len(LIVE_BUFFERS) == 0 and backend.layers == []
    backend.dispose()
    assert hooks == []  # dispose() unregisters the hook


def test_dropping_an_undisposed_backend_releases_buffers_before_accelerators(
    fake_runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(moe_utils.atexit, "register", lambda hook: None)
    backend = _backend(_release_dir(tmp_path))
    backend.launch()
    del backend  # what a server does when it unloads the model without dispose()
    gc.collect()
    assert BUFFERS_AT_ACCELERATOR_DESTRUCTION == [0, 0]


def test_buffer_bad_alloc_after_load_rolls_back_and_raises_alloc_error(fake_runtime, tmp_path: Path) -> None:
    backend = _backend(_release_dir(tmp_path))
    fake_runtime.fail_acquire_after = 5

    with pytest.raises(MobilintBackendAllocError):
        backend.launch()
    # Every loaded model is disposed even though the failure happened after all of them launched.
    assert fake_runtime.created and all(model.disposed for model in fake_runtime.created)
    assert backend.layers == [] and backend.lm_head is None and backend.accs == {}


def test_non_runtime_load_error_rolls_back(fake_runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(_release_dir(tmp_path))
    original = FakeModel.launch

    def launch(self: FakeModel, acc: Any) -> None:
        if "block_1_expert_2" in self.path:
            raise OSError("disk full")
        original(self, acc)

    monkeypatch.setattr(FakeModel, "launch", launch)
    with pytest.raises(OSError, match="disk full"):
        backend.launch()
    assert all(model.disposed for model in fake_runtime.created)


def test_failed_model_config_setup_releases_the_accelerators(
    fake_runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: "weakref.WeakSet[FakeAccelerator]" = weakref.WeakSet()

    class TrackedAccelerator(FakeAccelerator):
        def __init__(self, *args: Any) -> None:
            super().__init__(*args)
            created.add(self)

    monkeypatch.setattr(moe_utils, "Accelerator", TrackedAccelerator)
    backend = _backend(_release_dir(tmp_path))
    monkeypatch.setattr(backend, "_make_model_config", lambda role, dev: (_ for _ in ()).throw(ValueError("bad cores")))

    with pytest.raises(ValueError, match="bad cores"):
        backend.launch()
    gc.collect()
    assert backend.accs == {}
    assert len(created) == 0  # the failed launch keeps no accelerator alive


def test_ambiguous_expert_input_scales_are_rejected_without_a_configured_order(
    fake_runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Residual and expert input share the reciprocal scale, so the scales cannot tell them apart.
    monkeypatch.setattr(
        FakeModel,
        "get_output_scale",
        lambda self: [_Scale(0.25), _Scale(0.25), _Scale(0.1)] if self.is_shared else [_Scale(1.0)],
    )
    backend = _backend(_release_dir(tmp_path))

    with pytest.raises(ValueError, match="more than one"):
        backend.launch()
    assert all(model.disposed for model in fake_runtime.created)


# ---- Bounded chunk buffers ------------------------------------------------------------------------------------


def test_chunk_buffers_keep_only_recent_sizes(fake_runtime, tmp_path: Path) -> None:
    backend = _backend(_release_dir(tmp_path))
    backend.launch()

    first = backend._get_chunk_buffers(128)
    backend._get_chunk_buffers(64)
    assert backend._get_chunk_buffers(128) is first  # reused, and now the most recent
    backend._get_chunk_buffers(32)
    assert list(backend._chunk_buffers) == [128, 32]
    backend.dispose()


# ---- KV ownership ---------------------------------------------------------------------------------------------


class _SharedMxq:
    """A shared MXQ double whose single KV slot holds a string."""

    def __init__(self) -> None:
        self.kv = ""
        self.calls: list[str] = []

    def dump_cache_memory(self, cache_id: int = 0) -> list[bytes]:
        self.calls.append("dump")
        return [self.kv.encode()]

    def load_cache_memory(self, buffer: list[bytes], cache_id: int = 0) -> None:
        self.calls.append("load")
        self.kv = buffer[0].decode()


class _SlotBackend:
    """Backend double with real shared-MXQ slots; borrows kv_owner from the NPU backend."""

    kv_owner = moe_utils.MobilintMixtureOfExpertsNPUBackend.kv_owner

    def __init__(self, num_shared: int) -> None:
        self.layers = [type("Layer", (), {"shared": _SharedMxq(), "last_only": False})() for _ in range(num_shared)]


class _Generator:
    _require_single_batch = staticmethod(MobilintMixtureOfExpertsGenerationMixin._require_single_batch)
    _prepare_cache_for_generation = MobilintMixtureOfExpertsGenerationMixin._prepare_cache_for_generation
    _cache_matches_backend = MobilintMixtureOfExpertsGenerationMixin._cache_matches_backend
    _get_cache = MobilintMixtureOfExpertsGenerationMixin._get_cache
    _new_moe_cache = MobilintMixtureOfExpertsGenerationMixin._new_moe_cache
    _bind_moe_cache = MobilintMixtureOfExpertsModelMixin._bind_moe_cache
    _cache_layout_matches = MobilintMixtureOfExpertsGenerationMixin._cache_layout_matches

    def __init__(self, num_shared: int = 2) -> None:
        self.npu_backend = _SlotBackend(num_shared)

    def get_shared_mxq_models(self) -> list:
        return [layer.shared for layer in self.npu_backend.layers]

    def run(self, cache, text: str) -> None:
        """Stand-in for one forward: bind, write every shared slot from the cache position, advance."""
        self._bind_moe_cache(cache)
        start = 0 if cache is None else cache.get_seq_length()
        for model in self.get_shared_mxq_models():
            model.kv = model.kv[:start] + text
        if cache is not None:
            cache.update_seen_tokens(len(text))


def test_cache_displaced_by_a_cacheless_forward_resumes_instead_of_reset() -> None:
    generator = _Generator()
    cache = generator._new_moe_cache()
    generator.run(cache, "prompt")
    generator.run(None, "scratch")  # a forward without a cache saves the cache and overwrites the slots
    assert generator.npu_backend.kv_owner is None

    model_kwargs = {"past_key_values": cache}
    generator._prepare_cache_for_generation(None, model_kwargs, None, 1, 0)
    assert model_kwargs["past_key_values"] is cache and cache.get_seq_length() == 6

    generator.run(cache, "!")
    assert [model.kv for model in generator.get_shared_mxq_models()] == ["prompt!", "prompt!"]
    assert generator.npu_backend.kv_owner is cache


def test_two_caches_alternate_on_the_shared_mxqs() -> None:
    generator = _Generator()
    a, b = generator._new_moe_cache(), generator._new_moe_cache()
    generator.run(a, "aaa")
    generator.run(b, "bb")
    generator.run(a, "A")
    assert [model.kv for model in generator.get_shared_mxq_models()] == ["aaaA", "aaaA"]
    generator.run(b, "B")
    assert [model.kv for model in generator.get_shared_mxq_models()] == ["bbB", "bbB"]
    assert generator.npu_backend.kv_owner is b


def test_fake_prefilled_cache_is_never_dumped_or_loaded() -> None:
    generator = _Generator()
    other = generator._new_moe_cache()
    generator.run(other, "real")
    cache = generator._new_moe_cache()
    cache.fake_prefill(512)

    generator.run(cache, "t")

    assert cache.get_seq_length() == 513
    assert [layer.get_seq_length() for layer in cache.shared_layers] == [513, 513]
    assert all(model.calls == ["dump"] for model in generator.get_shared_mxq_models())  # only `other` was saved
    assert generator.npu_backend.kv_owner is cache


def test_cache_from_another_instance_moves_and_another_release_is_refused() -> None:
    source, target = _Generator(), _Generator()
    cache = source._new_moe_cache()
    source.run(cache, "hello")

    model_kwargs = {"past_key_values": cache}
    target._prepare_cache_for_generation(None, model_kwargs, None, 1, 0)
    assert model_kwargs["past_key_values"] is cache
    target.run(cache, "!")
    assert [model.kv for model in target.get_shared_mxq_models()] == ["hello!", "hello!"]
    assert source.npu_backend.kv_owner is None and target.npu_backend.kv_owner is cache

    other_release = _Generator(num_shared=3)
    with pytest.raises(MobilintCacheStateError):
        other_release._prepare_cache_for_generation(None, {"past_key_values": cache}, None, 1, 0)
    empty = source._new_moe_cache()
    model_kwargs = {"past_key_values": empty}
    other_release._prepare_cache_for_generation(None, model_kwargs, None, 1, 0)
    assert model_kwargs["past_key_values"] is not empty


def test_a_failed_forward_keeps_the_cache_on_its_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.transformers.text_generation.test_mixture_of_experts import _FakeMoEBackend, _harness, _small_config

    config = _small_config()
    backend = _FakeMoEBackend(
        config.num_hidden_layers, config.hidden_size, config.num_experts, config.vocab_size, False
    )
    for layer in backend.layers:
        layer.shared = _SharedMxq()
    shared_models = [layer.shared for layer in backend.layers]
    model = _harness(config, backend)
    cache = MobilintMixtureOfExpertsCache(shared_models)
    model.moe_forward(torch.zeros((1, 2, config.hidden_size)), cache, torch.arange(2))
    original = backend.shared_forward_token

    def fail_on_second_layer(layer_idx, *args):
        if layer_idx == 1:
            raise RuntimeError("NPU fault")
        return original(layer_idx, *args)

    monkeypatch.setattr(backend, "shared_forward_token", fail_on_second_layer)
    with pytest.raises(RuntimeError, match="NPU fault"):
        model.moe_forward(torch.zeros((1, 1, config.hidden_size)), cache, torch.arange(2, 3))

    # Writes start at the cursor, so the 2-token prefix is intact: the cache keeps its slots and its length.
    assert cache.get_seq_length() == 2
    assert cache.is_resident
    assert all(model.calls == [] for model in shared_models)


def test_benchmark_fake_prefill_uses_the_model_cache_builder() -> None:
    from transformers_mblt.utils import benchmark_utils

    built: list[int] = []
    sentinel = object()

    class _MoEModel:
        npu_backend = object()

        def get_cache_mxq_model(self) -> None:
            return None

        def build_mobilint_cache(self, batch_size: int) -> object:
            built.append(batch_size)
            return sentinel

    model = _MoEModel()
    assert benchmark_utils._supports_fake_decode_prefill(model)
    assert benchmark_utils._build_batched_mobilint_cache(model, 1) is sentinel
    assert built == [1]

    class _NoCacheModel:
        npu_backend = object()

        def get_cache_mxq_model(self) -> None:
            return None

    assert not benchmark_utils._supports_fake_decode_prefill(_NoCacheModel())


def test_npu_clock_counts_overlapping_calls_once(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([0.0, 1.0, 0.5, 2.0, 3.0, 3.5])  # [0, 1] and [0.5, 2] overlap (two devices), then [3, 3.5]
    monkeypatch.setattr(moe_utils.time, "perf_counter", lambda: next(ticks))
    clock = moe_utils._NpuClock()
    for _ in range(3):
        clock.call(lambda: None)

    assert clock.pop_seconds() == pytest.approx(2.5)
    assert clock.pop_seconds() == 0.0  # popping clears the recorded calls


def test_failed_packed_load_deletes_materialized_files_held_by_failed_models(
    fake_runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(moe_utils.MXQ_MATERIALIZE_ENV, "tempfile")
    monkeypatch.setenv(moe_utils.MXQ_TMPDIR_ENV, str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    backend = _backend(_packed_release_dir(tmp_path), pack_index="mxq/index.json")
    real_remove, real_launch = os.remove, FakeModel.launch

    def remove(path):  # Windows: a file still open in an undisposed Model cannot be deleted
        if any(model.path == path and not model.disposed for model in FakeModel.created):
            raise PermissionError("in use")
        real_remove(path)

    def launch(self, acc):
        if "block_1_expert_2" in self.path:
            raise OSError("launch failed")
        real_launch(self, acc)

    monkeypatch.setattr(moe_utils.os, "remove", remove)
    monkeypatch.setattr(FakeModel, "launch", launch)
    with pytest.raises(OSError, match="launch failed"):
        backend.launch()
    assert list((tmp_path / "tmp").iterdir()) == []


# ---- Pack reader fallbacks ------------------------------------------------------------------------------------


def _pack(tmp_path: Path) -> tuple[str, bytes]:
    payload = os.urandom(5000)
    index = MobilintMixtureOfExpertsPackIndex(pack="block_{layer}.mxqpack", alignment=4096, sizes=((len(payload),),))
    (tmp_path / "block_0.mxqpack").write_bytes(payload)
    (tmp_path / "index.json").write_text(index.to_json())
    return str(tmp_path / "index.json"), payload


def test_reader_falls_back_to_a_tempfile_when_proc_fd_paths_are_unusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path, payload = _pack(tmp_path)
    monkeypatch.setattr(moe_utils, "_memfd_path_usable", lambda: False)
    monkeypatch.setenv(moe_utils.MXQ_TMPDIR_ENV, str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()

    reader = _MxqPackReader(index_path, MobilintMixtureOfExpertsPackIndex.load(index_path))
    assert reader.mode == "tempfile" and reader.tempdir == str(tmp_path / "tmp")
    path, release = reader.materialize(0, 0, "block_0_shared.mxq")
    assert Path(path).parent == tmp_path / "tmp" and Path(path).read_bytes() == payload
    release()
    reader.close()
    assert list((tmp_path / "tmp").iterdir()) == []


def test_reader_copies_without_sendfile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_path, payload = _pack(tmp_path)
    monkeypatch.setenv(moe_utils.MXQ_MATERIALIZE_ENV, "tempfile")
    monkeypatch.setenv(moe_utils.MXQ_TMPDIR_ENV, str(tmp_path))
    reader = _MxqPackReader(index_path, MobilintMixtureOfExpertsPackIndex.load(index_path))
    reader._sendfile_ok = False
    path, release = reader.materialize(0, 0, "m.mxq")
    assert Path(path).read_bytes() == payload
    release()
    reader.close()


def test_reader_release_never_raises_and_retries_deletion_on_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path, _ = _pack(tmp_path)
    monkeypatch.setenv(moe_utils.MXQ_MATERIALIZE_ENV, "tempfile")
    monkeypatch.setenv(moe_utils.MXQ_TMPDIR_ENV, str(tmp_path))
    reader = _MxqPackReader(index_path, MobilintMixtureOfExpertsPackIndex.load(index_path))
    path, release = reader.materialize(0, 0, "held.mxq")

    real_remove = os.remove
    monkeypatch.setattr(os, "remove", lambda p: (_ for _ in ()).throw(PermissionError("in use")))
    release()  # a held file (Windows) must not raise into the loader thread
    assert os.path.exists(path)
    monkeypatch.setattr(os, "remove", real_remove)
    reader.close()
    assert not os.path.exists(path)


def test_reader_rejects_an_unknown_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_path, _ = _pack(tmp_path)
    monkeypatch.setenv(moe_utils.MXQ_MATERIALIZE_ENV, "ramdisk")
    with pytest.raises(ValueError, match="MBLT_MOE_MXQ_MATERIALIZE"):
        _MxqPackReader(index_path, MobilintMixtureOfExpertsPackIndex.load(index_path))


# ---- Output order packaging -----------------------------------------------------------------------------------


def _pack_script():
    spec = importlib.util.spec_from_file_location("pack_moe_mxq", REPO_ROOT / "scripts" / "pack_moe_mxq.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(tmp_path: Path, residual_last: int) -> str:
    blocks = []
    for layer in range(NUM_LAYERS):
        outputs = ["res", "ein", "gate"]
        if layer == NUM_LAYERS - 1 and residual_last != 0:
            outputs[0], outputs[residual_last] = outputs[residual_last], outputs[0]
        blocks.append(
            {
                "idx": layer,
                "partitions": {"shared": {"outputs": outputs}},
                "cpuOps": {"residual": {"inputs": ["res", "aggregate"]}},
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"blocks": blocks}))
    return str(path)


def test_output_order_is_derived_from_mxqs_and_cross_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _pack_script()
    monkeypatch.setattr(qbruntime, "Model", FakeModel)
    monkeypatch.setattr(qbruntime, "ModelConfig", _FakeModelConfig)
    release = _release_dir(tmp_path)
    shared = [str(release / f"mxq/block_{layer}_shared.mxq") for layer in range(NUM_LAYERS)]
    experts = [str(release / f"mxq/block_{layer}_expert_0.mxq") for layer in range(NUM_LAYERS)]

    orders = script.derive_shared_output_order(shared, experts, NUM_EXPERTS, manifest=_manifest(tmp_path, 0))
    assert orders == [["residual", "expert_in", "gate"]] * NUM_LAYERS
    assert script.compact_output_order(orders) == {"default": ["residual", "expert_in", "gate"]}
    assert script.compact_output_order([orders[0], ["expert_in", "residual", "gate"]]) == {
        "default": ["residual", "expert_in", "gate"],
        "1": ["expert_in", "residual", "gate"],
    }
    with pytest.raises(ValueError, match="manifest"):
        script.derive_shared_output_order(shared, experts, NUM_EXPERTS, manifest=_manifest(tmp_path, 1))
    assert all(model.disposed for model in FakeModel.created)


def test_ambiguous_scales_need_the_manifest_to_derive_the_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _pack_script()
    monkeypatch.setattr(qbruntime, "Model", FakeModel)
    monkeypatch.setattr(qbruntime, "ModelConfig", _FakeModelConfig)
    monkeypatch.setattr(
        FakeModel,
        "get_output_scale",
        lambda self: [_Scale(0.25), _Scale(0.25), _Scale(0.1)] if self.is_shared else [_Scale(1.0)],
    )
    release = _release_dir(tmp_path)
    shared = [str(release / f"mxq/block_{layer}_shared.mxq") for layer in range(NUM_LAYERS)]
    experts = [str(release / f"mxq/block_{layer}_expert_0.mxq") for layer in range(NUM_LAYERS)]

    with pytest.raises(ValueError, match="ambiguous"):
        script.derive_shared_output_order(shared, experts, NUM_EXPERTS)
    # The manifest names the residual, which settles the tie.
    orders = script.derive_shared_output_order(shared, experts, NUM_EXPERTS, manifest=_manifest(tmp_path, 1))
    assert orders[-1] == ["expert_in", "residual", "gate"]
    assert orders[0] == ["residual", "expert_in", "gate"]


def test_cache_from_a_release_with_another_kv_layout_is_refused_unless_empty() -> None:
    def with_layout(generator: _Generator, layer_hash: str) -> _Generator:
        for model in generator.get_shared_mxq_models():
            model.get_core_mode = lambda: "single"
            model.get_cache_infos = lambda h=layer_hash: [
                type("Info", (), {"name": "k", "size": 8, "num_batches": 1, "cache_type": "d", "layer_hash": h})()
            ]
            model.get_model_input_shape = lambda: [(1, -1, 4)]
            model.get_input_buffer_info = lambda: [type("Buf", (), {"max_cache_size": 64})()]
        return generator

    source = with_layout(_Generator(), "w8")
    cache = source._new_moe_cache()
    source.run(cache, "hello")

    same_release = with_layout(_Generator(), "w8")
    model_kwargs = {"past_key_values": cache}
    same_release._prepare_cache_for_generation(None, model_kwargs, None, 1, 0)
    assert model_kwargs["past_key_values"] is cache

    other_release = with_layout(_Generator(), "w4")
    with pytest.raises(MobilintCacheStateError, match="another KV layout"):
        other_release._prepare_cache_for_generation(None, {"past_key_values": cache}, None, 1, 0)

    empty = source._new_moe_cache()  # nothing to lose: replaced instead
    model_kwargs = {"past_key_values": empty}
    other_release._prepare_cache_for_generation(None, model_kwargs, None, 1, 0)
    assert model_kwargs["past_key_values"] is not empty


def _file_io(generator: _Generator) -> None:
    for model in generator.get_shared_mxq_models():
        model.dump_cache_memory_to = lambda cache_dir, cache_id=0, m=model: Path(cache_dir, "kv").write_text(m.kv)
        model.load_cache_memory_from = lambda cache_dir, cache_id=0, m=model: setattr(
            m, "kv", Path(cache_dir, "kv").read_text()
        )


def test_moe_cache_restored_from_files_before_any_forward_continues(tmp_path) -> None:
    generator = _Generator()
    _file_io(generator)
    saved = generator._new_moe_cache()
    generator.run(saved, "prompt")
    saved.dump_cache_memory_to(str(tmp_path))

    fresh = _Generator()  # a new process: new model, no owner table yet
    _file_io(fresh)
    restored = fresh._new_moe_cache()
    restored.load_cache_memory_from(str(tmp_path))

    fresh.run(restored, "!")

    assert [model.kv for model in fresh.get_shared_mxq_models()] == ["prompt!", "prompt!"]
    assert all(model.calls == [] for model in fresh.get_shared_mxq_models())


def test_moved_moe_cache_does_not_keep_its_source_backend_alive() -> None:
    source, target = _Generator(), _Generator()
    cache = MobilintMixtureOfExpertsCache(source.get_shared_mxq_models(), owner=source.npu_backend)
    source.run(cache, "hello")
    target.run(cache, "!")
    backend_ref = weakref.ref(source.npu_backend)

    del source
    gc.collect()

    assert backend_ref() is None
    assert cache.owner is None
    assert [model.kv for model in target.get_shared_mxq_models()] == ["hello!", "hello!"]
