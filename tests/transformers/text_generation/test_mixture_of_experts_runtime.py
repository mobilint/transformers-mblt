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

from transformers_mblt._npu import MobilintBackendAllocError
from transformers_mblt.models.qwen3_moe.configuration_qwen3_moe import MobilintQwen3MoEConfig
from transformers_mblt.utils import moe_utils
from transformers_mblt.utils.cache_utils import MobilintMixtureOfExpertsCache
from transformers_mblt.utils.generation_utils import MobilintMixtureOfExpertsGenerationMixin
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
    (tmp_path / "mxq").mkdir()
    (tmp_path / "mxq/lm_head.mxq").write_bytes(b"lm")
    for layer in range(NUM_LAYERS):
        (tmp_path / f"mxq/block_{layer}_shared.mxq").write_bytes(b"s")
        for expert in range(NUM_EXPERTS):
            (tmp_path / f"mxq/block_{layer}_expert_{expert}.mxq").write_bytes(b"e")
    return tmp_path


def _backend(release: Path) -> MobilintMixtureOfExpertsNPUBackend:
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
    )
    backend.create()
    return backend


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


class _Generator:
    _require_single_batch = staticmethod(MobilintMixtureOfExpertsGenerationMixin._require_single_batch)
    _prepare_cache_for_generation = MobilintMixtureOfExpertsGenerationMixin._prepare_cache_for_generation

    def __init__(self) -> None:
        self.npu_backend = type("Backend", (), {"kv_owner": None})()


def test_generate_resets_a_cache_whose_kv_was_overwritten_by_a_cacheless_forward() -> None:
    generator = _Generator()
    cache = MobilintMixtureOfExpertsCache([object()], owner=generator.npu_backend)
    cache.set_seq_length(12)
    generator.npu_backend.kv_owner = cache

    generator.npu_backend.kv_owner = None  # what moe_forward records for a forward without past_key_values
    model_kwargs = {"past_key_values": cache}
    generator._prepare_cache_for_generation(None, model_kwargs, None, 1, 0)

    assert cache.get_seq_length() == 0
    assert generator.npu_backend.kv_owner is cache


def test_restored_kv_is_reused_instead_of_reset() -> None:
    generator = _Generator()
    cache = MobilintMixtureOfExpertsCache([object()], owner=generator.npu_backend)
    generator.npu_backend.kv_owner = object()  # another sequence ran on the NPU

    cache.load_cache_memory()  # nothing dumped; restoring still claims the NPU KV
    cache.set_seq_length(7)
    generator._prepare_cache_for_generation(None, {"past_key_values": cache}, None, 1, 0)

    assert cache.get_seq_length() == 7
    assert cache.copy().owner is generator.npu_backend


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
