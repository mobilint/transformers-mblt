"""Portable caches: the backend's owner table moves KV between caches, slots and model instances."""

from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest

from transformers_mblt.utils.cache_residency import MobilintCacheStateError, residency_of
from transformers_mblt.utils.cache_utils import MobilintCache


class _SlotMxq:
    """MXQ double whose KV slots hold strings, so tests can see which sequence a slot holds."""

    def __init__(self, k: int = 1, signature: str = "llama-w8") -> None:
        self.slots: dict[int, object] = {cache_id: None for cache_id in range(k)}
        self.calls: list[tuple[str, int]] = []
        self.signature = signature

    def write(self, cache_id: int, content: str) -> None:
        self.slots[cache_id] = content

    def dump_cache_memory(self, cache_id: int = 0) -> list[bytes]:
        self.calls.append(("dump", cache_id))
        return [str(self.slots[cache_id]).encode()]

    def load_cache_memory(self, buffer: list[bytes], cache_id: int = 0) -> None:
        self.calls.append(("load", cache_id))
        self.slots[cache_id] = buffer[0].decode()

    # mxq_kv_signature inputs
    def get_core_mode(self) -> str:
        return "single"

    def get_cache_infos(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(name="k", size=8, num_batches=len(self.slots), cache_type="d", layer_hash=self.signature)
        ]

    def get_model_input_shape(self) -> list[tuple[int, ...]]:
        return [(1, -1, 4)]

    def get_input_buffer_info(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(max_cache_size=128)]


class _Backend:
    def __init__(self, n: int = 1, k: int = 1, signature: str = "llama-w8") -> None:
        self.mxq_models = [_SlotMxq(k, signature) for _ in range(n)]
        self.k_per_model = k


def _run(cache: MobilintCache, backend: _Backend, row: int, text: str) -> None:
    """Stand-in for one infer on ``row``: bind, write KV, advance the cursor."""
    cache.bind_rows(backend, [row])
    model_idx, cache_id = cache.slot_of(row)
    mxq = backend.mxq_models[model_idx]
    # Like the NPU: the infer reads the first ``seq_length`` positions and writes after them.
    current = str(mxq.slots[cache_id])[: cache.get_seq_length(row)] if cache.get_seq_length(row) > 0 else ""
    mxq.write(cache_id, f"{current}{text}")
    cache.update_seen_tokens({row: len(text)})


def _slot(backend: _Backend, model_idx: int = 0, cache_id: int = 0) -> object:
    return backend.mxq_models[model_idx].slots[cache_id]


def test_same_cache_keeps_decoding_without_dump_or_load() -> None:
    backend = _Backend()
    cache = MobilintCache(backend.mxq_models)

    _run(cache, backend, 0, "abc")
    _run(cache, backend, 0, "d")

    assert _slot(backend) == "abcd"
    assert backend.mxq_models[0].calls == []
    assert cache.is_resident


def test_switching_caches_saves_the_owner_and_restores_the_other() -> None:
    backend = _Backend()
    a = MobilintCache(backend.mxq_models)
    b = MobilintCache(backend.mxq_models)

    _run(a, backend, 0, "aaa")
    _run(b, backend, 0, "bb")  # A is saved; B starts from scratch, nothing to load
    assert backend.mxq_models[0].calls == [("dump", 0)]
    assert not a.is_resident and b.is_resident

    _run(a, backend, 0, "A")  # B is saved, A is loaded back and continues
    assert _slot(backend) == "aaaA"
    assert backend.mxq_models[0].calls == [("dump", 0), ("dump", 0), ("load", 0)]

    _run(b, backend, 0, "B")
    assert _slot(backend) == "bbB"


def test_cacheless_forward_saves_the_owner_and_leaves_the_slot_unowned() -> None:
    from transformers_mblt.utils.cache_residency import bind_cacheless

    backend = _Backend()
    a = MobilintCache(backend.mxq_models)
    _run(a, backend, 0, "aaa")

    bind_cacheless(backend, [(backend.mxq_models[0], 0)])
    backend.mxq_models[0].write(0, "scratch")

    assert residency_of(backend).owner(backend.mxq_models[0], 0) is None
    _run(a, backend, 0, "A")
    assert _slot(backend) == "aaaA"


def test_empty_cache_never_loads_and_reset_cache_reuses_the_slot() -> None:
    backend = _Backend()
    a = MobilintCache(backend.mxq_models)
    _run(a, backend, 0, "aaa")
    a.reset()

    _run(a, backend, 0, "x")

    assert _slot(backend) == "x"
    assert backend.mxq_models[0].calls == []


def test_fake_prefill_cache_is_never_dumped_or_loaded() -> None:
    backend = _Backend()
    a = MobilintCache(backend.mxq_models)
    b = MobilintCache(backend.mxq_models)
    a.fake_prefill(64)

    _run(a, backend, 0, "t")
    _run(b, backend, 0, "b")
    _run(a, backend, 0, "t")

    assert [call for call in backend.mxq_models[0].calls if call[0] == "load"] == []
    assert ("dump", 0) in backend.mxq_models[0].calls  # B was real KV and was saved
    assert a.get_seq_length() == 66


def test_garbage_collected_owner_frees_the_slot_without_dump() -> None:
    backend = _Backend()
    a = MobilintCache(backend.mxq_models)
    _run(a, backend, 0, "aaa")
    del a
    gc.collect()

    b = MobilintCache(backend.mxq_models)
    _run(b, backend, 0, "b")

    assert backend.mxq_models[0].calls == []


def test_copy_is_a_snapshot_that_restores_on_first_use() -> None:
    backend = _Backend()
    a = MobilintCache(backend.mxq_models)
    _run(a, backend, 0, "prefix")

    snapshot = a.copy()
    assert a.is_resident and not snapshot.is_resident

    _run(a, backend, 0, "-a")
    _run(snapshot, backend, 0, "-s")

    assert _slot(backend) == "prefix-s"
    _run(a, backend, 0, "!")
    assert _slot(backend) == "prefix-a!"


def test_offload_frees_the_slot_ahead_of_time() -> None:
    backend = _Backend()
    a = MobilintCache(backend.mxq_models)
    _run(a, backend, 0, "aaa")

    a.offload()
    backend.mxq_models[0].calls.clear()
    b = MobilintCache(backend.mxq_models)
    _run(b, backend, 0, "b")

    assert backend.mxq_models[0].calls == []
    assert a.host_nbytes == len(b"aaa")


def test_explicit_dump_and_load_keep_working_alongside_binding() -> None:
    backend = _Backend()
    a = MobilintCache(backend.mxq_models)
    b = MobilintCache(backend.mxq_models)
    _run(a, backend, 0, "aaa")
    a.dump_cache_memory()
    _run(a, backend, 0, "zz")

    _run(b, backend, 0, "bb")
    a.load_cache_memory()  # restore the explicit snapshot: B is saved first, A's 3-token prefix comes back

    assert a.get_seq_length() == 3
    assert _slot(backend) == "aaa"
    _run(b, backend, 0, "B")
    assert _slot(backend) == "bbB"


def test_rows_on_chosen_slots_bind_independently() -> None:
    backend = _Backend(n=2, k=2)
    first = MobilintCache(backend.mxq_models, per_model_batch=2, slots=[(1, 1)])
    second = MobilintCache(backend.mxq_models, per_model_batch=2, slots=[(0, 0)])

    _run(first, backend, 0, "one")
    _run(second, backend, 0, "two")

    assert _slot(backend, 1, 1) == "one"
    assert _slot(backend, 0, 0) == "two"
    assert first.is_resident and second.is_resident
    assert all(not mxq.calls for mxq in backend.mxq_models)


def test_slots_are_validated() -> None:
    backend = _Backend(n=1, k=2)
    with pytest.raises(ValueError, match="distinct"):
        MobilintCache(backend.mxq_models, per_model_batch=2, slots=[(0, 1), (0, 1)])
    with pytest.raises(ValueError, match="outside"):
        MobilintCache(backend.mxq_models, per_model_batch=2, slots=[(0, 2)])


def test_cache_moves_between_model_instances() -> None:
    instance_a = _Backend()
    instance_b = _Backend()
    cache = MobilintCache(instance_a.mxq_models)
    _run(cache, instance_a, 0, "hello")

    _run(cache, instance_b, 0, "!")

    assert instance_a.mxq_models[0].calls == [("dump", 0)]
    assert instance_b.mxq_models[0].calls == [("load", 0)]
    assert _slot(instance_b) == "hello!"
    assert residency_of(instance_a).owner(instance_a.mxq_models[0], 0) is None
    assert cache.mxq_models == instance_b.mxq_models

    _run(cache, instance_a, 0, "?")  # and back
    assert _slot(instance_a) == "hello!?"


def test_cache_refuses_to_move_to_an_incompatible_mxq() -> None:
    w8 = _Backend(signature="llama-w8")
    w4 = _Backend(signature="llama-w4")
    cache = MobilintCache(w8.mxq_models)
    _run(cache, w8, 0, "hello")

    with pytest.raises(MobilintCacheStateError, match="different KV layout"):
        _run(cache, w4, 0, "!")


def test_cache_refuses_to_move_to_a_different_topology() -> None:
    single = _Backend(n=1, k=1)
    batch = _Backend(n=1, k=4)
    cache = MobilintCache(single.mxq_models)
    _run(cache, single, 0, "hello")

    with pytest.raises(ValueError, match="different slot topology"):
        _run(cache, batch, 0, "!")


def test_disposed_backend_drops_unsaved_kv_and_says_so() -> None:
    backend = _Backend()
    cache = MobilintCache(backend.mxq_models)
    _run(cache, backend, 0, "hello")

    backend.mxq_models = [_SlotMxq()]  # dispose + create: new handles, old KV memory gone

    with pytest.raises(MobilintCacheStateError, match="dump_cache_memory"):
        _run(cache, backend, 0, "!")


def test_disposed_backend_restores_kv_saved_before_dispose() -> None:
    backend = _Backend()
    cache = MobilintCache(backend.mxq_models)
    _run(cache, backend, 0, "hello")
    cache.dump_cache_memory()

    backend.mxq_models = [_SlotMxq()]
    _run(cache, backend, 0, "!")

    assert _slot(backend) == "hello!"


def test_rewound_cache_restores_only_what_its_host_copy_still_matches() -> None:
    backend = _Backend()
    a = MobilintCache(backend.mxq_models)
    b = MobilintCache(backend.mxq_models)
    _run(a, backend, 0, "abcdef")
    a.dump_cache_memory()
    a.set_seq_length(3)
    _run(a, backend, 0, "XY")  # writes positions 3.. while resident

    _run(b, backend, 0, "b")  # evicts A: the dump captures the rewritten tail
    _run(a, backend, 0, "!")

    assert _slot(backend)[: a.get_seq_length()] == "abcXY!"


# ----------------------------------------------------------------------
# Through the model: llm_forward (single row) and MultiSlotDispatcher (batched rows)
# ----------------------------------------------------------------------

import numpy as np  # noqa: E402
import torch  # noqa: E402

from transformers_mblt.utils.modeling_utils import MobilintModelMixin  # noqa: E402
from transformers_mblt.utils.multi_slot_dispatch import MultiSlotDispatcher  # noqa: E402


class _TokenKvMxq(_SlotMxq):
    """Slot KV is the list of token ids written so far; logits encode the visible prefix.

    An infer reads the first ``cache_size`` ids of its slot, so running on another sequence's KV changes the
    logits. That is the silent corruption the owner table prevents.
    """

    vocab = 3

    def __init__(self, k: int = 1) -> None:
        super().__init__(k)
        self.slots = {cache_id: [] for cache_id in range(k)}

    def dump_cache_memory(self, cache_id: int = 0) -> list[bytes]:
        self.calls.append(("dump", cache_id))
        return [bytes(self.slots[cache_id])]

    def load_cache_memory(self, buffer: list[bytes], cache_id: int = 0) -> None:
        self.calls.append(("load", cache_id))
        self.slots[cache_id] = list(buffer[0])

    def get_model_output_shape(self):
        return [(1, 1, self.vocab)]

    def _step(self, cache_id: int, cache_size: int, tokens: list[int]) -> np.ndarray:
        visible = self.slots[cache_id][:cache_size] + tokens
        self.slots[cache_id] = visible
        return np.array([len(visible), sum(visible), sum(i * t for i, t in enumerate(visible))], dtype=np.float32)

    def infer(self, inputs, _outputs, cache_size, batch_params=None):
        chunk = np.asarray(inputs[0]).reshape(-1, np.asarray(inputs[0]).shape[-1])
        tokens = [int(row[0]) for row in chunk]
        if batch_params is None:
            return [self._step(0, int(cache_size), tokens)[None, None, :]]
        rows, offset = [], 0
        for param in batch_params:
            n = int(param.sequence_length)
            rows.append(self._step(int(param.cache_id), int(param.cache_size), tokens[offset : offset + n]))
            offset += n
        return [np.stack(rows)]


class _ModelBackend:
    def __init__(self, n: int = 1, k: int = 1) -> None:
        self.mxq_models = [_TokenKvMxq(k) for _ in range(n)]
        self.k_per_model = k
        self.output_layout = "n_items"
        self._dispatcher = None

    @property
    def mxq_model(self):
        return self.mxq_models[0]

    @property
    def dispatcher(self):
        if self._dispatcher is None:
            self._dispatcher = MultiSlotDispatcher(self)
        return self._dispatcher


def _model(backend: _ModelBackend) -> MobilintModelMixin:
    model = MobilintModelMixin.__new__(MobilintModelMixin)
    model.npu_backend = backend
    model.config = type("Config", (), {"npu_prefill_chunk_size": None})()
    model.npu_time = None
    return model


def _forward(model: MobilintModelMixin, cache, tokens: list[int]) -> list[float]:
    start = 0 if cache is None else cache.get_seq_length()
    embeds = torch.tensor(tokens, dtype=torch.float32).reshape(1, len(tokens), 1)
    logits = model.llm_forward(embeds, cache, torch.arange(start, start + len(tokens)), npu_prefill_chunk_size=64)
    return logits.reshape(-1).tolist()


def test_llm_forward_resumes_each_cache_on_its_own_kv() -> None:
    reference = _model(_ModelBackend())
    ref_cache = MobilintCache(reference.npu_backend.mxq_models)
    _forward(reference, ref_cache, [1, 2, 3])
    expected = _forward(reference, ref_cache, [4])

    model = _model(_ModelBackend())
    a = MobilintCache(model.npu_backend.mxq_models)
    b = MobilintCache(model.npu_backend.mxq_models)
    _forward(model, a, [1, 2, 3])
    _forward(model, b, [7, 8])
    _forward(model, None, [9, 9, 9, 9])  # a forward without a cache also overwrites the slot

    assert _forward(model, a, [4]) == expected
    assert _forward(model, b, [5]) == [3.0, 20.0, 18.0]


def test_llm_forward_moves_a_cache_between_model_instances() -> None:
    first = _model(_ModelBackend())
    second = _model(_ModelBackend())
    cache = MobilintCache(first.npu_backend.mxq_models)
    _forward(first, cache, [1, 2, 3])

    logits = _forward(second, cache, [4])

    assert logits == [4.0, 10.0, 20.0]
    assert first.npu_backend.mxq_models[0].calls == [("dump", 0)]
    assert second.npu_backend.mxq_models[0].calls == [("load", 0)]


def test_llm_forward_rejects_a_cache_placed_off_slot_zero() -> None:
    model = _model(_ModelBackend(k=2))
    cache = MobilintCache(model.npu_backend.mxq_models, per_model_batch=2, slots=[(0, 1)])

    with pytest.raises(ValueError, match=r"slot \(0, 0\)"):
        _forward(model, cache, [1])


def _dispatch(backend: _ModelBackend, cache, rows: list[int], tokens: list[list[int]]) -> np.ndarray:
    sizes = [cache.get_seq_length(row) for row in rows] if cache is not None else [0] * len(rows)
    chunks = [torch.tensor(t, dtype=torch.float32).reshape(len(t), 1) for t in tokens]
    merged, _ = backend.dispatcher.dispatch(
        rows,
        [len(t) for t in tokens],
        sizes,
        chunks,
        max_sequence_length=max(len(t) for t in tokens),
        past_key_values=cache,
    )
    if cache is not None:
        cache.update_seen_tokens({row: len(t) for row, t in zip(rows, tokens)})
    return merged


@pytest.mark.parametrize("n,k", [(2, 1), (1, 2)])
def test_dispatch_switches_batched_caches_per_row(n: int, k: int) -> None:
    backend = _ModelBackend(n=n, k=k)
    a = MobilintCache(backend.mxq_models, per_model_batch=k)
    b = MobilintCache(backend.mxq_models, per_model_batch=k)

    _dispatch(backend, a, [0, 1], [[1, 2], [3]])
    _dispatch(backend, b, [0, 1], [[7], [8, 8]])
    resumed = _dispatch(backend, a, [0, 1], [[4], [5]])

    assert resumed[:, 0].tolist() == [3.0, 2.0]  # each row still sees only its own prefix
    assert resumed[:, 1].tolist() == [7.0, 8.0]


def test_dispatch_places_single_row_caches_on_their_slots() -> None:
    backend = _ModelBackend(n=1, k=4)
    request_a = MobilintCache(backend.mxq_models, per_model_batch=4, slots=[(0, 2)])
    request_b = MobilintCache(backend.mxq_models, per_model_batch=4, slots=[(0, 3)])

    _dispatch(backend, request_a, [0], [[1, 1]])
    _dispatch(backend, request_b, [0], [[2]])

    assert backend.mxq_models[0].slots[2] == [1, 1]
    assert backend.mxq_models[0].slots[3] == [2]
    assert backend.mxq_models[0].calls == []


def test_file_dump_of_an_evicted_cache_writes_its_own_kv(tmp_path) -> None:
    backend = _Backend()
    written: dict[str, object] = {}
    mxq = backend.mxq_models[0]
    mxq.dump_cache_memory_to = lambda cache_dir, cache_id=0: written.setdefault(cache_dir, mxq.slots[cache_id])
    a = MobilintCache(backend.mxq_models)
    b = MobilintCache(backend.mxq_models)
    _run(a, backend, 0, "aaa")
    _run(b, backend, 0, "bb")

    a.dump_cache_memory_to(str(tmp_path))

    assert written[str(tmp_path)] == "aaa"
    _run(b, backend, 0, "B")  # B was saved before A came back, so it resumes correctly
    assert _slot(backend) == "bbB"


def test_moving_off_a_disposed_backend_reports_lost_kv_without_touching_dead_models() -> None:
    source = _Backend()
    target = _Backend()
    cache = MobilintCache(source.mxq_models)
    _run(cache, source, 0, "hello")
    dead = source.mxq_models[0]
    dead.dump_cache_memory = lambda cache_id=0: (_ for _ in ()).throw(AssertionError("dumped a disposed Model"))
    source.mxq_models = []  # dispose

    with pytest.raises(MobilintCacheStateError, match="dump_cache_memory"):
        _run(cache, target, 0, "!")


def test_moving_off_a_disposed_backend_uses_kv_saved_before_dispose() -> None:
    source = _Backend()
    target = _Backend()
    cache = MobilintCache(source.mxq_models)
    _run(cache, source, 0, "hello")
    cache.dump_cache_memory()
    source.mxq_models = []

    _run(cache, target, 0, "!")

    assert _slot(target) == "hello!"


def test_cache_on_chosen_slots_refuses_to_grow() -> None:
    backend = _Backend(n=1, k=4)
    cache = MobilintCache(backend.mxq_models, per_model_batch=4, slots=[(0, 2)])

    with pytest.raises(ValueError, match="chosen slots"):
        cache.ensure_batch_size(2)


def test_move_is_refused_when_the_target_kv_layout_cannot_be_read() -> None:
    source = _Backend()
    target = _Backend()
    target.mxq_models[0].get_cache_infos = lambda: (_ for _ in ()).throw(RuntimeError("query failed"))
    cache = MobilintCache(source.mxq_models)
    _run(cache, source, 0, "hello")

    with pytest.raises(MobilintCacheStateError, match="KV layout"):
        _run(cache, target, 0, "!")


def test_snapshot_of_an_evicted_fake_prefill_cache_restores_length_only() -> None:
    backend = _Backend()
    fake = MobilintCache(backend.mxq_models)
    other = MobilintCache(backend.mxq_models)
    fake.fake_prefill(64)
    _run(fake, backend, 0, "t")
    _run(other, backend, 0, "o")  # evicts the fake-prefill cache: nothing is saved

    fake.dump_cache_memory()
    fake.set_seq_length(0)
    fake.load_cache_memory()

    assert fake.get_seq_length() == 65
    assert not fake.layers[0].kv_valid
    assert [call for call in backend.mxq_models[0].calls if call[0] == "load"] == []


def test_batched_request_larger_than_the_chosen_slots_is_rejected_up_front() -> None:
    cache = MobilintCache(_Backend(n=1, k=4).mxq_models, per_model_batch=4, slots=[(0, 1), (0, 2)])

    MobilintModelMixin._validate_batch_cache(cache, 2)
    with pytest.raises(ValueError, match="cache rows=2, input batch_size=3"):
        MobilintModelMixin._validate_batch_cache(cache, 3)


def test_snapshots_after_dispose_never_touch_the_dead_model() -> None:
    backend = _Backend()
    saved = MobilintCache(backend.mxq_models)
    lost = MobilintCache(backend.mxq_models, per_model_batch=1)
    _run(saved, backend, 0, "kept")
    saved.dump_cache_memory()
    _run(lost, backend, 0, "gone")  # saved is evicted (dumped), lost is resident
    dead = backend.mxq_models[0]
    dead.dump_cache_memory = lambda cache_id=0: (_ for _ in ()).throw(AssertionError("dumped a disposed Model"))
    backend.mxq_models = []  # dispose

    assert not lost.is_resident
    copied = saved.copy()  # uses the host copy
    saved.dump_cache_memory()
    assert saved.layers[0].buffer == [b"kept"] and copied.get_seq_length() == 4
    with pytest.raises(MobilintCacheStateError):
        lost.dump_cache_memory()  # its KV was never saved


def test_cache_with_slots_matching_the_default_layout_still_refuses_to_grow() -> None:
    backend = _Backend(n=1, k=1)
    cache = MobilintCache(backend.mxq_models, slots=[(0, 0)])

    with pytest.raises(ValueError, match="chosen slots"):
        cache.ensure_batch_size(2)
    with pytest.raises(ValueError, match="chosen slots"):
        cache.copy().ensure_batch_size(2)


def test_explicit_load_on_a_never_bound_cache_saves_the_slot_owner_first() -> None:
    backend = _Backend()
    a = MobilintCache(backend.mxq_models)
    _run(a, backend, 0, "aaa")  # A is resident; the backend's owner table now knows the Model

    restored = MobilintCache(backend.mxq_models)  # never bound, e.g. rebuilt from a saved snapshot
    restored.layers[0].buffer = [b"ccc"]
    restored.layers[0].buffer_seq_length = 3
    restored.load_cache_memory()

    assert _slot(backend) == "ccc"
    assert restored.is_resident and not a.is_resident
    _run(a, backend, 0, "!")  # A was saved before the load overwrote the slot
    assert _slot(backend) == "aaa!"


def _restore_from_file(cache: MobilintCache, backend: _Backend, content: str, tmp_path) -> None:
    """A file restore right after loading the model: the backend has no owner table yet."""
    mxq = backend.mxq_models[0]
    mxq.load_cache_memory_from = lambda cache_dir, cache_id=0: mxq.write(cache_id, content)
    (tmp_path / "seq_length.txt").write_text(f"{len(content)}\n")
    cache.load_cache_memory_from(str(tmp_path))


def test_file_restore_before_any_forward_continues_on_first_use(tmp_path) -> None:
    backend = _Backend()
    cache = MobilintCache(backend.mxq_models)
    _restore_from_file(cache, backend, "saved", tmp_path)

    _run(cache, backend, 0, "!")

    assert _slot(backend) == "saved!"
    assert backend.mxq_models[0].calls == []  # adopted as the owner: no reload


def test_file_restore_before_any_forward_survives_another_cache_running_first(tmp_path) -> None:
    backend = _Backend()
    restored = MobilintCache(backend.mxq_models)
    _restore_from_file(restored, backend, "saved", tmp_path)

    other = MobilintCache(backend.mxq_models)
    _run(other, backend, 0, "x")  # saves the restored KV before overwriting the slot
    _run(restored, backend, 0, "!")

    assert _slot(backend) == "saved!"


def test_two_file_restores_before_any_forward_keep_both_caches(tmp_path) -> None:
    backend = _Backend()
    first, second = MobilintCache(backend.mxq_models), MobilintCache(backend.mxq_models)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _restore_from_file(first, backend, "first", tmp_path / "a")
    _restore_from_file(second, backend, "second", tmp_path / "b")  # saves `first` before overwriting the slot

    assert second.is_resident and not first.is_resident
    _run(first, backend, 0, "!")
    assert _slot(backend) == "first!"
    _run(second, backend, 0, "?")
    assert _slot(backend) == "second?"


def test_offload_of_a_cache_restored_before_any_forward(tmp_path) -> None:
    backend = _Backend()
    restored = MobilintCache(backend.mxq_models)
    _restore_from_file(restored, backend, "saved", tmp_path)
    assert restored.is_resident

    restored.offload()
    backend.mxq_models[0].write(0, "scratch")  # e.g. an unrelated load into the slot

    assert not restored.is_resident and restored.host_nbytes == len(b"saved")
    _run(restored, backend, 0, "!")
    assert _slot(backend) == "saved!"
