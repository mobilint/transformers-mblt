import copy
import json
import os
import threading
import time
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Type, Union

import qbruntime
import torch
from transformers.cache_utils import Cache

from . import cache_residency as residency
from .cache_residency import MobilintCacheStateError, mxq_kv_signature

try:
    from transformers.cache_utils import CacheLayerMixin
except ImportError:
    # transformers < 4.54 compat shim: CacheLayerMixin was introduced in transformers 4.54.0.
    # This stub only satisfies subclassing at class-definition time so this module can import
    # cleanly for GPU-only workflows (e.g. running text-generation benchmarks on transformers
    # 4.53.x, which is required by some third-party custom modeling code such as EXAONE-3.5).
    # Any actual MobilintCache/MobilintLayer runtime path relies on the real CacheLayerMixin
    # contract and requires transformers>=4.54; instantiating those on the stub will fail
    # downstream. That is intentional — the stub must not silently masquerade as the real class.
    class CacheLayerMixin:  # type: ignore[no-redef]
        """Compat stub for transformers<4.54; MobilintCache paths require the real class."""

        pass


def is_whisper_beam_debug_trace_enabled() -> bool:
    """Return whether Whisper beam-cache debug tracing is enabled."""
    return bool(os.environ.get("MBLT_WHISPER_BEAM_DEBUG_TRACE"))


def append_whisper_beam_debug_event(event: dict[str, Any]) -> None:
    """Append one Whisper beam-cache debug event when tracing is enabled."""
    trace_path = os.environ.get("MBLT_WHISPER_BEAM_DEBUG_TRACE")
    if not trace_path:
        return
    payload = {"time_s": time.time(), **event}
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


class MobilintLayer(CacheLayerMixin):
    """One sequence's KV state for one MXQ KV slot ``(mxq_model, cache_id)``.

    The NPU slot holds one sequence at a time. Once this layer's KV is put on a slot through the owner table (a
    model binding it before an infer, an explicit load, a file restore; see
    :mod:`transformers_mblt.utils.cache_residency`), the layer is *tracked*: the table records whether the slot
    holds its KV, and the layer moves its KV between the slot and its host copy on its own. A layer that was never
    tracked keeps talking to the MXQ directly.

    Two host copies exist. ``buffer`` / ``buffer_seq_length`` is the snapshot the caller takes with
    :meth:`dump_cache_memory` and restores with :meth:`load_cache_memory`; nothing else touches it. ``_host_kv``
    is where the layer's KV goes when another cache takes its slot. ``_host_valid_len`` counts the leading
    positions where ``_host_kv`` equals the layer's current KV: saving or loading sets it, and every infer that
    writes from position ``p`` caps it at ``p``. A tracked layer that is not resident can be restored only when
    ``_host_valid_len`` covers ``_seen_tokens``. Blob lists are replaced, never mutated, so the two copies may
    share one list.
    """

    is_sliding = False

    def __init__(self, mxq_model: qbruntime.Model, cache_id: int = 0):
        self.mxq_model = mxq_model
        self.cache_id = cache_id
        self._seen_tokens = 0
        self.buffer: list[bytes] = []
        self.buffer_seq_length: Optional[int] = None
        # False after fake_prefill: the length is real but the KV contents are not, so they are never moved.
        self.kv_valid = True
        self.buffer_kv_valid = True  # kv_valid at the time ``buffer`` was taken
        self._host_kv: list[bytes] = []
        self._host_valid_len = 0
        self._tracked = False  # whether the owner table has managed this layer's KV
        # A speculative layer (the EAGLE-3 draft) only proposes tokens that the base model verifies, so its KV may
        # be partly stale: restoring it loads whatever host copy exists instead of refusing a short one.
        self.speculative = False
        self._kv_signature: Optional[tuple] = None
        self._owner_cache: Optional[Any] = None  # weakref to the multi-layer cache this layer belongs to, if any

    def lazy_initialization(self, key_states: torch.Tensor, value_states: Optional[torch.Tensor] = None):
        del value_states
        raise NotImplementedError("lazy_initialization is not implemented")

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: Optional[dict[str, Any]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("update is not implemented")

    def get_mask_sizes(self, cache_position: int | torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = int(cache_position) if isinstance(cache_position, int) else cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    def get_seq_length(self, cache_position=None) -> int:
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        return self.mxq_model.get_input_buffer_info()[0].max_cache_size

    def get_max_length(self) -> Optional[int]:
        """Return the maximum sequence length under the Transformers 5.13 cache API."""
        return self.get_max_cache_shape()

    def set_seq_length(self, seq_length: int) -> None:
        """Set the cached sequence length for an in-memory cache snapshot."""
        if seq_length < 0:
            raise ValueError(f"seq_length must be non-negative, got {seq_length}")
        if seq_length > self._seen_tokens and self._tracked and self.is_resident:
            # While resident, an infer may have written past the old length without advancing the cursor (the
            # caller sets it here instead): the host copy is no longer known to match those positions.
            self._host_valid_len = min(self._host_valid_len, self._seen_tokens)
        self._seen_tokens = seq_length

    def fake_prefill(self, seq_length: int) -> None:
        """Mark this layer as prefilled without restoring KV cache memory.

        This helper is intended for NPU decode TPS benchmarks where qbruntime
        receives the requested cache size directly and the actual KV payload is
        not needed for measuring decode compute cost.
        """
        self.reset()
        self.set_seq_length(seq_length)
        self.kv_valid = False

    def reset(self) -> None:
        self._seen_tokens = 0
        self.buffer = []
        self.buffer_seq_length = None
        self.kv_valid = True
        self.buffer_kv_valid = True
        self._host_kv = []
        self._host_valid_len = 0

    def reorder_cache(self, beam_idx: torch.LongTensor):
        raise NotImplementedError("reorder_cache is not implemented")

    def update_cache_position(self, cache_position: torch.Tensor):
        self._note_write()
        self._seen_tokens += cache_position.numel()

    def update_seen_tokens(self, num_new_seen_tokens: int):
        self._note_write()
        self._seen_tokens += num_new_seen_tokens

    # ------------------------------------------------------------------
    # Residency (see transformers_mblt.utils.cache_residency)
    # ------------------------------------------------------------------

    @property
    def is_managed(self) -> bool:
        """Whether the owner table tracks where this layer's KV lives."""
        return self._tracked

    @property
    def is_resident(self) -> bool:
        """Whether the NPU slot currently holds this layer's KV."""
        return residency.is_resident(self)

    @property
    def host_nbytes(self) -> int:
        """Bytes of KV this layer keeps in host memory (both copies, counted once when they are shared)."""
        lists = [self.buffer] if self.buffer is self._host_kv else [self.buffer, self._host_kv]
        return sum(
            len(blob) if isinstance(blob, (bytes, bytearray)) else int(getattr(blob, "nbytes", 0))
            for blobs in lists
            for blob in blobs
        )

    def _note_write(self) -> None:
        # The infer that just ran wrote KV from position ``_seen_tokens`` on.
        self._host_valid_len = min(self._host_valid_len, self._seen_tokens)

    def _signature(self) -> Optional[tuple]:
        if self._kv_signature is None:
            self._kv_signature = mxq_kv_signature(self.mxq_model)
        return self._kv_signature

    def _save_from_slot(self) -> None:
        """Dump the slot into the host buffer; called by the owner table while this layer is resident.

        Skips the dump when the host copy already matches every cached position (a ``copy()`` or
        ``dump_cache_memory()`` saved it and no infer wrote since), so evicting a just-copied layer costs nothing.
        """
        if self._seen_tokens > 0 and self.kv_valid:
            if self._host_kv and self._host_valid_len >= self._seen_tokens:
                return
            self._host_kv = self.mxq_model.dump_cache_memory(self.cache_id)
            self._host_valid_len = self._seen_tokens

    def _load_into_slot(self) -> None:
        """Load the host buffer into the slot, which the owner table is about to assign to this layer."""
        if self._seen_tokens <= 0 or not self.kv_valid:
            return
        if self._host_valid_len < self._seen_tokens:
            if self.speculative:
                if self._host_kv:
                    self.mxq_model.load_cache_memory(self._host_kv, self.cache_id)
                return
            raise MobilintCacheStateError(
                f"Cannot restore {self._seen_tokens} cached tokens: the host copy holds only "
                f"{self._host_valid_len}. The KV was on an NPU slot that was released (for example the backend "
                "was disposed) before it was saved. Call dump_cache_memory() before disposing, or reset() the cache."
            )
        self.mxq_model.load_cache_memory(self._host_kv, self.cache_id)

    def _retarget(self, mxq_model: Any, cache_id: int) -> None:
        """Point this layer at another slot, possibly on another model instance with the same MXQ."""
        if self._seen_tokens > 0 and self.kv_valid:
            source = self._signature()
            target = mxq_kv_signature(mxq_model)
            if source is not None and target is not None and source != target:
                raise MobilintCacheStateError(
                    "Cannot move this cache to a model whose MXQ has a different KV layout (different compiled "
                    "model, quantization, core mode or KV capacity). Use a cache built for that model."
                )
        self.mxq_model = mxq_model
        self.cache_id = int(cache_id)
        self._kv_signature = None

    def offload(self) -> None:
        """Save this layer's KV to its host copy and free its NPU slot for other caches."""
        residency.release(self)

    def dump_cache_memory(self):
        """Snapshot the current KV into ``buffer``; :meth:`load_cache_memory` restores it."""
        if self._tracked and not self.is_resident:
            # The KV is not on the NPU: it was saved to the host when this layer lost its slot.
            if self._seen_tokens > 0 and self.kv_valid and self._host_valid_len < self._seen_tokens:
                raise MobilintCacheStateError(
                    f"Cannot snapshot {self._seen_tokens} cached tokens: only {self._host_valid_len} were saved "
                    "before the NPU slot was released."
                )
            self.buffer = self._host_kv
        else:
            self.buffer = self.mxq_model.dump_cache_memory(self.cache_id)
            self._host_kv = self.buffer
            self._host_valid_len = self.get_seq_length()
        self.buffer_seq_length = self.get_seq_length()
        self.buffer_kv_valid = self.kv_valid

    def load_cache_memory(self):
        if self.buffer_seq_length is not None:
            self.set_seq_length(self.buffer_seq_length)
            self._host_kv = self.buffer
            self._host_valid_len = self.buffer_seq_length
            # A snapshot of a fake-prefilled layer restores its length, not KV contents.
            self.kv_valid = self.buffer_kv_valid
        # Restore through the owner table so the cache that holds the slot is saved first.
        residency.acquire(self, self.mxq_model, self.cache_id, reload=True)

    def dump_cache_memory_to(self, cache_dir: str):
        if self._tracked and not self.is_resident:
            # The slot holds another cache's KV: put this layer's KV back on it before writing the files.
            residency.acquire(self, self.mxq_model, self.cache_id)
        self.mxq_model.dump_cache_memory_to(cache_dir, self.cache_id)
        seq_path = Path(cache_dir) / "seq_length.txt"
        seq_path.write_text(f"{self.get_seq_length()}\n", encoding="utf-8")

    def load_cache_memory_from(self, cache_dir: str):
        # The file overwrites the slot: save the cache that holds it, then take it over.
        residency.evict(self.mxq_model, self.cache_id)
        self.reset()
        seq_path = Path(cache_dir) / "seq_length.txt"
        if seq_path.exists():
            seq_length = int(seq_path.read_text(encoding="utf-8").strip())
        else:
            seq_length = 0
        self.set_seq_length(seq_length)
        self.mxq_model.load_cache_memory_from(cache_dir, self.cache_id)
        residency.claim(self)

    def copy(self) -> "MobilintLayer":
        """Return an independent snapshot of this sequence's KV.

        A resident layer saves its slot first, so the copy's host buffer is current. The copy is not resident:
        its first bind loads the buffer into a slot.
        """
        if self.is_resident:
            self._save_from_slot()
        copied = MobilintLayer(self.mxq_model, self.cache_id)
        copied._seen_tokens = self._seen_tokens
        copied.buffer = copy.deepcopy(self.buffer)
        copied.buffer_seq_length = self.buffer_seq_length
        copied.kv_valid = self.kv_valid
        copied.speculative = self.speculative
        copied.buffer_kv_valid = self.buffer_kv_valid
        copied._host_kv = self._host_kv
        copied._host_valid_len = self._host_valid_len
        copied._tracked = self._tracked
        copied._kv_signature = self._kv_signature
        return copied


class MobilintCache(Cache):
    def __init__(
        self,
        mxq_models: Union[List[qbruntime.Model], qbruntime.Model],
        per_model_batch: int = 1,
        *,
        batch_size: Optional[int] = None,
        slots: Optional[Sequence[Tuple[int, int]]] = None,
    ):
        """Create a cache with fresh logical cursors across one or more runtime models.

        The cache dualizes KV state along ``(model_idx, cache_id)``: it holds
        ``N = len(mxq_models)`` Model handles with ``K = per_model_batch``
        cache slots each, laid out as flat rows so external callers can keep
        the historical row-index API. Row ``i`` maps to
        ``(i // K, i % K)``; helpers :meth:`slot_of`, :meth:`model_of`, and
        :meth:`group_by_model` expose that routing without leaking the slot
        concept into upstream dispatch.

        qbruntime selects the readable KV prefix from the ``cache_size`` passed
        to inference; fresh layers therefore start at sequence length zero,
        causing the next inference to overwrite cache entries from the
        beginning rather than making a previous request's KV prefix
        addressable.

        Args:
            mxq_models: One :class:`qbruntime.Model` or a list of them. A
                single Model is promoted to a length-1 list for backward
                compatibility.
            per_model_batch: Cache slots per Model (``K``). Total rows =
                ``N * K``.
            batch_size: Legacy keyword-only alias for ``per_model_batch``.
                Only accepted when the cache hosts a single Model (``N = 1``)
                so that ``MobilintCache(model, batch_size=K)`` keeps working;
                rejected with a multi-Model list because the legacy alias
                would silently double the total capacity and pin the first
                ``K`` rows entirely to slot 0.
            slots: Place row ``i`` on ``slots[i] = (model_idx, cache_id)``
                instead of the default ``divmod(i, K)`` layout. The cache then
                has ``len(slots)`` rows; a serving engine uses this to keep one
                cache per request on the slot it assigned. Slots must be
                distinct and inside ``N * K``.

        Raises:
            TypeError: If both ``per_model_batch`` and legacy ``batch_size``
                are provided, or if the legacy ``batch_size`` is combined
                with a multi-Model list.
            ValueError: If ``mxq_models`` is an empty list.
        """
        if isinstance(mxq_models, list):
            models_list: List[qbruntime.Model] = list(mxq_models)
        else:
            models_list = [mxq_models]

        if not models_list:
            raise ValueError("mxq_models must contain at least one Model")

        if batch_size is not None:
            if per_model_batch != 1:
                raise TypeError(
                    "Pass either per_model_batch or the legacy batch_size, not both"
                )
            if len(models_list) > 1:
                raise TypeError(
                    "MobilintCache legacy batch_size= is a single-Model (N=1) shim; "
                    f"pass per_model_batch=K instead (got N={len(models_list)} Models). "
                    "The legacy alias would misroute rows across slots and defeat "
                    "multi-slot dispatch."
                )
            per_model_batch = int(batch_size)

        self.mxq_models: List[qbruntime.Model] = models_list
        self.k_per_model: int = max(1, int(per_model_batch))
        self.n_models: int = len(self.mxq_models)
        self._batch_size: int = self.n_models * self.k_per_model

        if slots is None:
            row_slots = [
                (model_idx, cache_id) for model_idx in range(self.n_models) for cache_id in range(self.k_per_model)
            ]
        else:
            row_slots = [(int(model_idx), int(cache_id)) for model_idx, cache_id in slots]
            if not row_slots:
                raise ValueError("slots must name at least one (model_idx, cache_id) slot")
            if len(set(row_slots)) != len(row_slots):
                raise ValueError(f"slots must be distinct, got {row_slots}")
            for model_idx, cache_id in row_slots:
                if not (0 <= model_idx < self.n_models and 0 <= cache_id < self.k_per_model):
                    raise ValueError(
                        f"slot ({model_idx}, {cache_id}) is outside the cache's N={self.n_models} x "
                        f"K={self.k_per_model} topology"
                    )
            self._batch_size = len(row_slots)
        self._row_slots: List[Tuple[int, int]] = row_slots
        self._explicit_slots = slots is not None

        self.layers: list[MobilintLayer] = [
            MobilintLayer(self.mxq_models[model_idx], cache_id) for model_idx, cache_id in row_slots
        ]
        self.layer_classes = MobilintLayer

        self.num_hidden_layers = 1
        self.cache_processor = None
        # M-RoPE offset per row (Qwen2-VL, Qwen3-VL, Qwen3-ASR): text position minus cache position after the
        # prompt. It belongs to the sequence in this cache, not to the model, so two conversations can alternate.
        self.rope_deltas: Optional[torch.Tensor] = None
        # Whether the rows were joined from request caches by ``from_rows`` (see there).
        self.joined_rows = False

    @property
    def batch_size(self) -> int:
        """Return the logical batch capacity across all runtime-model slots."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        """Keep the legacy single-model growth path compatible with HF 5.13+."""
        self._batch_size = int(value)

    @property
    def mxq_model(self) -> Optional[qbruntime.Model]:
        """First Model handle for callers written against the pre-multi-Model API."""
        return self.mxq_models[0] if self.mxq_models else None

    def slot_of(self, row: int) -> Tuple[int, int]:
        """Return ``(model_idx, local_cache_id)`` for a flat row index."""
        row = int(row)
        if row < 0 or row >= len(self.layers):
            raise IndexError(
                f"row {row} out of range for cache with {len(self.layers)} rows "
                f"(N={self.n_models}, K={self.k_per_model})"
            )
        return self._row_slots[row]

    # ------------------------------------------------------------------
    # Residency (see transformers_mblt.utils.cache_residency)
    # ------------------------------------------------------------------

    @property
    def is_resident(self) -> bool:
        """Whether every row's KV is on its NPU slot."""
        return all(layer.is_resident for layer in self.layers)

    @property
    def host_nbytes(self) -> int:
        """Bytes of KV this cache keeps in host memory."""
        return sum(layer.host_nbytes for layer in self.layers)

    def offload(self) -> None:
        """Save every resident row to host memory and free its NPU slot for other caches."""
        for layer in self.layers:
            layer.offload()

    def adopt_backend(self, backend: Any) -> None:
        """Point the cache at ``backend``'s live Model handles, moving it from another model instance if needed.

        Rows keep their ``(model_idx, cache_id)`` placement, so the backend must have the same ``N x K``
        topology. KV moves lazily: each row is saved on its old slot and loaded on the new one the next time it
        is bound.
        """
        backend_models = list(getattr(backend, "mxq_models", None) or [])
        if len(backend_models) == len(self.mxq_models) and all(
            mine is theirs for mine, theirs in zip(self.mxq_models, backend_models)
        ):
            return
        backend_k = max(1, int(getattr(backend, "k_per_model", 1) or 1))
        if len(backend_models) != self.n_models or backend_k != self.k_per_model:
            raise ValueError(
                "Cannot move this MobilintCache to a backend with a different slot topology: cache "
                f"N={self.n_models}, K={self.k_per_model}; backend N={len(backend_models)}, K={backend_k}."
            )
        self.mxq_models = backend_models

    def bind_rows(self, backend: Any, rows: Iterable[int]) -> None:
        """Make the NPU hold the KV of ``rows`` before an infer reads and writes them."""
        from .cache_residency import bind_cache_layers

        self.adopt_backend(backend)
        bind_cache_layers(
            (self.layers[row], self.mxq_models[self.slot_of(row)[0]], self.slot_of(row)[1]) for row in rows
        )

    def model_of(self, row: int) -> qbruntime.Model:
        """Return the :class:`qbruntime.Model` that owns the KV state for ``row``."""
        model_idx, _ = self.slot_of(row)
        return self.mxq_models[model_idx]

    def group_by_model(self, rows: Iterable[int]) -> Dict[int, List[Tuple[int, int]]]:
        """Group flat rows by owning Model.

        Args:
            rows: Iterable of flat row indices.

        Returns:
            Mapping ``model_idx -> [(row, local_cache_id), ...]`` in the input
            order, so upstream dispatch can issue one blocking
            :meth:`qbruntime.Model.infer` per Model without inspecting slots.
        """
        grouped: Dict[int, List[Tuple[int, int]]] = {}
        for row in rows:
            model_idx, local_cache_id = self.slot_of(int(row))
            grouped.setdefault(model_idx, []).append((int(row), local_cache_id))
        return grouped

    def get_seq_length(self, index: int = 0) -> int:
        return self.layers[index].get_seq_length()

    def set_seq_length(self, sequence_lengths: Union[dict[int, int], int], index: int = 0) -> None:
        """Set cached sequence lengths for one cache entry or a batch of entries."""
        if isinstance(sequence_lengths, int):
            self.layers[index].set_seq_length(sequence_lengths)
            return
        for cache_id, seq_len in sequence_lengths.items():
            self.layers[cache_id].set_seq_length(seq_len)

    def fake_prefill(self, sequence_lengths: Union[dict[int, int], int], index: int = 0) -> None:
        """Mark one or more cache entries as prefilled without loading cache memory.

        Args:
            sequence_lengths: Single sequence length or per-cache-id sequence
                lengths to expose via ``get_seq_length()``. A single sequence length
                is applied to every cache entry in the batch.
            index: Unused compatibility argument for scalar sequence lengths.

        Raises:
            ValueError: If any sequence length is negative.
        """
        # A synthetic prefix has no image tokens: the offset of the prompt the row held before no longer applies.
        if isinstance(sequence_lengths, int):
            for layer in self.layers:
                layer.fake_prefill(sequence_lengths)
            self.rope_deltas = None
            return
        for cache_id, seq_len in sequence_lengths.items():
            self.layers[cache_id].fake_prefill(seq_len)
            if self.rope_deltas is not None and cache_id < int(self.rope_deltas.shape[0]):
                self.rope_deltas = self.rope_deltas.clone()
                self.rope_deltas[cache_id] = 0

    def update_cache_position(self, cache_position: torch.Tensor, index: int = 0):
        self.layers[index].update_cache_position(cache_position)

    def update_seen_tokens(self, sequence_lengths: Union[dict[int, int], int], index: int = 0):
        if isinstance(sequence_lengths, int):
            self.layers[index].update_seen_tokens(sequence_lengths)
            return
        for cache_id, seq_len in sequence_lengths.items():
            self.layers[cache_id].update_seen_tokens(seq_len)

    def dump_cache_memory(self):
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].dump_cache_memory()

    def load_cache_memory(self):
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx].load_cache_memory()

    _ROPE_DELTAS_FILE = "rope_deltas.json"

    def dump_cache_memory_to(self, cache_dir: str, index: int = 0):
        self.layers[index].dump_cache_memory_to(cache_dir)
        # Rows past the active batch have no offset: saved as unknown and left out on restore.
        has_row = self.rope_deltas is not None and index < int(self.rope_deltas.shape[0])
        row = self.rope_deltas[index].tolist() if has_row else None
        (Path(cache_dir) / self._ROPE_DELTAS_FILE).write_text(json.dumps({"rope_deltas": row}), encoding="utf-8")

    def load_cache_memory_from(self, cache_dir: str, index: int = 0):
        self.layers[index].load_cache_memory_from(cache_dir)
        rope_path = Path(cache_dir) / self._ROPE_DELTAS_FILE
        row = json.loads(rope_path.read_text(encoding="utf-8")).get("rope_deltas") if rope_path.exists() else None
        known = 0 if self.rope_deltas is None else int(self.rope_deltas.shape[0])
        if row is None:
            if self.layers[index].get_seq_length() > 0:
                # This row holds tokens whose offset was not saved: recomputed from the full input, not guessed.
                self.rope_deltas = None
            elif index < known:
                # An empty row inside the active batch: its next prompt computes the offset.
                self.rope_deltas = self.rope_deltas.clone()
                self.rope_deltas[index] = 0
            # An empty row past the active batch leaves the offsets alone, so they keep the active batch's size.
            return
        # The offsets cover the rows that had one when saved (the active batch), not the cache's full capacity.
        size = max(known, index + 1)
        if not all(self.layers[i].get_seq_length() == 0 for i in range(known, self.batch_size) if i != index):
            self.rope_deltas = None  # other rows hold tokens with unknown offsets
            return
        rope_deltas = torch.zeros((size, len(row)), dtype=torch.long)
        if known:
            rope_deltas[:known] = self.rope_deltas
        rope_deltas[index] = torch.tensor(row, dtype=torch.long)
        self.rope_deltas = rope_deltas

    def reset(self) -> None:
        """Reset all cache entries in this Mobilint cache."""
        for layer in self.layers:
            layer.reset()
        self.rope_deltas = None

    def ensure_batch_size(self, batch_size: int) -> None:
        """Grow logical cache entries so batched generation can track each active row.

        Growth beyond ``N * K`` is only supported on the legacy single-Model
        hardware-batch path (``N == 1``); multi-Model caches must be sized
        upfront because slot count is fixed by the backend.
        """
        batch_size = max(1, int(batch_size))
        if batch_size <= self.batch_size:
            return
        if self._explicit_slots:
            raise ValueError(
                f"cannot grow a MobilintCache whose rows sit on chosen slots {self._row_slots}; "
                "build it with every slot it needs"
            )
        if self.n_models != 1:
            raise ValueError(
                f"cannot grow multi-Model cache beyond {self.batch_size} rows "
                f"(N={self.n_models}, K={self.k_per_model}); allocate a larger cache upfront"
            )
        only_model = self.mxq_models[0]
        for cache_id in range(self.batch_size, batch_size):
            self.layers.append(MobilintLayer(only_model, cache_id))
            self._row_slots.append((0, cache_id))
        self.batch_size = batch_size
        self.k_per_model = batch_size

    @classmethod
    def from_rows(cls, caches: Sequence["MobilintCache"]) -> "MobilintCache":
        """Return a batch cache whose row ``i`` is ``caches[i]``'s only row, shared rather than copied.

        A serving engine keeps one cache per request and runs the requests of a step together: pass the view to a
        batched forward (with an ``attention_mask``) and every request's cache advances by its own row, each from its
        own length. The rows keep their slots, so the requests must sit on distinct slots of the same backend.

        The view carries no ``rope_deltas``: an M-RoPE model cannot tell from the rows where each one's positions
        continue, so it needs the caller's ``position_ids`` for a view that holds tokens
        (:func:`require_positions_for_joined_rows`).
        """
        caches = list(caches)
        if not caches:
            raise ValueError("from_rows needs at least one cache")
        first = caches[0]
        if not isinstance(first, cls):
            raise TypeError(f"{cls.__name__}.from_rows got a {type(first).__name__}")
        for cache in caches:
            if type(cache) is not type(first) or isinstance(cache, MobilintBeamCache):
                raise TypeError(
                    f"from_rows joins caches of one type that move KV through the owner table; got "
                    f"{type(first).__name__} and {type(cache).__name__}"
                )
            if len(cache.layers) != 1:
                raise ValueError(f"from_rows joins one-row caches; got a cache with {len(cache.layers)} rows")
            if (
                cache.k_per_model != first.k_per_model
                or len(cache.mxq_models) != len(first.mxq_models)
                or any(mine is not theirs for mine, theirs in zip(cache.mxq_models, first.mxq_models))
            ):
                raise ValueError(
                    "from_rows joins caches of one backend; move the others first with adopt_backend(backend)"
                )
        view = first._empty_like([cache.slot_of(0) for cache in caches])
        view.layers = [cache.layers[0] for cache in caches]
        view.joined_rows = True
        return view

    def _empty_like(self, slots: Sequence[Tuple[int, int]]) -> "MobilintCache":
        """An empty cache of this type on the same backend, with rows on ``slots``."""
        return MobilintCache(list(self.mxq_models), per_model_batch=self.k_per_model, slots=slots)

    def copy(self):
        """Return an independent snapshot; resident rows save their KV first (see :meth:`MobilintLayer.copy`)."""
        copied = MobilintCache(list(self.mxq_models), per_model_batch=self.k_per_model)
        copied._row_slots = list(self._row_slots)
        copied._explicit_slots = self._explicit_slots
        copied._batch_size = self._batch_size
        copied.layers = [layer.copy() for layer in self.layers]
        copied.rope_deltas = copy_rope_deltas(self)
        copied.joined_rows = self.joined_rows
        return copied


def require_positions_for_joined_rows(cache: Any, position_ids: Optional[torch.Tensor]) -> None:
    """Refuse an M-RoPE forward over a :meth:`MobilintCache.from_rows` batch that holds tokens without ``position_ids``.

    Each row continues from its own length and its own M-RoPE offset, which the view does not carry, so upstream's
    positions (one start for the batch, one offset on the model) would be wrong for some rows.
    """
    if position_ids is None and getattr(cache, "joined_rows", False) and _cache_holds_tokens(cache):
        raise ValueError(
            "This batch was joined from request caches with MobilintCache.from_rows: pass position_ids (each row's "
            "M-RoPE positions, continuing from its own cached tokens)."
        )


def copy_rope_deltas(cache: Any) -> Optional[torch.Tensor]:
    """Return an independent copy of ``cache.rope_deltas`` (``None`` when the cache has none)."""
    rope_deltas = getattr(cache, "rope_deltas", None)
    return None if rope_deltas is None else rope_deltas.clone()


def use_cache_rope_deltas(holder: Any, cache: Any) -> None:
    """Point ``holder.rope_deltas`` (the attribute upstream M-RoPE code reads) at ``cache``'s offset.

    Upstream Qwen-VL code keeps ``rope_deltas`` on the model, so a cache that resumes after another conversation ran
    would get that conversation's offset. Call this before a forward or ``generate()`` step with ``cache``, and
    :func:`keep_cache_rope_deltas` after it. A cache without an offset sets ``None``, whether or not it holds tokens:

    - an empty cache is about to take a prompt, which computes its own offset. Upstream skips that for a prompt
      without images, whose offset is 0; leaving the previous conversation's offset on ``holder`` would hand it to
      this cache and shift every later position.
    - a cache with tokens but no offset (from an older release, or a file without one) recomputes it from the full
      input instead of borrowing another sequence's.
    - no cache means the forward starts a new sequence and builds its cache: the same as an empty one.

    ``generate()`` computes a prompt's offset before its forward and gives it to the cache first (see
    :class:`~transformers_mblt.utils.generation_utils.MobilintCacheRopeDeltasMixin`), so the forward finds it there.
    """
    if cache is None:
        holder.rope_deltas = None
        return
    if not hasattr(cache, "get_seq_length"):
        return
    holder.rope_deltas = getattr(cache, "rope_deltas", None)


def _cache_holds_tokens(cache: Any) -> bool:
    """Whether any row of ``cache`` holds tokens (``get_seq_length()`` alone reads row 0)."""
    layers = getattr(cache, "layers", None)
    if layers:
        return any(layer.get_seq_length() > 0 for layer in layers)
    return cache.get_seq_length() > 0


_ROPE_DELTAS_LOCKS: "weakref.WeakKeyDictionary[Any, threading.RLock]" = weakref.WeakKeyDictionary()
_ROPE_DELTAS_LOCKS_GUARD = threading.Lock()


@contextmanager
def cache_rope_deltas(holder: Any, cache: Any) -> Iterator[Callable[[Any], None]]:
    """Run upstream M-RoPE code with ``cache``'s offset on ``holder``, one request at a time per holder.

    ``holder.rope_deltas`` is shared by every request on the model, so loading a cache's offset, running the code
    that reads or recomputes it, and storing it back happen under one lock. Yields ``keep(target)``, which stores
    the holder's offset on ``target`` (the cache the step filled, which a forward without a cache builds itself).
    """
    with _ROPE_DELTAS_LOCKS_GUARD:
        lock = _ROPE_DELTAS_LOCKS.get(holder)
        if lock is None:
            lock = _ROPE_DELTAS_LOCKS[holder] = threading.RLock()
    with lock:
        use_cache_rope_deltas(holder, cache)
        yield lambda target: keep_cache_rope_deltas(holder, target)


def keep_cache_rope_deltas(holder: Any, cache: Any) -> None:
    """Store ``holder.rope_deltas`` on ``cache`` after a step that may have computed it."""
    if cache is None or not hasattr(cache, "get_seq_length"):
        return
    cache.rope_deltas = getattr(holder, "rope_deltas", None)


def _resolve_language_model_candidate(model: Any) -> Optional[Any]:
    """Return the nested language model commonly used by VLM wrappers."""
    nested_model = getattr(model, "model", None)
    if nested_model is not None:
        language_model = getattr(nested_model, "language_model", None)
        if language_model is not None:
            return language_model
    return getattr(model, "language_model", None)


def _call_maybe_getter(obj: Any, name: str) -> Optional[Any]:
    """Return an attribute value, calling it when it is a zero-argument getter."""
    candidate = getattr(obj, name, None)
    if candidate is None:
        return None
    if callable(candidate):
        try:
            return candidate()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
    return candidate


def resolve_multi_slot_backend(model: Any) -> Optional[Any]:
    """Return the Mobilint NPU backend that hosts one or more Model slots for ``model``.

    Walks ``model`` and its nested language model (the two shapes current VLM
    wrappers use — ``model.language_model`` and ``model.model.language_model``)
    looking for an ``npu_backend`` that exposes a non-empty ``mxq_models``
    list. Returns ``None`` for models that do not expose a Mobilint backend
    (e.g. unit-test stubs that only carry a bare ``get_cache_mxq_model``).
    """
    for candidate in (model, _resolve_language_model_candidate(model)):
        if candidate is None:
            continue
        backend = getattr(candidate, "npu_backend", None)
        if backend is None:
            continue
        if getattr(backend, "mxq_models", None):
            return backend
    return None


def resolve_cache_mxq_model(model: Any) -> Optional[qbruntime.Model]:
    """Return the single ``qbruntime.Model`` used by the legacy cache path.

    Falls back through ``get_cache_mxq_model``, then ``get_mxq_model`` on both
    ``model`` and its nested language model, so wrappers that override
    ``get_cache_mxq_model`` to delegate into ``language_model`` continue to work.
    """
    for candidate in (model, _resolve_language_model_candidate(model)):
        if candidate is None:
            continue
        for getter_name in ("get_cache_mxq_model", "get_mxq_model"):
            mxq_model = _call_maybe_getter(candidate, getter_name)
            if mxq_model is not None:
                return mxq_model
    return None


def build_mobilint_cache_from_model(
    model: Any,
    batch_size: int,
    *,
    cache_cls: Type[MobilintCache] = MobilintCache,
    **cache_kwargs: Any,
) -> MobilintCache:
    """Build a Mobilint cache routed across every ``qbruntime.Model`` slot the backend hosts.

    Uses the multi-slot ``cache_cls(mxq_models, per_model_batch=K)`` signature
    when the model exposes a multi-slot :class:`MobilintNPUBackend`, so
    :meth:`MobilintCache.slot_of` routes each flat row to its owning Model.
    Falls back to ``cache_cls(mxq_model, batch_size=batch_size)`` when the
    backend cannot be resolved (unit-test stubs and single-Model wrappers
    without a discoverable backend).

    On the resolved-backend path the cache is sized upfront to the backend's
    aggregate ``N * K`` capacity and a caller ``batch_size`` that exceeds that
    capacity is rejected with :class:`ValueError`; growing the cache after
    construction via :meth:`MobilintCache.ensure_batch_size` would silently
    misroute rows to slots that do not exist because no additional
    :class:`qbruntime.Model` is launched. The backend-less legacy path still
    passes ``batch_size`` through to ``cache_cls`` so ``ensure_batch_size``
    growth remains available for single-Model hardware-batch callers.
    """
    backend = resolve_multi_slot_backend(model)
    if backend is None:
        # Backend-less legacy path: allow ensure_batch_size growth via
        # ``cache_cls``'s single-Model constructor. This is the only caller
        # that may grow past ``cache.batch_size`` after construction.
        mxq_model = resolve_cache_mxq_model(model)
        if mxq_model is None:
            raise RuntimeError(
                "Cannot build MobilintCache: no Mobilint NPU backend or "
                "get_cache_mxq_model resolver on this model."
            )
        return cache_cls(mxq_model, batch_size=batch_size, **cache_kwargs)

    # Multi-slot resolved-backend path: size is fixed by (N, K) at build.
    # ``ensure_batch_size`` cannot grow the cache here because the backend
    # launched exactly N Model slots; a larger request must lower
    # ``max_batch_size`` (spread across more devices) or recompile with a
    # larger compiled batch axis K.
    mxq_models = list(getattr(backend, "mxq_models", []) or [])
    if not mxq_models:
        raise RuntimeError("Mobilint NPU backend has no loaded Model slots.")
    k_per_model = int(getattr(backend, "k_per_model", 1) or 1)
    capacity = len(mxq_models) * max(1, k_per_model)
    if batch_size > capacity:
        raise ValueError(
            f"Requested batch_size={batch_size} exceeds Mobilint NPU backend "
            f"capacity N*K = {len(mxq_models)}*{k_per_model} = {capacity}. "
            "Recompile the MXQ with a larger compiled batch axis K, or raise "
            "the backend's max_batch_size at construction time so it launches "
            "more Model slots."
        )
    return cache_cls(mxq_models, per_model_batch=k_per_model, **cache_kwargs)


def cache_matches_backend_topology(cache: Any, model: Any) -> bool:
    """Return True when ``cache`` was built for ``model``'s current backend topology.

    Cache reuse across a dispose+recreate cycle must invalidate on any change to
    ``(mxq_models, k_per_model)`` because :meth:`MobilintCache.slot_of` and
    :meth:`MobilintCache.model_of` bake that routing in at construction time. Two
    backends with the same aggregate row capacity (e.g. ``N=2, K=2`` vs
    ``N=4, K=1``) hand out incompatible ``(model_idx, cache_id)`` pairs, so a
    reuse guard that only checks ``batch_size`` would silently misroute rows.

    Legacy single-Model fallback (no discoverable multi-slot backend): the cache
    was built via ``cache_cls(mxq_model, batch_size=B)`` and may have grown via
    :meth:`MobilintCache.ensure_batch_size`, so only the Model handle identity
    is compared and ``k_per_model`` growth is left to the caller's capacity check.
    """
    if not isinstance(cache, MobilintCache):
        return False

    backend = resolve_multi_slot_backend(model)
    if backend is not None:
        backend_mxq_models = list(getattr(backend, "mxq_models", []) or [])
        backend_k_per_model = int(getattr(backend, "k_per_model", 1) or 1)
        if cache.k_per_model != backend_k_per_model:
            return False
    else:
        fallback_mxq_model = resolve_cache_mxq_model(model)
        if fallback_mxq_model is None:
            return False
        backend_mxq_models = [fallback_mxq_model]

    if len(cache.mxq_models) != len(backend_mxq_models):
        return False
    for cached_model, backend_model in zip(cache.mxq_models, backend_mxq_models):
        if cached_model is not backend_model:
            return False
    return True


class MobilintReplaySlot:
    """Owner-table entry for a cache that rebuilds its slot by replaying tokens instead of moving KV.

    :class:`MobilintBeamCache` keeps no KV of its own: it records which token history the slot holds and replays
    whatever is missing. When another cache (or a forward without a cache) takes the slot, or the beam cache moves
    to another slot, that record no longer describes the slot, so it is forgotten and the next forward replays the
    full history. Nothing is dumped or loaded.
    """

    speculative = False

    def __init__(self, cache: "MobilintBeamCache", mxq_model: Any, cache_id: int = 0) -> None:
        self._cache_ref = weakref.ref(cache)
        self.mxq_model = mxq_model
        self.cache_id = int(cache_id)
        self._tracked = False

    def _forget(self) -> None:
        cache = self._cache_ref()
        if cache is not None:
            cache.forget_active_tokens()

    def _save_from_slot(self) -> None:
        self._forget()  # the slot is about to hold something else

    def _load_into_slot(self) -> None:
        self._forget()  # whatever the slot holds now is not this cache's history

    def _retarget(self, mxq_model: Any, cache_id: int) -> None:
        self.mxq_model = mxq_model
        self.cache_id = int(cache_id)


class MobilintBeamCache(MobilintCache):
    """Mobilint beam cache tracked by token histories instead of KV snapshots.

    qbruntime owns one active KV cache. This class tracks the token history for
    each logical beam and the token history currently represented by the active
    qbruntime cache. Callers can compare a target beam history with the active
    history, skip the common prefix, and forward only the suffix with the proper
    cache position.

    The beam-cache dispatch path is ``N == 1`` only: it issues one blocking
    ``mxq_model.infer`` on the single tracked slot with no cross-slot routing or
    beam-cache reorder. The ``mxq_models`` argument alone cannot detect the
    owning-backend topology when a caller passes ``get_cache_mxq_model()`` (which
    returns slot 0 as a single :class:`qbruntime.Model`) even though the backend
    launched ``N > 1`` slots. Callers that own a Mobilint NPU backend should pass
    ``n_slots=backend.dispatcher.n_slots`` (or equivalent) so the invariant is
    enforced at construction time rather than deep in generation. When
    ``n_slots`` is not supplied, only the existing multi-Model list guard runs
    to preserve backward compatibility for unit-test stubs.
    """

    def __init__(
        self,
        mxq_models: Union[List[qbruntime.Model], qbruntime.Model],
        batch_size: int = 1,
        *,
        n_slots: Optional[int] = None,
    ) -> None:
        if isinstance(mxq_models, list) and len(mxq_models) > 1:
            raise NotImplementedError(
                "MobilintBeamCache does not support multi-Model dispatch (N > 1); "
                "beam search keeps N=1 (encoder-decoder) — use MobilintCache for N > 1"
            )
        if n_slots is not None and int(n_slots) > 1:
            raise NotImplementedError(
                "MobilintBeamCache does not support multi-slot dispatch "
                f"(owning backend launched N={int(n_slots)} slots); beam search "
                "keeps N=1. Cap the owning backend's max_batch_size at K so it "
                "keeps a single slot, or compile a batched (K>1) MXQ so slot-0 "
                "hardware batching serves the load — use MobilintCache for N > 1."
            )
        super().__init__(mxq_models=mxq_models, batch_size=batch_size)
        self._beam_token_histories: list[list[int]] = [[] for _ in range(self.batch_size)]
        self._beam_source_indices: list[int | None] = [None for _ in range(self.batch_size)]
        self._active_token_history: list[int] = []
        self._active_source_index: int | None = None
        self._beam_seq_lengths: list[int] = [0 for _ in range(self.batch_size)]
        self._replay_slot = MobilintReplaySlot(self, self.mxq_model, 0)

    def bind_active(self, mxq_model: Any) -> None:
        """Take slot ``(mxq_model, 0)`` before replaying on it.

        If the slot held another cache, or this cache comes from another slot, the active history is forgotten
        and the next forward replays every beam from the start.
        """
        residency.acquire(self._replay_slot, mxq_model, 0)

    def truncate_active_tokens(self, length: int) -> None:
        """Keep only the first ``length`` tokens of the active history (a replay is about to overwrite the rest)."""
        del self._active_token_history[int(length) :]

    def forget_active_tokens(self) -> None:
        """Forget which history the slot holds; the next forward replays from the start."""
        self._active_token_history = []
        self._active_source_index = None

    @property
    def is_resident(self) -> bool:
        """Whether the slot still holds the active token history."""
        return residency.is_resident(self._replay_slot)

    @property
    def host_nbytes(self) -> int:
        """Beam caches replay tokens instead of saving KV, so they keep none in host memory."""
        return 0

    def offload(self) -> None:
        """Free the slot for other caches; the next forward replays the full history."""
        residency.release(self._replay_slot)

    def reset(self) -> None:
        """Reset active qbruntime cache bookkeeping and clear beam token histories."""
        super().reset()
        self._beam_token_histories = [[] for _ in range(self.batch_size)]
        self._beam_source_indices = [None for _ in range(self.batch_size)]
        self._active_token_history = []
        self._active_source_index = None
        self._beam_seq_lengths = [0 for _ in range(self.batch_size)]

    def matches_live_topology(
        self,
        expected_mxq_model: qbruntime.Model,
        n_slots: Optional[int] = None,
    ) -> bool:
        """Return True when this cache can be reused for the current backend topology.

        The beam-cache contract is ``N == 1``: one active qbruntime cache and one
        Model handle. Callers reuse an existing beam cache via :meth:`reset`, but
        the constructor-time guard does not re-run on that path. This helper
        performs the same invariant check so ``_get_cache`` implementations can
        rebuild rather than reset when either (a) the owning backend now hosts
        more than one slot (breaking the ``N == 1`` contract) or (b) the stored
        Model handle no longer matches the current slot 0 (the backend was
        disposed and re-created with a fresh Model).
        """
        if n_slots is not None and int(n_slots) > 1:
            return False
        cached_models = getattr(self, "mxq_models", None)
        if not cached_models:
            return False
        if len(cached_models) != 1:
            return False
        return cached_models[0] is expected_mxq_model

    def ensure_batch_size(self, batch_size: int) -> None:
        """Grow logical beam token storage for beam-expanded generation."""
        previous_batch_size = self.batch_size
        super().ensure_batch_size(batch_size)
        if self.batch_size <= previous_batch_size:
            return
        self._beam_token_histories.extend([[] for _ in range(self.batch_size - previous_batch_size)])
        self._beam_source_indices.extend([None for _ in range(self.batch_size - previous_batch_size)])
        self._beam_seq_lengths.extend([0 for _ in range(self.batch_size - previous_batch_size)])

    def get_seq_length(self, index: int = 0) -> int:
        """Return the stored sequence length for one logical beam."""
        self.ensure_batch_size(index + 1)
        return self._beam_seq_lengths[index]

    def set_seq_length(self, sequence_lengths: Union[dict[int, int], int], index: int = 0) -> None:
        """Set stored sequence lengths for one or more logical beams."""
        if isinstance(sequence_lengths, int):
            self.ensure_batch_size(index + 1)
            if sequence_lengths < 0:
                raise ValueError(f"seq_length must be non-negative, got {sequence_lengths}")
            self._beam_seq_lengths[index] = sequence_lengths
            self._beam_token_histories[index] = self._beam_token_histories[index][:sequence_lengths]
            self.layers[index].set_seq_length(sequence_lengths)
            return
        if sequence_lengths:
            self.ensure_batch_size(max(sequence_lengths) + 1)
        for beam_id, seq_len in sequence_lengths.items():
            if seq_len < 0:
                raise ValueError(f"seq_length must be non-negative, got {seq_len}")
            self._beam_seq_lengths[beam_id] = seq_len
            self._beam_token_histories[beam_id] = self._beam_token_histories[beam_id][:seq_len]
            self.layers[beam_id].set_seq_length(seq_len)

    def update_cache_position(self, cache_position: torch.Tensor, index: int = 0) -> None:
        """Update one logical beam length after its active qbruntime cache advances."""
        self.ensure_batch_size(index + 1)
        self._beam_seq_lengths[index] += int(cache_position.numel())
        self.layers[index].set_seq_length(self._beam_seq_lengths[index])

    def build_target_tokens(self, beam_index: int, input_ids: torch.Tensor) -> list[int]:
        """Return the target token history for one beam after appending new ids."""
        self.ensure_batch_size(beam_index + 1)
        new_tokens = self._tensor_to_token_list(input_ids)
        return [*self._beam_token_histories[beam_index], *new_tokens]

    def get_beam_tokens(self, beam_index: int) -> list[int]:
        """Return a copy of the stored token history for one logical beam."""
        self.ensure_batch_size(beam_index + 1)
        return list(self._beam_token_histories[beam_index])

    def get_beam_source_index(self, beam_index: int) -> int | None:
        """Return the source row identity stored for one logical beam."""
        self.ensure_batch_size(beam_index + 1)
        return self._beam_source_indices[beam_index]

    def set_beam_source_indices(self, source_indices: Sequence[int | None]) -> None:
        """Store source row identities for logical beams when they are first resolved."""
        self.ensure_batch_size(len(source_indices))
        for beam_index, source_index in enumerate(source_indices):
            self._beam_source_indices[beam_index] = None if source_index is None else int(source_index)

    def get_active_tokens(self) -> list[int]:
        """Return a copy of the token history represented by the active qbruntime cache."""
        return list(self._active_token_history)

    def get_active_source_index(self) -> int | None:
        """Return the source row represented by the active qbruntime cache."""
        return self._active_source_index

    def get_common_prefix_length(self, target_tokens: Sequence[int], source_index: int | None = None) -> int:
        """Return how many target tokens already match the active qbruntime cache."""
        if source_index is not None and self._active_source_index != int(source_index):
            return 0
        prefix_length = 0
        for active_token, target_token in zip(self._active_token_history, target_tokens):
            if active_token != target_token:
                break
            prefix_length += 1
        return prefix_length

    def commit_beam_tokens(self, beam_index: int, target_tokens: Sequence[int]) -> None:
        """Store the completed target history for one logical beam."""
        self.ensure_batch_size(beam_index + 1)
        token_history = [int(token) for token in target_tokens]
        self._beam_token_histories[beam_index] = token_history
        self._beam_seq_lengths[beam_index] = len(token_history)
        self.layers[beam_index].set_seq_length(len(token_history))

    def commit_active_tokens(self, target_tokens: Sequence[int], source_index: int | None = None) -> None:
        """Record which token history is now represented by active qbruntime cache memory."""
        self._active_token_history = [int(token) for token in target_tokens]
        self._active_source_index = None if source_index is None else int(source_index)
        self.layers[0].set_seq_length(len(self._active_token_history))

    def _tensor_to_token_list(self, input_ids: torch.Tensor) -> list[int]:
        """Convert a one-row token tensor to a flat Python token list."""
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor")
        if input_ids.ndim == 0:
            input_ids = input_ids.reshape(1)
        return [int(token) for token in input_ids.reshape(-1).detach().cpu().tolist()]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> "MobilintBeamCache":
        """Reorder application-level beam token histories in HF beam order."""
        beam_idx = self._validate_beam_indices(beam_idx)
        trace_enabled = is_whisper_beam_debug_trace_enabled()

        if trace_enabled:
            append_whisper_beam_debug_event(
                {
                    "event": "cache_reorder_before",
                    "beam_idx": [int(index) for index in beam_idx.cpu().tolist()],
                    "beam_token_histories": [list(tokens) for tokens in self._beam_token_histories],
                    "beam_source_indices": list(self._beam_source_indices),
                    "beam_seq_lengths": list(self._beam_seq_lengths),
                    "active_token_history": list(self._active_token_history),
                    "active_source_index": self._active_source_index,
                }
            )

        if torch.equal(beam_idx.cpu(), torch.arange(int(beam_idx.numel()), dtype=torch.long)):
            if trace_enabled:
                append_whisper_beam_debug_event(
                    {
                        "event": "cache_reorder_identity",
                        "beam_idx": [int(index) for index in beam_idx.cpu().tolist()],
                        "active_token_history": list(self._active_token_history),
                        "active_source_index": self._active_source_index,
                    }
                )
            return self

        old_token_histories = [list(tokens) for tokens in self._beam_token_histories]
        old_source_indices = list(self._beam_source_indices)
        old_seq_lengths = list(self._beam_seq_lengths)
        beam_indices = [int(index) for index in beam_idx.cpu().tolist()]
        self._beam_token_histories = [list(old_token_histories[index]) for index in beam_indices]
        self._beam_source_indices = [old_source_indices[index] for index in beam_indices]
        self._beam_seq_lengths = [old_seq_lengths[index] for index in beam_indices]
        for beam_id, seq_length in enumerate(self._beam_seq_lengths):
            self.layers[beam_id].set_seq_length(seq_length)
        if trace_enabled:
            append_whisper_beam_debug_event(
                {
                    "event": "cache_reorder_after",
                    "beam_idx": beam_indices,
                    "beam_token_histories": [list(tokens) for tokens in self._beam_token_histories],
                    "beam_source_indices": list(self._beam_source_indices),
                    "beam_seq_lengths": list(self._beam_seq_lengths),
                    "active_token_history": list(self._active_token_history),
                    "active_source_index": self._active_source_index,
                }
            )
        return self

    def _validate_beam_indices(self, beam_idx: torch.LongTensor) -> torch.LongTensor:
        """Validate beam indices before reordering token histories."""
        if not isinstance(beam_idx, torch.Tensor):
            raise TypeError("beam_idx must be a torch.Tensor")
        if beam_idx.ndim != 1:
            raise ValueError(f"beam_idx must be rank 1, got shape {tuple(beam_idx.shape)}")

        beam_idx = beam_idx.to(dtype=torch.long)
        self.ensure_batch_size(int(beam_idx.numel()))
        if beam_idx.numel() > 0 and (int(beam_idx.min()) < 0 or int(beam_idx.max()) >= int(beam_idx.numel())):
            raise ValueError(f"beam_idx contains out-of-range values for {int(beam_idx.numel())} beams")
        return beam_idx

    def copy(self) -> "MobilintBeamCache":
        """Return a copy preserving application-level beam token histories."""
        copied = self.__class__(list(self.mxq_models), batch_size=self.k_per_model)
        if len(copied.layers) < self.batch_size:
            copied.ensure_batch_size(self.batch_size)
        for i in range(len(self.layers)):
            copied.layers[i] = self.layers[i].copy()
        copied._beam_token_histories = [list(tokens) for tokens in self._beam_token_histories]
        copied._beam_source_indices = list(self._beam_source_indices)
        copied._active_token_history = list(self._active_token_history)
        copied._active_source_index = self._active_source_index
        copied._beam_seq_lengths = list(self._beam_seq_lengths)
        copied.rope_deltas = copy_rope_deltas(self)
        return copied


class MobilintWhisperCache(MobilintBeamCache):
    """Whisper cache using token-history beam replay."""

    def __init__(
        self,
        mxq_models: Union[List[qbruntime.Model], qbruntime.Model],
        batch_size: int = 1,
        *,
        n_slots: Optional[int] = None,
    ) -> None:
        super().__init__(mxq_models=mxq_models, batch_size=batch_size, n_slots=n_slots)
        self._encoder_source_count: int | None = None

    def reset(self) -> None:
        """Reset beam cache state and forget the current encoder source grouping."""
        super().reset()
        self._encoder_source_count = None

    def set_encoder_source_count(self, source_count: int) -> None:
        """Record the original audio batch size before Hugging Face beam expansion."""
        source_count = int(source_count)
        if source_count < 1:
            raise ValueError(f"source_count must be positive, got {source_count}")
        self._encoder_source_count = source_count

    def get_encoder_source_count(self) -> int | None:
        """Return the original encoder source count when it is known."""
        return self._encoder_source_count

    def copy(self) -> "MobilintWhisperCache":
        """Return a copy preserving Whisper encoder source grouping metadata."""
        copied = super().copy()
        assert isinstance(copied, MobilintWhisperCache)
        copied._encoder_source_count = self._encoder_source_count
        return copied


class MobilintDeepStackCache(MobilintCache):
    """Mobilint KV cache carrying Qwen3-VL deepstack decoder inputs.

    Qwen3-VL text MXQ uses token embeddings and a dense deepstack tensor as decoder inputs.
    This cache keeps the KV sequence length in ``MobilintCache`` while providing the matching
    deepstack chunk for each decoder invocation. Fake prefill stores only the requested sequence
    length and lazily serves zero deepstack chunks for synthetic decode TPS measurements.

    The constructor mirrors :class:`MobilintCache`'s dual signature so a multi-slot backend can
    build the deepstack cache through :func:`build_mobilint_cache_from_model` with
    ``per_model_batch=K`` while the legacy single-Model path keeps its ``batch_size=B`` keyword.
    """

    def __init__(
        self,
        mxq_models: Union[List[qbruntime.Model], qbruntime.Model],
        per_model_batch: int = 1,
        *,
        num_deepstack_layers: int = 0,
        hidden_size: int = 0,
        batch_size: Optional[int] = None,
        slots: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        super().__init__(
            mxq_models=mxq_models,
            per_model_batch=per_model_batch,
            batch_size=batch_size,
            slots=slots,
        )
        if num_deepstack_layers < 0:
            raise ValueError(f"num_deepstack_layers must be non-negative, got {num_deepstack_layers}")
        if hidden_size < 0:
            raise ValueError(f"hidden_size must be non-negative, got {hidden_size}")
        self.num_deepstack_layers = int(num_deepstack_layers)
        self.hidden_size = int(hidden_size)
        self._deepstack_tensor: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Reset KV sequence length and clear any per-call deepstack tensor."""
        for layer in self.layers:
            layer.reset()
        self._deepstack_tensor = None
        self.rope_deltas = None

    def fake_prefill(self, sequence_lengths: Union[dict[int, int], int], index: int = 0) -> None:
        """Mark the cache as fake-prefilled and clear real deepstack payloads."""
        super().fake_prefill(sequence_lengths, index=index)
        self._deepstack_tensor = None

    def set_deepstack_tensor(self, deepstack_tensor: torch.Tensor) -> None:
        """Set the deepstack tensor for the current decoder forward call.

        Args:
            deepstack_tensor: Dense tensor with shape ``(layers, seq_len, hidden_size)``.

        Raises:
            ValueError: If the tensor rank or configured dimensions do not match.
        """
        if deepstack_tensor.ndim != 3:
            raise ValueError(f"Expected deepstack tensor rank 3, got shape {tuple(deepstack_tensor.shape)}")
        if int(deepstack_tensor.shape[0]) != self.num_deepstack_layers:
            raise ValueError(
                "Deepstack layer count mismatch: "
                f"{int(deepstack_tensor.shape[0])} vs {self.num_deepstack_layers}"
            )
        if int(deepstack_tensor.shape[2]) != self.hidden_size:
            raise ValueError(
                f"Deepstack hidden size mismatch: {int(deepstack_tensor.shape[2])} vs {self.hidden_size}"
            )
        self._deepstack_tensor = deepstack_tensor

    def get_deepstack_chunk(
        self,
        start_index: int,
        end_index: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the deepstack input chunk for the current decoder chunk.

        Args:
            start_index: Inclusive local token offset in the current forward call.
            end_index: Exclusive local token offset in the current forward call.
            device: Device for a lazily-created fake chunk.
            dtype: Dtype for a lazily-created fake chunk.

        Returns:
            Tensor with shape ``(layers, end_index - start_index, hidden_size)``.
        """
        if start_index < 0 or end_index < start_index:
            raise ValueError(f"Invalid deepstack chunk range: {start_index}:{end_index}")

        if self._deepstack_tensor is not None and end_index <= int(self._deepstack_tensor.shape[1]):
            return self._deepstack_tensor[:, start_index:end_index, :].to(device=device, dtype=dtype)

        chunk_len = end_index - start_index
        return torch.zeros(
            (self.num_deepstack_layers, chunk_len, self.hidden_size),
            dtype=dtype,
            device=device,
        )

    def _empty_like(self, slots: Sequence[Tuple[int, int]]) -> "MobilintDeepStackCache":
        return MobilintDeepStackCache(
            list(self.mxq_models),
            per_model_batch=self.k_per_model,
            num_deepstack_layers=self.num_deepstack_layers,
            hidden_size=self.hidden_size,
            slots=slots,
        )

    def copy(self) -> "MobilintDeepStackCache":
        """Return a copy preserving KV state and the current deepstack tensor."""
        copied = MobilintDeepStackCache(
            list(self.mxq_models),
            per_model_batch=self.k_per_model,
            num_deepstack_layers=self.num_deepstack_layers,
            hidden_size=self.hidden_size,
        )
        copied._row_slots = list(self._row_slots)
        copied._explicit_slots = self._explicit_slots
        copied._batch_size = self._batch_size
        copied.layers = [layer.copy() for layer in self.layers]
        copied._deepstack_tensor = None if self._deepstack_tensor is None else self._deepstack_tensor.clone()
        copied.rope_deltas = copy_rope_deltas(self)
        copied.joined_rows = self.joined_rows
        return copied


def defer_eagle3_accept_tokens(cache: Any) -> None:
    """Move an EAGLE-3 cache's accepted-but-unfed tokens to ``unfed_tokens`` (see :class:`MobilintEagle3Cache`)."""
    accepted = getattr(cache, "accept_tokens", None)
    if accepted is None:
        return
    if accepted.shape[1] > 0:
        unfed = getattr(cache, "unfed_tokens", None)
        cache.unfed_tokens = accepted if unfed is None else torch.cat((unfed, accepted.to(unfed.device)), dim=1)
    cache.accept_tokens = None


def defer_eagle3_round_tokens(cache: Any) -> None:
    """Keep the tokens an EAGLE-3 cache left between rounds returned but never fed: the accepted tokens, then the
    next tree's root (``eagle3_decode_step`` returns it as ``next_token``). No-op without a pending tree.
    """
    next_tree = getattr(cache, "pending_draft_tokens", None)
    if next_tree is None:
        return
    defer_eagle3_accept_tokens(cache)
    cache.accept_tokens = next_tree[:, :1]
    defer_eagle3_accept_tokens(cache)


def take_eagle3_unfed_tokens(cache: Any) -> Optional[torch.LongTensor]:
    """Return and clear the tokens an EAGLE-3 cache still has to feed before the next input."""
    tokens = getattr(cache, "unfed_tokens", None)
    if tokens is not None:
        cache.unfed_tokens = None
    return tokens


def note_eagle3_fed_tokens(cache: Any, token_ids: Optional[torch.LongTensor]) -> None:
    """Record ids just fed to the base MXQ (after its length advanced), or ``None`` when they are unknown.

    ``fed_token_ids`` lets the next call tell a full-sequence input (it starts with exactly these ids) from a
    delta-only one. A gap in the record (ids fed somewhere that did not note them) makes it unknown.
    """
    if token_ids is None:
        cache.fed_token_ids = None
        return
    token_ids = token_ids.detach().to("cpu", dtype=torch.long)
    base_len = int(cache.get_base_seq_length())
    previous_len = base_len - int(token_ids.shape[1])
    known = eagle3_fed_tokens(cache, length=previous_len)
    if known is not None:
        cache.fed_token_ids = torch.cat((known, token_ids), dim=1)
    elif previous_len == 0:
        cache.fed_token_ids = token_ids.clone()
    else:
        cache.fed_token_ids = None


def eagle3_fed_tokens(cache: Any, length: Optional[int] = None) -> Optional[torch.LongTensor]:
    """The ids of the first ``length`` (default: all cached) base tokens, if the cache knows them."""
    length = int(cache.get_base_seq_length()) if length is None else int(length)
    if length == 0:
        return torch.empty((1, 0), dtype=torch.long)
    known = getattr(cache, "fed_token_ids", None)
    if known is None or known.shape[1] < length:
        return None
    return known[:, :length]


class MobilintEagle3Cache(Cache):
    """Mobilint cache for EAGLE-3 speculative decoding.

    This cache carries both base and draft MXQ cache states plus the mutable tree
    decoding state that upstream EAGLE-3 stores on the model instance.
    """

    def __init__(
        self,
        base_mxq_model: qbruntime.Model,
        draft_mxq_model: qbruntime.Model,
    ) -> None:
        self.base_mxq_model = base_mxq_model
        self.draft_mxq_model = draft_mxq_model
        self.base_layer = MobilintLayer(base_mxq_model, 0)
        self.draft_layer = MobilintLayer(draft_mxq_model, 0)
        self.draft_layer.speculative = True
        self.layers = [self.base_layer]
        self.layer_classes = MobilintLayer
        self.num_hidden_layers = 1
        self.cache_processor = None
        self.accept_tokens: Optional[torch.LongTensor] = None
        self.tree_mask: Optional[torch.Tensor] = None
        self.retrieve_indices: Optional[torch.LongTensor] = None
        self.tree_position_ids: Optional[torch.LongTensor] = None
        self.pending_draft_tokens: Optional[torch.LongTensor] = None
        # Tokens already returned to the caller whose KV the base and draft MXQs do not hold yet: generation stops
        # without feeding the last accepted tokens, and the next generate() or forward feeds them first.
        self.unfed_tokens: Optional[torch.LongTensor] = None
        # Ids of the tokens the base MXQ holds, when every feed recorded them (see note_eagle3_fed_tokens).
        self.fed_token_ids: Optional[torch.LongTensor] = torch.empty((1, 0), dtype=torch.long)

    @property
    def is_resident(self) -> bool:
        """Whether the base and draft MXQs hold this cache's KV."""
        return self.base_layer.is_resident and self.draft_layer.is_resident

    @property
    def host_nbytes(self) -> int:
        return self.base_layer.host_nbytes + self.draft_layer.host_nbytes

    def offload(self) -> None:
        """Save the base and draft KV to host memory and free both MXQ slots for other caches."""
        self.base_layer.offload()
        self.draft_layer.offload()

    def defer_accept_tokens(self) -> None:
        """Keep accepted tokens that were not fed for the next call instead of counting them as cached."""
        defer_eagle3_accept_tokens(self)

    def take_unfed_tokens(self) -> Optional[torch.LongTensor]:
        """Return and clear the tokens the next call must feed before its own input."""
        return take_eagle3_unfed_tokens(self)

    def get_seq_length(self, index: int = 0) -> int:
        del index
        return self.base_layer.get_seq_length()

    def get_base_seq_length(self) -> int:
        return self.base_layer.get_seq_length()

    def get_draft_seq_length(self) -> int:
        return self.draft_layer.get_seq_length()

    def set_seq_length(self, sequence_length: int, index: int = 0) -> None:
        del index
        self.set_base_seq_length(sequence_length)

    def set_base_seq_length(self, sequence_length: int) -> None:
        self.base_layer.set_seq_length(sequence_length)
        self.fed_token_ids = eagle3_fed_tokens(self)  # keep only the ids of the tokens still cached

    def set_draft_seq_length(self, sequence_length: int) -> None:
        self.draft_layer.set_seq_length(sequence_length)

    def sync_draft_seq_length_to_base(self) -> None:
        """Align the draft cache length with the committed base cache length."""
        self.draft_layer.set_seq_length(self.get_base_seq_length())

    def update_cache_position(self, cache_position: torch.Tensor, index: int = 0) -> None:
        del index
        self.base_layer.update_cache_position(cache_position)

    def update_base_seen_tokens(self, num_new_seen_tokens: int) -> None:
        self.base_layer.update_seen_tokens(num_new_seen_tokens)

    def update_draft_seen_tokens(self, num_new_seen_tokens: int) -> None:
        self.draft_layer.update_seen_tokens(num_new_seen_tokens)

    def fake_prefill(self, sequence_length: int, index: int = 0) -> None:
        del index
        self.base_layer.fake_prefill(sequence_length)
        self.draft_layer.fake_prefill(sequence_length)
        self.clear_tree_state()
        self.unfed_tokens = None
        self.fed_token_ids = None

    def clear_tree_state(self) -> None:
        """Drop speculative decoding metadata while preserving KV cache state."""
        self.accept_tokens = None
        self.tree_mask = None
        self.retrieve_indices = None
        self.tree_position_ids = None
        self.pending_draft_tokens = None

    def reset(self) -> None:
        self.base_layer.reset()
        self.draft_layer.reset()
        self.clear_tree_state()
        self.unfed_tokens = None
        self._snapshot_unfed_tokens = None
        self.fed_token_ids = torch.empty((1, 0), dtype=torch.long)

    def bind_base(self, base_mxq_model: Any) -> None:
        """Make ``base_mxq_model`` hold this cache's base KV before a base infer (moving it there if needed)."""
        residency.acquire(self.base_layer, base_mxq_model, 0)
        self.base_mxq_model = base_mxq_model

    def bind_draft(self, draft_mxq_model: Any) -> None:
        """Make ``draft_mxq_model`` hold this cache's draft KV before a draft infer (moving it there if needed)."""
        residency.acquire(self.draft_layer, draft_mxq_model, 0)
        self.draft_mxq_model = draft_mxq_model

    # A snapshot is the base/draft KV plus the tokens still to be fed: their KV is not in the snapshot, so restoring
    # without them would let a delta-only continuation skip tokens the caller already received.
    _UNFED_TOKENS_FILE = "unfed_tokens.json"

    def dump_cache_memory(self) -> None:
        self.base_layer.dump_cache_memory()
        self.draft_layer.dump_cache_memory()
        unfed = getattr(self, "unfed_tokens", None)
        self._snapshot_unfed_tokens = None if unfed is None else unfed.clone()
        fed = eagle3_fed_tokens(self)
        self._snapshot_fed_token_ids = None if fed is None else fed.clone()

    def load_cache_memory(self) -> None:
        self.base_layer.load_cache_memory()
        self.draft_layer.load_cache_memory()
        snapshot = getattr(self, "_snapshot_unfed_tokens", None)
        self.unfed_tokens = None if snapshot is None else snapshot.clone()
        fed = getattr(self, "_snapshot_fed_token_ids", None)
        self.fed_token_ids = None if fed is None else fed.clone()

    def dump_cache_memory_to(self, cache_dir: str, index: int = 0) -> None:
        del index
        base_dir = Path(cache_dir) / "base"
        draft_dir = Path(cache_dir) / "draft"
        base_dir.mkdir(parents=True, exist_ok=True)
        draft_dir.mkdir(parents=True, exist_ok=True)
        self.base_layer.dump_cache_memory_to(str(base_dir))
        self.draft_layer.dump_cache_memory_to(str(draft_dir))
        unfed = getattr(self, "unfed_tokens", None)
        unfed_path = Path(cache_dir) / self._UNFED_TOKENS_FILE
        fed = eagle3_fed_tokens(self)
        unfed_path.write_text(
            json.dumps(
                {
                    "unfed_tokens": None if unfed is None else unfed.tolist(),
                    "fed_token_ids": None if fed is None else fed.tolist(),
                }
            ),
            encoding="utf-8",
        )

    def load_cache_memory_from(self, cache_dir: str, index: int = 0) -> None:
        del index
        self.reset()
        self.base_layer.load_cache_memory_from(str(Path(cache_dir) / "base"))
        self.draft_layer.load_cache_memory_from(str(Path(cache_dir) / "draft"))
        unfed_path = Path(cache_dir) / self._UNFED_TOKENS_FILE
        self.fed_token_ids = None  # unknown unless the snapshot recorded them
        if unfed_path.exists():
            state = json.loads(unfed_path.read_text(encoding="utf-8"))
            if state.get("unfed_tokens") is not None:
                self.unfed_tokens = torch.tensor(state["unfed_tokens"], dtype=torch.long)
            if state.get("fed_token_ids") is not None:
                self.fed_token_ids = torch.tensor(state["fed_token_ids"], dtype=torch.long)

    def copy(self) -> "MobilintEagle3Cache":
        """Return a snapshot of the committed KV state and the tokens still to be fed.

        Speculative tree metadata is intentionally not copied because it is
        per-generation-call transient state and should be reconstructed by
        ``initialize_tree``. Resident layers save their KV first.
        """
        copied = MobilintEagle3Cache(self.base_mxq_model, self.draft_mxq_model)
        copied.base_layer = self.base_layer.copy()
        copied.draft_layer = self.draft_layer.copy()
        copied.layers = [copied.base_layer]
        copied.clear_tree_state()
        unfed = getattr(self, "unfed_tokens", None)
        copied.unfed_tokens = None if unfed is None else unfed.clone()
        snapshot = getattr(self, "_snapshot_unfed_tokens", None)
        copied._snapshot_unfed_tokens = None if snapshot is None else snapshot.clone()
        fed = getattr(self, "fed_token_ids", None)
        copied.fed_token_ids = None if fed is None else fed.clone()
        fed_snapshot = getattr(self, "_snapshot_fed_token_ids", None)
        copied._snapshot_fed_token_ids = None if fed_snapshot is None else fed_snapshot.clone()
        return copied


class MobilintMixtureOfExpertsCache(Cache):
    """Mobilint cache for Mixture-of-Experts releases.

    Every decoder layer of a MoE release is its own ``shared`` MXQ holding its own KV cache. One forward step feeds
    the same tokens through every shared MXQ, so their caches always advance in lockstep and one sequence length
    describes all of them. The cache keeps one :class:`MobilintLayer` per shared MXQ so KV dumps and loads reach
    every layer; ``layers[0]`` is the authoritative cursor, matching the single-entry ``MobilintCache`` API
    (callers that rewind with ``cache.layers[0]._seen_tokens = n`` keep working).

    Only ``batch_size == 1`` is supported: each shared MXQ hosts a single cache entry.

    The shared MXQs hold one sequence at a time. Like :class:`MobilintCache`, the cache is portable: before every
    forward the model binds all shared layers (see :mod:`transformers_mblt.utils.cache_residency`), so a cache that
    another sequence displaced is saved and restored instead of reset, and a cache can move to another instance of
    the same release. ``owner`` is accepted for compatibility, unused, and held weakly so a cache that moved away
    never keeps its source backend alive.
    """

    def __init__(self, shared_mxq_models: Sequence[qbruntime.Model], owner: Any = None) -> None:
        if not shared_mxq_models:
            raise ValueError("shared_mxq_models must contain at least one Model")
        self.owner = owner
        self.shared_mxq_models: List[qbruntime.Model] = list(shared_mxq_models)
        self.shared_layers: list[MobilintLayer] = [MobilintLayer(model, 0) for model in self.shared_mxq_models]
        self._tag_layers()
        self.layers = [self.shared_layers[0]]
        self.layer_classes = MobilintLayer
        self.num_hidden_layers = 1
        self.cache_processor = None

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def max_batch_size(self) -> int:
        return 1

    @property
    def owner(self) -> Any:
        return self._owner_ref() if self._owner_ref is not None else None

    @owner.setter
    def owner(self, value: Any) -> None:
        try:
            self._owner_ref = weakref.ref(value) if value is not None else None
        except TypeError:  # objects without __weakref__ are dropped rather than kept alive
            self._owner_ref = None

    def _tag_layers(self) -> None:
        # Lets the backend's ``kv_owner`` map a slot's owner layer back to its cache.
        for layer in self.shared_layers:
            layer._owner_cache = weakref.ref(self)

    @property
    def is_resident(self) -> bool:
        return all(layer.is_resident for layer in self.shared_layers)

    @property
    def host_nbytes(self) -> int:
        return sum(layer.host_nbytes for layer in self.shared_layers)

    def offload(self) -> None:
        """Save every shared layer's KV to host memory and free the shared MXQs for other caches."""
        self._sync_shared_layers()
        for layer in self.shared_layers:
            layer.offload()

    def bind_shared(self, shared_models: Sequence[Any]) -> None:
        """Make each shared MXQ hold this cache's KV before a forward, moving the cache to ``shared_models``."""
        from .cache_residency import bind_cache_layers

        shared_models = list(shared_models)
        if len(shared_models) != len(self.shared_layers):
            raise ValueError(
                f"Cannot bind a cache of {len(self.shared_layers)} shared layers to a model with "
                f"{len(shared_models)} shared MXQs."
            )
        self._sync_shared_layers()
        bind_cache_layers((layer, model, 0) for layer, model in zip(self.shared_layers, shared_models))
        self.shared_mxq_models = shared_models

    def _sync_shared_layers(self) -> None:
        """Propagate the ``layers[0]`` cursor to every shared layer."""
        seq_length = self.shared_layers[0].get_seq_length()
        for layer in self.shared_layers[1:]:
            layer.set_seq_length(seq_length)

    def get_seq_length(self, index: int = 0) -> int:
        del index
        return self.shared_layers[0].get_seq_length()

    def set_seq_length(self, sequence_lengths: Union[dict[int, int], int], index: int = 0) -> None:
        del index
        if isinstance(sequence_lengths, dict):
            if list(sequence_lengths) != [0]:
                raise ValueError(
                    f"MobilintMixtureOfExpertsCache has a single cache entry; got cache ids {list(sequence_lengths)}"
                )
            sequence_lengths = sequence_lengths[0]
        for layer in self.shared_layers:
            layer.set_seq_length(int(sequence_lengths))

    def fake_prefill(self, sequence_lengths: Union[dict[int, int], int], index: int = 0) -> None:
        """Mark every shared layer as prefilled without restoring KV cache memory (decode TPS benchmarks).

        The KV contents are whatever the shared MXQs last held, so the decoded tokens are meaningless, but each step
        attends over ``sequence_lengths`` positions and the measured decode cost is the real one. The layers are
        marked as holding no real KV, so binding them never dumps or loads.
        """
        for layer in self.shared_layers:
            layer.fake_prefill(0)
        self.set_seq_length(sequence_lengths, index=index)

    def update_cache_position(self, cache_position: torch.Tensor, index: int = 0) -> None:
        del index
        self._sync_shared_layers()
        for layer in self.shared_layers:
            layer.update_cache_position(cache_position)

    def update_seen_tokens(self, num_new_seen_tokens: int, index: int = 0) -> None:
        del index
        self._sync_shared_layers()
        for layer in self.shared_layers:
            layer.update_seen_tokens(num_new_seen_tokens)

    def reset(self) -> None:
        for layer in self.shared_layers:
            layer.reset()

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            raise ValueError(f"max_length must be non-negative, got {max_length}")
        self.set_seq_length(min(self.get_seq_length(), int(max_length)))

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        raise NotImplementedError("MobilintMixtureOfExpertsCache does not support beam search (batch_size=1).")

    def dump_cache_memory(self) -> None:
        self._sync_shared_layers()
        for layer in self.shared_layers:
            layer.dump_cache_memory()

    def load_cache_memory(self) -> None:
        for layer in self.shared_layers:
            layer.load_cache_memory()

    def dump_cache_memory_to(self, cache_dir: str, index: int = 0) -> None:
        del index
        self._sync_shared_layers()
        for layer_idx, layer in enumerate(self.shared_layers):
            layer_dir = Path(cache_dir) / f"layer_{layer_idx}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            layer.dump_cache_memory_to(str(layer_dir))

    def load_cache_memory_from(self, cache_dir: str, index: int = 0) -> None:
        del index
        for layer_idx, layer in enumerate(self.shared_layers):
            layer.load_cache_memory_from(str(Path(cache_dir) / f"layer_{layer_idx}"))

    def copy(self) -> "MobilintMixtureOfExpertsCache":
        """Return a snapshot: resident layers save their KV first, and the copy is loaded on its first forward."""
        self._sync_shared_layers()
        copied = MobilintMixtureOfExpertsCache(self.shared_mxq_models, owner=self.owner)
        copied.shared_layers = [layer.copy() for layer in self.shared_layers]
        copied._tag_layers()
        copied.layers = [copied.shared_layers[0]]
        return copied

    def __repr__(self) -> str:
        return f"MobilintMixtureOfExpertsCache(num_layers={len(self.shared_layers)}, seq_length={self.get_seq_length()})"
