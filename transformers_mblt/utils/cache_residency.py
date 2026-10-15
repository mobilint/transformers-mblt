"""Track which cache owns each NPU KV slot, so caches can move between slots and model instances.

A qbruntime ``Model`` keeps one KV memory region per ``cache_id``: a *slot*. A :class:`MobilintLayer` is one
sequence's KV state for one slot. Several layers can target the same slot (two caches built for the same
model, a cache and its ``copy()``, a cache handed from one model instance to another), but the slot holds only
one of them at a time.

One process-wide table records which layer each slot holds: ``Model -> {cache_id: layer}``, keyed weakly by the
Model and holding the layer weakly. Every operation that puts a layer's KV on a slot goes through it: a model
binding a cache before an infer, an explicit ``load_cache_memory()``, a file restore. Putting a layer on a slot
(:func:`acquire`):

1. saves the layer's KV from the slot it currently owns, if it owns another one (a move between slots or model
   instances);
2. evicts the slot's current owner, dumping its KV into that layer's host copy;
3. loads the incoming layer's host copy into the slot;
4. records the incoming layer as the owner.

Acquiring a slot the layer already owns costs nothing, so a cache that keeps decoding on its own slot never dumps.
A caller that no longer needs a layer's KV frees its slot with :func:`discard` instead, which saves nothing.
KV only moves between compatible MXQs: see :func:`mxq_kv_signature`.

A slot is live while its Model is alive and not disposed (qbruntime's ``Model.dispose()`` drops ``_acc``). The KV
of a disposed slot is gone: its owner can only be restored from a host copy saved earlier.

Beam caches (Whisper, Qwen3-ASR) put a :class:`~transformers_mblt.utils.cache_utils.MobilintReplaySlot` in the
table instead of a layer. It saves and loads nothing: losing or regaining the slot makes the beam cache replay its
token history from the start.

Layers that the table never tracked keep the pre-residency behavior of :class:`MobilintLayer`
(``dump_cache_memory`` reads the slot directly), so code paths that do not bind yet are unaffected.

Not thread-safe: bind from the thread that drives the model, before fanning infers out to worker threads.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Iterable, Optional

if TYPE_CHECKING:
    from .cache_utils import MobilintLayer

# Model -> {cache_id: weakref(layer)}. Models that cannot be weakly referenced (test doubles) are not tracked.
_SLOT_OWNERS: "weakref.WeakKeyDictionary[Any, dict[int, weakref.ref[MobilintLayer]]]" = weakref.WeakKeyDictionary()
_SIGNATURES: "weakref.WeakKeyDictionary[Any, Optional[tuple]]" = weakref.WeakKeyDictionary()
_SIGNATURE_QUERIES = ("get_core_mode", "get_cache_infos", "get_model_input_shape", "get_input_buffer_info")


class MobilintCacheStateError(RuntimeError):
    """A cache's KV cannot be restored: it was never saved, or it belongs to an incompatible MXQ."""


def mxq_kv_signature(mxq_model: Any) -> Optional[tuple]:
    """Return what must match for one MXQ's KV dump to load into another.

    Built from the compiled KV buffers (name, size, batches, type and ``layer_hash`` of each), the core mode,
    the input shapes and the KV capacity. ``Model.get_identifier()`` is a per-handle counter and
    ``get_model_path()`` can be a transient ``/proc/self/fd`` path, so neither is used.

    Returns ``None`` only for objects that do not expose these queries at all (test doubles). When a query fails
    on a handle that does expose them, raises :class:`MobilintCacheStateError`: a move must not proceed on an
    unverified KV layout.
    """
    try:
        return _SIGNATURES[mxq_model]
    except (KeyError, TypeError):
        pass
    if not all(callable(getattr(mxq_model, name, None)) for name in _SIGNATURE_QUERIES):
        return None
    try:
        signature: Optional[tuple] = (
            str(mxq_model.get_core_mode()),
            tuple(
                (info.name, int(info.size), int(info.num_batches), str(info.cache_type), info.layer_hash)
                for info in mxq_model.get_cache_infos()
            ),
            tuple(tuple(shape) for shape in mxq_model.get_model_input_shape()),
            int(mxq_model.get_input_buffer_info()[0].max_cache_size),
        )
    except Exception as exc:
        raise MobilintCacheStateError(
            "Cannot read the MXQ's KV layout to check that a cache can move to it; refusing to load unverified KV."
        ) from exc
    try:
        _SIGNATURES[mxq_model] = signature
    except TypeError:
        pass
    return signature


# ----------------------------------------------------------------------
# Owner table
# ----------------------------------------------------------------------


def _is_disposed(mxq_model: Any) -> bool:
    """Whether a qbruntime ``Model`` was disposed: ``dispose()`` drops its accelerator (``_acc``)."""
    return hasattr(mxq_model, "_acc") and getattr(mxq_model, "_acc") is None


def _slots(mxq_model: Any, *, create: bool = False) -> Optional[dict[int, "weakref.ref[MobilintLayer]"]]:
    """The live slot map of ``mxq_model``; ``None`` if it is untracked, disposed, or cannot be tracked."""
    try:
        slots = _SLOT_OWNERS.get(mxq_model)
    except TypeError:
        return None
    if slots is not None and _is_disposed(mxq_model):
        # The KV memory is gone: forget the owners. Their KV can only come back from a saved host copy.
        del _SLOT_OWNERS[mxq_model]
        return None
    if slots is None and create and not _is_disposed(mxq_model):
        slots = {}
        try:
            _SLOT_OWNERS[mxq_model] = slots
        except TypeError:
            return None
    return slots


def owner(mxq_model: Any, cache_id: int) -> Optional["MobilintLayer"]:
    """Return the layer whose KV the slot holds, or ``None``."""
    slots = _slots(mxq_model)
    if slots is None:
        return None
    ref = slots.get(int(cache_id))
    layer = ref() if ref is not None else None
    if layer is None or layer.mxq_model is not mxq_model or int(layer.cache_id) != int(cache_id):
        slots.pop(int(cache_id), None)
        return None
    return layer


def is_resident(layer: "MobilintLayer") -> bool:
    """Whether the layer's slot holds the layer's KV."""
    return owner(layer.mxq_model, layer.cache_id) is layer


def claim(layer: "MobilintLayer") -> None:
    """Record ``layer`` as the owner of its slot, whose KV it just wrote."""
    layer._tracked = True
    slots = _slots(layer.mxq_model, create=True)
    if slots is not None:
        slots[int(layer.cache_id)] = weakref.ref(layer)


def evict(mxq_model: Any, cache_id: int) -> None:
    """Save the slot's owner to its host copy and leave the slot unowned (something is about to overwrite it)."""
    layer = owner(mxq_model, cache_id)
    if layer is None:
        return
    layer._save_from_slot()
    slots = _slots(mxq_model)
    if slots is not None:
        slots.pop(int(cache_id), None)


def acquire(layer: "MobilintLayer", mxq_model: Any, cache_id: int, *, reload: bool = False) -> None:
    """Make ``(mxq_model, cache_id)`` hold ``layer``'s KV.

    ``reload=True`` loads the host copy even when the layer already owns the slot; this is what an explicit
    :meth:`MobilintLayer.load_cache_memory` asks for.
    """
    cache_id = int(cache_id)
    same_slot = layer.mxq_model is mxq_model and int(layer.cache_id) == cache_id
    if same_slot and is_resident(layer):
        if reload:
            layer._load_into_slot()
        return
    if not same_slot:
        if is_resident(layer):
            evict(layer.mxq_model, layer.cache_id)  # leaving another slot or model instance: save it there
        layer._retarget(mxq_model, cache_id)
    evict(mxq_model, cache_id)
    layer._load_into_slot()
    claim(layer)


def release(layer: "MobilintLayer") -> None:
    """Save ``layer``'s KV to its host copy and free its slot, if it holds one."""
    if is_resident(layer):
        evict(layer.mxq_model, layer.cache_id)


def discard(layer: "MobilintLayer") -> None:
    """Free ``layer``'s slot without saving its KV, if it holds one: the next layer bound there overwrites it.

    For a caller that no longer needs the KV (a prefix-cache entry it drops). The layer itself is not changed, so
    callers reset it too: :meth:`MobilintLayer.discard` and the caches' ``discard()`` do both. A slot the layer does
    not hold keeps its owner.
    """
    if not is_resident(layer):
        return
    slots = _slots(layer.mxq_model)
    if slots is not None:
        slots.pop(int(layer.cache_id), None)


def bind_cache_layers(bindings: Iterable[tuple["MobilintLayer", Any, int]]) -> None:
    """Make each ``(layer, mxq_model, cache_id)`` slot hold its layer's KV before an infer."""
    for layer, mxq_model, cache_id in bindings:
        acquire(layer, mxq_model, cache_id)


def bind_cacheless(slots: Iterable[tuple[Any, int]]) -> None:
    """Prepare slots for a forward without a cache: save whatever cache they hold, then leave them unowned."""
    for mxq_model, cache_id in slots:
        evict(mxq_model, cache_id)


__all__ = [
    "MobilintCacheStateError",
    "acquire",
    "bind_cache_layers",
    "bind_cacheless",
    "claim",
    "discard",
    "evict",
    "is_resident",
    "mxq_kv_signature",
    "owner",
    "release",
]
