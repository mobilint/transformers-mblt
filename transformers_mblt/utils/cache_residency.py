"""Track which cache owns each NPU KV slot, so caches can move between slots and model instances.

A qbruntime ``Model`` keeps one KV memory region per ``cache_id``: a *slot*. A :class:`MobilintLayer` is one
sequence's KV state for one slot. Several layers can target the same slot (two caches built for the same
model, a cache and its ``copy()``, a cache handed from one model instance to another), but the slot holds only
one of them at a time.

:class:`MobilintKVResidency` records, per backend, which layer each slot currently holds. Before every infer, the
model calls :func:`bind_cache_layers` with the layers that infer will read and write. Binding a layer that is
not resident:

1. evicts the layer that holds the slot, dumping its KV into that layer's host ``buffer``;
2. loads the incoming layer's host ``buffer`` into the slot;
3. records the incoming layer as the owner.

Binding the layer that is already resident costs nothing, so a cache that keeps decoding on its own slot never
dumps. A layer resident on another backend or slot is evicted there first, which is how a cache moves between
model instances. A layer's KV is only moved between compatible MXQs: see :func:`mxq_kv_signature`.

Layers that were never bound keep the pre-residency behavior of :class:`MobilintLayer` (``dump_cache_memory``
and ``load_cache_memory`` talk to the MXQ directly), so code paths that do not bind yet are unaffected.

Not thread-safe: bind from the thread that drives the model, before fanning infers out to worker threads.
"""

from __future__ import annotations

import logging
import weakref
from typing import TYPE_CHECKING, Any, Iterable, Optional

if TYPE_CHECKING:
    from .cache_utils import MobilintLayer

logger = logging.getLogger(__name__)

_RESIDENCY_ATTR = "_mblt_kv_residency"
_SIGNATURES: "weakref.WeakKeyDictionary[Any, Optional[tuple]]" = weakref.WeakKeyDictionary()


class MobilintCacheStateError(RuntimeError):
    """A cache's KV cannot be restored: it was never saved, or it belongs to an incompatible MXQ."""


def mxq_kv_signature(mxq_model: Any) -> Optional[tuple]:
    """Return what must match for one MXQ's KV dump to load into another, or ``None`` if unknown.

    Built from the compiled KV buffers (name, size, batches, type and ``layer_hash`` of each), the core mode,
    the input shapes and the KV capacity. ``Model.get_identifier()`` is a per-handle counter and
    ``get_model_path()`` can be a transient ``/proc/self/fd`` path, so neither is used. Handles that do not
    expose these queries (test doubles) return ``None``, which never blocks a move.
    """
    try:
        return _SIGNATURES[mxq_model]
    except (KeyError, TypeError):
        pass
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
    except Exception:
        signature = None
    try:
        _SIGNATURES[mxq_model] = signature
    except TypeError:
        pass
    return signature


class MobilintKVResidency:
    """Owner table of one NPU backend's KV slots: ``(Model, cache_id) -> MobilintLayer``.

    Entries hold weak references to the Model and the layer. A slot whose layer was garbage-collected is free;
    a slot whose Model is no longer one of the backend's live handles (the backend was disposed and created
    again) is dropped without a dump, because that KV memory no longer exists.
    """

    def __init__(self) -> None:
        self._owners: dict[tuple[int, int], tuple["weakref.ref[Any]", int, "weakref.ref[MobilintLayer]"]] = {}

    # ------------------------------------------------------------------
    # Owner table
    # ------------------------------------------------------------------

    def owner(self, mxq_model: Any, cache_id: int) -> Optional["MobilintLayer"]:
        """Return the layer whose KV the slot holds, or ``None``."""
        entry = self._owners.get((id(mxq_model), int(cache_id)))
        if entry is None:
            return None
        model_ref, _, layer_ref = entry
        layer = layer_ref()
        if model_ref() is not mxq_model or layer is None:
            del self._owners[(id(mxq_model), int(cache_id))]
            return None
        return layer

    def _record(self, mxq_model: Any, cache_id: int, layer: Optional["MobilintLayer"]) -> None:
        key = (id(mxq_model), int(cache_id))
        if layer is None:
            self._owners.pop(key, None)
            return
        self._owners[key] = (weakref.ref(mxq_model), int(cache_id), weakref.ref(layer))
        layer._resident_at = weakref.ref(self)
        layer._residency_hint = weakref.ref(self)

    def is_resident(self, layer: "MobilintLayer") -> bool:
        return self.owner(layer.mxq_model, layer.cache_id) is layer

    def forget_stale_models(self, live_models: Iterable[Any]) -> None:
        """Drop entries for Models that are not among ``live_models`` (a disposed launch)."""
        live = {id(model) for model in live_models}
        for key in list(self._owners):
            model_ref, _, layer_ref = self._owners[key]
            model = model_ref()
            if model is None or id(model) not in live:
                layer = layer_ref()
                del self._owners[key]
                if layer is not None and layer._resident_at is not None and layer._resident_at() is self:
                    layer._resident_at = None

    def clear(self) -> None:
        for _, _, layer_ref in self._owners.values():
            layer = layer_ref()
            if layer is not None and layer._resident_at is not None and layer._resident_at() is self:
                layer._resident_at = None
        self._owners.clear()

    # ------------------------------------------------------------------
    # Moving KV in and out of slots
    # ------------------------------------------------------------------

    def evict(self, mxq_model: Any, cache_id: int) -> None:
        """Save the slot's current owner to its host buffer and free the slot."""
        layer = self.owner(mxq_model, cache_id)
        if layer is None:
            return
        layer._save_from_slot()
        self._record(mxq_model, cache_id, None)
        layer._resident_at = None

    def evict_layer(self, layer: "MobilintLayer") -> None:
        """Evict ``layer`` from its slot if it is resident here."""
        if self.is_resident(layer):
            self.evict(layer.mxq_model, layer.cache_id)
        elif layer._resident_at is not None and layer._resident_at() is self:
            layer._resident_at = None

    def acquire(self, layer: "MobilintLayer", mxq_model: Any, cache_id: int, *, reload: bool = False) -> None:
        """Make ``(mxq_model, cache_id)`` hold ``layer``'s KV.

        ``reload=True`` loads the host buffer even when the layer already owns the slot; this is what an explicit
        :meth:`MobilintLayer.load_cache_memory` asks for.
        """
        cache_id = int(cache_id)
        same_slot = layer.mxq_model is mxq_model and int(layer.cache_id) == cache_id
        if same_slot and self.is_resident(layer):
            if reload:
                layer._load_into_slot()
            return

        # Resident somewhere else (another slot here, or another backend): save it there first.
        previous = layer._resident_at() if layer._resident_at is not None else None
        if previous is not None:
            previous.evict_layer(layer)

        if not same_slot:
            layer._retarget(mxq_model, cache_id)

        self.evict(mxq_model, cache_id)
        layer._load_into_slot()
        self._record(mxq_model, cache_id, layer)

    def release_slot(self, mxq_model: Any, cache_id: int) -> None:
        """Hand a slot to a forward that runs without a cache: save the owner, leave the slot unowned."""
        self.evict(mxq_model, cache_id)


def residency_of(backend: Any) -> MobilintKVResidency:
    """Return the backend's owner table, creating it on first use and dropping slots of disposed Models."""
    residency = getattr(backend, _RESIDENCY_ATTR, None)
    if residency is None:
        residency = MobilintKVResidency()
        setattr(backend, _RESIDENCY_ATTR, residency)
    live_models = getattr(backend, "mxq_models", None)
    if live_models is not None:
        residency.forget_stale_models(live_models)
    return residency


def bind_cache_layers(backend: Any, bindings: Iterable[tuple["MobilintLayer", Any, int]]) -> None:
    """Make each ``(layer, mxq_model, cache_id)`` slot hold its layer's KV before an infer."""
    residency = residency_of(backend)
    for layer, mxq_model, cache_id in bindings:
        residency.acquire(layer, mxq_model, cache_id)


def bind_cacheless(backend: Any, slots: Iterable[tuple[Any, int]]) -> None:
    """Prepare slots for a forward without a cache: save whatever cache they hold, then leave them unowned."""
    residency = residency_of(backend)
    for mxq_model, cache_id in slots:
        residency.release_slot(mxq_model, cache_id)


__all__ = [
    "MobilintCacheStateError",
    "MobilintKVResidency",
    "bind_cache_layers",
    "bind_cacheless",
    "mxq_kv_signature",
    "residency_of",
]
