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
# Every live Model a residency has seen -> that residency, so an explicit load on a cache that was never bound still
# goes through the owner table of the backend that runs the Model.
_MODEL_RESIDENCY: "weakref.WeakKeyDictionary[Any, weakref.ref[MobilintKVResidency]]" = weakref.WeakKeyDictionary()


class MobilintCacheStateError(RuntimeError):
    """A cache's KV cannot be restored: it was never saved, or it belongs to an incompatible MXQ."""


_SIGNATURE_QUERIES = ("get_core_mode", "get_cache_infos", "get_model_input_shape", "get_input_buffer_info")


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


class MobilintKVResidency:
    """Owner table of one NPU backend's KV slots: ``(Model, cache_id) -> MobilintLayer``.

    Entries hold weak references to the Model and the layer. A slot whose layer was garbage-collected is free;
    a slot whose Model is no longer one of the backend's live handles (the backend was disposed and created
    again) is dropped without a dump, because that KV memory no longer exists.
    """

    def __init__(self, backend: Any = None) -> None:
        self._owners: dict[tuple[int, int], tuple["weakref.ref[Any]", int, "weakref.ref[MobilintLayer]"]] = {}
        self._backend_ref: Optional[Any] = None
        self._live_ids: Optional[set[int]] = None  # ids of the backend's live Models at the last refresh
        if backend is not None:
            try:
                self._backend_ref = weakref.ref(backend)
            except TypeError:  # objects without __weakref__ (test doubles); the table lives on the backend anyway
                self._backend_ref = lambda: backend

    def refresh(self) -> None:
        """Drop slots whose Model is no longer one of the backend's live handles (or the backend is gone)."""
        if self._backend_ref is None:
            return
        backend = self._backend_ref()
        live_models = kv_slot_models(backend) if backend is not None else []
        if live_models is None:
            return
        self.forget_stale_models(live_models)
        self._live_ids = {id(model) for model in live_models}
        for model in live_models:
            try:
                if _MODEL_RESIDENCY.get(model) is None or _MODEL_RESIDENCY[model]() is not self:
                    _MODEL_RESIDENCY[model] = weakref.ref(self)
            except TypeError:  # Models without __weakref__ (test doubles)
                pass

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
        # Refresh first: a slot on a disposed backend or a replaced Model must not count as live.
        self.refresh()
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
        """Evict ``layer`` from its slot if it is resident here.

        Runs on the residency the layer is leaving, which may belong to another backend: refresh first so a slot
        on a disposed backend is dropped (its KV is gone) instead of dumped from a dead Model.
        """
        self.refresh()
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
        if same_slot and self.owner(mxq_model, cache_id) is layer:  # callers refreshed already
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


def kv_slot_models(backend: Any) -> Optional[list[Any]]:
    """Return the backend's live Models that hold KV, or ``None`` when the backend does not say.

    A backend lists them as ``kv_slot_models`` (the MoE backend: one shared MXQ per layer) or, for the regular
    backend, as ``mxq_models``.
    """
    models = getattr(backend, "kv_slot_models", None)
    if models is None:
        models = getattr(backend, "mxq_models", None)
    return None if models is None else list(models)


def residency_for_model(mxq_model: Any) -> Optional[MobilintKVResidency]:
    """Return the owner table of the backend running ``mxq_model``, if a model has bound a cache on it."""
    try:
        ref = _MODEL_RESIDENCY.get(mxq_model)
    except TypeError:
        return None
    residency = ref() if ref is not None else None
    if residency is None:
        return None
    residency.refresh()
    if residency._live_ids is not None and id(mxq_model) not in residency._live_ids:
        return None  # the Model belongs to a disposed launch
    return residency


def residency_of(backend: Any) -> MobilintKVResidency:
    """Return the backend's owner table, creating it on first use and dropping slots of disposed Models."""
    residency = getattr(backend, _RESIDENCY_ATTR, None)
    if residency is None:
        residency = MobilintKVResidency(backend)
        setattr(backend, _RESIDENCY_ATTR, residency)
    residency.refresh()
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
    "kv_slot_models",
    "mxq_kv_signature",
    "residency_for_model",
    "residency_of",
]
