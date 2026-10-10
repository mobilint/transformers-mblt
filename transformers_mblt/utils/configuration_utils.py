import inspect
from inspect import Parameter, Signature
from typing import Any, Callable, Optional, TypeVar, Union

from transformers.configuration_utils import PretrainedConfig

try:
    from transformers.configuration_utils import SpecificPretrainedConfigType
except ImportError:
    try:
        from transformers.configuration_utils import SpecificPreTrainedConfigType as SpecificPretrainedConfigType
    except ImportError:
        SpecificPretrainedConfigType = TypeVar(
            "SpecificPretrainedConfigType",
            bound=PretrainedConfig,
        )

from .._npu import MobilintNPUBackend

# Re-export the migration helpers so external callers (tests, downstream
# integrations) can still import them from this module. The canonical
# implementations now live alongside :class:`NPUTargetSpec`.
from mblt_npu.npu_target import (
    _DEFAULT_DEV_NO,
    NPUTargetSpec,
    _migrate_target_clusters,
    _migrate_target_cores,
)

__all__ = [
    "MobilintConfigMixin",
    "MobilintEncoderDecoderConfigMixin",
    "MobilintVisionTextConfigMixin",
    "MobilintEagle3ConfigMixin",
    "MobilintMixtureOfExpertsConfigMixin",
    "NPUTargetSpec",
    "_normalize_npu_target_kwargs",
    "_migrate_target_cores",
    "_migrate_target_clusters",
]


def _normalize_npu_target_kwargs(kwargs: dict[str, Any], prefix: str = "") -> None:
    """Normalize NPU target fields inside ``kwargs`` in place.

    Thin wrapper over :meth:`NPUTargetSpec.from_kwargs`: rewrites
    ``{prefix}target_cores`` and ``{prefix}target_clusters`` to canonical
    fully-qualified representations, expands ``{prefix}dev_no`` sugar when
    both target lists are absent, unifies grain to the field appropriate for
    ``{prefix}core_mode``, and validates global8 coverage / device-set
    consistency. Delegated to :class:`NPUTargetSpec` so the same
    normalization logic runs from every entry point (config load, backend
    ctor, per-field setter).

    Args:
        kwargs: The keyword-argument dict handed to a config mixin's
            ``__init__`` or ``__post_init__``. Mutated in place: the target
            fields are replaced with canonical values and off-mode grain is
            popped.
        prefix: Optional prefix that scopes the NPU keys (e.g. ``"encoder_"``,
            ``"vision_"``, ``"base_"``). Empty for the default backend.

    Raises:
        ValueError: For any inconsistency the spec calls out — mixed legacy
            and new items, list-shaped ``dev_no`` combined with legacy items,
            device-set mismatch, or incomplete global8 coverage.
        TypeError: When a target entry has an unsupported type.
    """
    # The runtime's historical backend constructor defaults missing values to
    # ``single``. Keep the package policy in this shared config entry point so
    # every model and composite sub-config gets the same fallback.
    kwargs.setdefault(f"{prefix}core_mode", "auto")
    NPUTargetSpec.from_kwargs(kwargs, prefix=prefix)


# NPU fields that ``MobilintNPUBackend.from_dict`` consumes. Buffering them
# through :func:`_pop_consumed_backend_kwargs` / :func:`_split_npu_backend_kwargs`
# keeps HF's downstream setattr loop from re-firing every property setter
# (which would pollute :class:`NPUTargetSpecPending` with baseline echoes)
# and from probing topology getters against a stale pending (which would
# finalize the source spec against overrides intended for a different
# board — the hasattr hazard called out for
# :class:`MobilintVisionTextConfigMixin`).
#
# ``npu_prefill_chunk_size`` is deliberately excluded: it is a config-only
# attribute stored directly on ``self.__dict__`` and is not consumed by the
# shared backend, so leaving HF to apply it via the normal setattr loop
# preserves the caller's configured value.
_NPU_BACKEND_KWARG_FIELDS: tuple[str, ...] = (
    "mxq_path",
    "target_device",
    "target_cores",
    "target_clusters",
    "core_mode",
    "dev_no",
    "max_batch_size",
    "revision",
    "commit_hash",
)


# Order in which :func:`_apply_npu_backend_kwargs` replays the buffered
# NPU kwargs onto the config after :meth:`super().from_dict` returns.
# ``target_device`` runs first so a cross-board override rebuilds the
# backend before the topology / mode fields land on the (now-fresh)
# destination. The remaining order is arbitrary — each subsequent
# ``setattr`` accumulates on the fresh :class:`NPUTargetSpecPending` and
# the canonical spec finalizes only on the next getter read.
_NPU_APPLY_ORDER: tuple[str, ...] = (
    "target_device",
    "mxq_path",
    "revision",
    "commit_hash",
    "max_batch_size",
    "dev_no",
    "core_mode",
    "target_cores",
    "target_clusters",
)


# Kwargs consumed by ``MobilintNPUBackend.from_dict`` that
# ``MobilintConfigMixin`` does not surface as a forwarding property (either
# top-level or prefixed on the multi-backend mixins). Routing them through
# ``setattr(config, f"{prefix}{field}", ...)`` would land on the outer
# ``config.__dict__`` and never reach the nested backend, so
# :func:`_apply_npu_backend_kwargs` writes them straight onto the caller-
# supplied ``backend`` when one is given. The value maps the caller-facing
# kwarg name to the backend's storage-name — ``commit_hash`` is exposed
# publicly as a kwarg but stored as ``_commit_hash`` on the backend.
_NPU_BACKEND_DIRECT_FIELDS: dict[str, str] = {
    "revision": "revision",
    "commit_hash": "_commit_hash",
}


def _pop_consumed_backend_kwargs(kwargs: dict[str, Any], prefix: str = "") -> None:
    """Remove NPU-backend fields from ``kwargs`` after the backend consumes them.

    ``_ensure_*_npu_backend`` hands ``kwargs`` to
    :meth:`MobilintNPUBackend.from_dict`, which reads the fields it needs
    and constructs the backend. The caller-facing ``kwargs`` dict, however,
    still carries the very same NPU keys — HF's downstream
    ``PretrainedConfig.__init__(**kwargs)`` iterates every entry and calls
    ``setattr``, which routes right back through the config's NPU property
    setters. Each such setter appends a raw override to the backend's
    :class:`NPUTargetSpecPending` accumulator (``raw_cores`` /
    ``raw_clusters`` / etc.), so a freshly-loaded config ends up with
    pending state that echoes the just-baked baseline instead of the
    caller-neutral :data:`_UNSET`. That echo later corrupts any
    cross-board rebuild in :func:`_rebuild_backend_for_target_device`,
    which replays the source pending onto the destination backend on
    the assumption that non-``_UNSET`` slots represent real user intent.
    """
    for field in _NPU_BACKEND_KWARG_FIELDS:
        kwargs.pop(f"{prefix}{field}", None)


def _split_npu_backend_kwargs(
    kwargs: dict[str, Any], prefix: str = ""
) -> dict[str, Any]:
    """Pop the NPU-backend fields for ``prefix`` off ``kwargs``.

    Callers use the returned unprefixed dict with
    :func:`_apply_npu_backend_kwargs` after :meth:`super().from_dict` has
    processed the remaining kwargs, so the buffered fields are never
    exposed to HF's ``PretrainedConfig.from_dict`` kwargs loop. That loop
    otherwise probes every override with ``hasattr`` before ``setattr``,
    and the topology getters (``target_cores``, ``target_clusters``,
    ``core_mode``, ``dev_no``) all trigger a spec finalize on the current
    pending state. When a caller overrides ``target_device`` alongside
    a topology field, the finalize fires against the source board's
    topology before the target-device rebuild has a chance to run —
    e.g. ``core_mode="global8"`` on a Regulus config raises during the
    probe. Buffering routes the whole override set through
    :meth:`_apply_npu_backend_kwargs` in a controlled order instead.
    """
    sub: dict[str, Any] = {}
    for field in _NPU_BACKEND_KWARG_FIELDS:
        key = f"{prefix}{field}"
        if key in kwargs:
            sub[field] = kwargs.pop(key)
    return sub


def _apply_npu_backend_kwargs(
    config: PretrainedConfig,
    sub_kwargs: dict[str, Any],
    prefix: str = "",
    *,
    resolve_backend: Optional[Callable[[PretrainedConfig], MobilintNPUBackend]] = None,
) -> None:
    """Apply buffered NPU-backend fields to ``config`` in :data:`_NPU_APPLY_ORDER`.

    Companion to :func:`_split_npu_backend_kwargs`. ``target_device`` is
    applied first so a cross-board override rebuilds the backend before
    subsequent topology / mode assignments land — the destination
    board's fresh pending then absorbs the remaining fields without a
    partial-state finalize against the source topology.

    Fields listed in :data:`_NPU_BACKEND_DIRECT_FIELDS` (``revision`` /
    ``commit_hash``) have no forwarding property on
    ``MobilintConfigMixin`` (nor its multi-backend variants), so a plain
    ``setattr(config, f"{prefix}{field}", ...)`` would land on the outer
    config's ``__dict__`` and never reach the nested backend — remote MXQ
    resolution would silently keep the shipped revision.

    When the caller passes ``resolve_backend``, those fields are written
    straight onto the backend the callable returns, mapping the caller-
    facing kwarg name to the backend's storage-name (``commit_hash`` →
    ``_commit_hash``). The callable is invoked **per field**, not once,
    so a preceding ``target_device`` override that rebuilds the config's
    backend (:func:`_rebuild_backend_for_target_device` allocates a fresh
    instance and reassigns ``config.<prefix>npu_backend``) is followed by
    ``resolve_backend(config)`` returning the newly-installed backend —
    ``revision`` / ``commit_hash`` therefore always land on the live
    destination backend, never on the discarded source.
    """

    def _apply(field: str, value: Any) -> None:
        if resolve_backend is not None and field in _NPU_BACKEND_DIRECT_FIELDS:
            setattr(resolve_backend(config), _NPU_BACKEND_DIRECT_FIELDS[field], value)
        else:
            setattr(config, f"{prefix}{field}", value)

    remaining = dict(sub_kwargs)
    for field in _NPU_APPLY_ORDER:
        if field in remaining:
            _apply(field, remaining.pop(field))
    for field, value in remaining.items():
        _apply(field, value)


def _serialized_target_cores(backend: MobilintNPUBackend) -> list[str]:
    """Return JSON-safe core assignments for a Transformers configuration."""

    return list(backend._target_cores_serialized)


def _serialized_target_clusters(backend: MobilintNPUBackend) -> list[int]:
    """Return JSON-safe cluster assignments for a Transformers configuration."""

    return list(backend._target_clusters_serialized)


def _rebuild_backend_for_target_device(
    backend: MobilintNPUBackend, value: str, prefix: str = ""
) -> MobilintNPUBackend:
    """Apply a ``target_device`` reassignment to an existing NPU backend.

    HF :meth:`PretrainedConfig.from_dict` instantiates the config with the
    ``config.json`` values first and then applies any caller
    ``from_pretrained`` kwargs (including ``target_device``) via
    :func:`setattr`. The initial ``__init__`` has already dispatched
    :meth:`MobilintNPUBackend.__new__` to a concrete subclass based on the
    config-file board, so a plain string mutation on the existing backend
    would leave it on the wrong class (e.g. ``MobilintAriesBackend`` when
    the caller asked for ``regulus-rb-usb``).

    Three paths:

    * Unknown / unresolvable ``value``: mutate the string and let the
      backend layer raise its documented error on the next validation
      check rather than shadowing it here.
    * Same board class: mutate in place, but first normalize the value
      so a legacy alias like ``"aries"`` becomes the canonical
      ``"aries-rb"`` — otherwise ``to_dict`` would emit the alias and
      :class:`qbruntime.Accelerator` would receive an identifier its
      1.4 signature does not accept.
    * Cross-board: rebuild via :meth:`MobilintNPUBackend.from_dict` from
      the source backend's board-agnostic raw attributes (mxq_path,
      dev_no baseline, revision, commit_hash, max_batch_size,
      name_or_path). :meth:`to_dict` is deliberately avoided because it
      would finalize the source :class:`NPUTargetSpecPending` — a
      preceding kwarg like ``core_mode="global8"`` set on a Regulus
      backend accumulates as a pending override that would raise here
      against the source topology, making the rebuild order-dependent.
      After ``from_dict`` seeds a fresh backend with the destination
      board's default sugar (e.g. Regulus's sole ``d:0:0`` core in
      ``single`` mode), the caller's raw pending overrides (``dev_no``,
      ``core_mode``, ``target_cores``, ``target_clusters``) are replayed
      onto the fresh backend's pending accumulator so the complete
      override set applies atomically to the destination board
      regardless of setattr order in the HF kwargs loop.

    Args:
        backend: The current NPU backend held by the config.
        value: The requested board identifier (canonical name or legacy
            alias like ``"aries"`` / ``"regulus"``).
        prefix: Serialization prefix that scopes the NPU keys on this
            backend (``""``, ``"encoder_"``, ``"decoder_"``, ``"base_"``,
            ``"draft_"``, ``"fc_"``). Must match the prefix used by
            :meth:`MobilintNPUBackend.to_dict` / :meth:`from_dict` for
            the same backend.

    Returns:
        The backend to store on the config after the assignment. Either
        the original ``backend`` (mutated in place for unknown-value or
        same-class paths) or a freshly rebuilt one for the cross-board
        path.
    """
    from mblt_npu import (
        MobilintNPUBackend as _Backend,
    )
    from mblt_npu import (
        backend_class_for,
        normalize_target_device,
    )

    try:
        desired_cls = backend_class_for(value)
    except (KeyError, ValueError):
        backend.target_device = value
        return backend
    if type(backend) is desired_cls:
        backend.target_device = normalize_target_device(value)
        return backend
    source_pending = backend._pending
    state = {
        f"{prefix}mxq_path": backend.mxq_path,
        f"{prefix}max_batch_size": backend.max_batch_size,
        f"{prefix}revision": backend.revision,
        f"{prefix}commit_hash": backend._commit_hash,
        f"{prefix}name_or_path": backend.name_or_path,
        f"{prefix}target_device": value,
        # Baseline reads the last-finalized ``dev_no`` without triggering
        # a pending-state finalize; any caller ``dev_no`` override still
        # rides on ``source_pending.raw_dev_no`` and is replayed below.
        f"{prefix}dev_no": source_pending.baseline.dev_no_public(),
    }
    fresh = _Backend.from_dict(state, prefix=prefix)
    fresh._pending = fresh._pending._with(
        dev_no=source_pending.raw_dev_no,
        core_mode=source_pending.raw_core_mode,
        target_cores=source_pending.raw_cores,
        target_clusters=source_pending.raw_clusters,
    )
    fresh._finalized = None
    return fresh


class MobilintConfigMixin(PretrainedConfig):
    # ``dev_no`` is exposed as syntactic sugar for the device-prefix component
    # of the canonical target strings. It accepts either a single device index
    # or a list of indices; downstream normalization expands it into
    # ``target_cores`` / ``target_clusters`` when the caller does not specify
    # targets directly.
    _NPU_SIGNATURE_FIELDS = (
        ("mxq_path", "", str),
        ("dev_no", _DEFAULT_DEV_NO, Union[int, list[int]]),
        ("max_batch_size", 1, int),
        ("core_mode", "auto", str),
        ("target_device", "aries-rb", str),
        ("target_cores", None, Any),
        ("target_clusters", None, Any),
        ("revision", None, Any),
        ("npu_prefill_chunk_size", None, Any),
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._augment_init_signature()

    @classmethod
    def _augment_init_signature(cls) -> None:
        """Expose Mobilint backend kwargs to upstream config introspection."""
        init = cls.__init__
        signature = inspect.signature(init)
        if any(name in signature.parameters for name, _, _ in cls._NPU_SIGNATURE_FIELDS):
            return

        parameters = list(signature.parameters.values())
        insert_at = next(
            (index for index, parameter in enumerate(parameters) if parameter.kind == Parameter.VAR_KEYWORD),
            len(parameters),
        )
        extra_parameters = [
            Parameter(name=name, kind=Parameter.KEYWORD_ONLY, default=default, annotation=annotation)
            for name, default, annotation in cls._NPU_SIGNATURE_FIELDS
        ]
        init.__signature__ = Signature(parameters[:insert_at] + extra_parameters + parameters[insert_at:])

    def _ensure_npu_backend(self, kwargs: dict[str, Any]) -> None:
        if not hasattr(self, "npu_backend"):
            _normalize_npu_target_kwargs(kwargs, prefix="")
            self.npu_backend = MobilintNPUBackend.from_dict(kwargs, prefix="")
            _pop_consumed_backend_kwargs(kwargs, prefix="")

    def __init__(self, *args, **kwargs):
        self._ensure_npu_backend(kwargs)
        super().__init__(*args, **kwargs)

    def __post_init__(self, **kwargs: Any) -> None:
        self._ensure_npu_backend(kwargs)
        super().__post_init__(**kwargs)

    @property
    def mxq_path(self) -> str:
        return self.npu_backend.mxq_path

    @mxq_path.setter
    def mxq_path(self, value: str) -> None:
        self.npu_backend.mxq_path = value

    @property
    def target_device(self) -> str:
        """Board identifier used by the shared NPU backend."""
        return self.npu_backend.target_device

    @target_device.setter
    def target_device(self, value: str) -> None:
        self.npu_backend = _rebuild_backend_for_target_device(self.npu_backend, value)

    @property
    def dev_no(self) -> int:
        return self.npu_backend.dev_no

    @dev_no.setter
    def dev_no(self, value: int) -> None:
        self.npu_backend.dev_no = value

    @property
    def core_mode(self) -> str:
        return self.npu_backend.core_mode

    @core_mode.setter
    def core_mode(self, value: str) -> None:
        self.npu_backend.core_mode = value

    @property
    def max_batch_size(self) -> int:
        return self.npu_backend.max_batch_size

    @max_batch_size.setter
    def max_batch_size(self, value: int) -> None:
        self.npu_backend.max_batch_size = max(1, value)

    @property
    def target_cores(self) -> list:
        return _serialized_target_cores(self.npu_backend)

    @target_cores.setter
    def target_cores(self, values: list) -> None:
        self.npu_backend.target_cores = values

    @property
    def target_clusters(self) -> list:
        return _serialized_target_clusters(self.npu_backend)

    @target_clusters.setter
    def target_clusters(self, values: list) -> None:
        self.npu_backend.target_clusters = values

    @property
    def npu_prefill_chunk_size(self) -> Any:
        return self.__dict__.get("npu_prefill_chunk_size", None)

    @npu_prefill_chunk_size.setter
    def npu_prefill_chunk_size(self, value: Any) -> None:
        self.__dict__["npu_prefill_chunk_size"] = value

    @classmethod
    def from_dict(
        cls: type[SpecificPretrainedConfigType], config_dict: dict[str, Any], **kwargs
    ) -> Union["MobilintConfigMixin", tuple["MobilintConfigMixin", dict[str, Any]]]:
        """Buffer NPU-backend kwargs across HF's ``from_dict`` kwargs loop.

        HF ``PretrainedConfig.from_dict`` probes every override with
        ``hasattr`` before ``setattr``. The topology getters
        (``target_cores`` / ``target_clusters`` / ``core_mode`` /
        ``dev_no``) all trigger a spec finalize on the current pending
        state, so a caller override set that combines ``target_device``
        with a board-specific mode (e.g. ``core_mode="global8"``) raises
        during the probe against the source board's topology before the
        target-device rebuild has a chance to run. Extract the NPU
        fields ahead of time and apply them once ``super().from_dict``
        has finished with the remaining kwargs; the shared apply order
        (see :data:`_NPU_APPLY_ORDER`) runs ``target_device`` first, so
        cross-board rebuilds land before topology / mode overrides.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        npu_sub_kwargs = _split_npu_backend_kwargs(kwargs)

        config, unused_kwargs = super().from_dict(
            config_dict, return_unused_kwargs=True, **kwargs
        )  # type: ignore[misc]

        _apply_npu_backend_kwargs(
            config,
            npu_sub_kwargs,
            resolve_backend=lambda c: c.npu_backend,
        )

        if return_unused_kwargs:
            return config, unused_kwargs
        return config

    def _remove_keys_not_serialized(self, d: dict[str, Any]) -> None:
        if hasattr(self, "npu_backend"):
            _ = d.pop("npu_backend", None)

        super()._remove_keys_not_serialized(d)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config and flatten Mobilint NPU backend fields into the top level.

        The ``npu_backend`` attribute is temporarily detached before delegating to the upstream
        :meth:`PretrainedConfig.to_dict` implementation so it is neither serialized as a nested
        object nor picked up by upstream diff/equality helpers. It is reattached in a ``finally``
        block, and its unprefixed field mapping is merged into the returned dictionary so callers
        see the individual NPU parameters (``mxq_path``, ``dev_no``, etc.) at the top level.

        Returns:
            A dictionary representation of the config with the Mobilint NPU backend fields
            merged in at the top level (no ``npu_backend`` key).
        """
        npu_backend = getattr(self, "npu_backend", None)
        if npu_backend is not None:
            del self.npu_backend
        try:
            output = super().to_dict()
        finally:
            if npu_backend is not None:
                self.npu_backend = npu_backend
        if npu_backend is not None:
            output.update(npu_backend.to_dict(prefix=""))
        return output


class MobilintEncoderDecoderConfigMixin(PretrainedConfig):
    def _ensure_encoder_decoder_npu_backends(self, kwargs: dict[str, Any]) -> None:
        if not hasattr(self, "encoder_npu_backend"):
            _normalize_npu_target_kwargs(kwargs, prefix="encoder_")
            self.encoder_npu_backend = MobilintNPUBackend.from_dict(kwargs, prefix="encoder_")
            _pop_consumed_backend_kwargs(kwargs, prefix="encoder_")

        if not hasattr(self, "decoder_npu_backend"):
            _normalize_npu_target_kwargs(kwargs, prefix="decoder_")
            self.decoder_npu_backend = MobilintNPUBackend.from_dict(kwargs, prefix="decoder_")
            _pop_consumed_backend_kwargs(kwargs, prefix="decoder_")

    def __init__(self, **kwargs):
        self._ensure_encoder_decoder_npu_backends(kwargs)
        super().__init__(**kwargs)

    def __post_init__(self, **kwargs: Any) -> None:
        self._ensure_encoder_decoder_npu_backends(kwargs)
        super().__post_init__(**kwargs)

    @property
    def encoder_mxq_path(self) -> str:
        return self.encoder_npu_backend.mxq_path

    @encoder_mxq_path.setter
    def encoder_mxq_path(self, value: str) -> None:
        self.encoder_npu_backend.mxq_path = value

    @property
    def encoder_target_device(self) -> str:
        """Board identifier used by the encoder NPU backend."""
        return self.encoder_npu_backend.target_device

    @encoder_target_device.setter
    def encoder_target_device(self, value: str) -> None:
        self.encoder_npu_backend = _rebuild_backend_for_target_device(
            self.encoder_npu_backend, value, prefix="encoder_"
        )

    @property
    def encoder_dev_no(self) -> int:
        return self.encoder_npu_backend.dev_no

    @encoder_dev_no.setter
    def encoder_dev_no(self, value: int) -> None:
        self.encoder_npu_backend.dev_no = value

    @property
    def encoder_core_mode(self) -> str:
        return self.encoder_npu_backend.core_mode

    @encoder_core_mode.setter
    def encoder_core_mode(self, value: str) -> None:
        self.encoder_npu_backend.core_mode = value

    @property
    def encoder_max_batch_size(self) -> int:
        return self.encoder_npu_backend.max_batch_size

    @encoder_max_batch_size.setter
    def encoder_max_batch_size(self, value: int) -> None:
        self.encoder_npu_backend.max_batch_size = max(1, value)

    @property
    def encoder_target_cores(self) -> list:
        return _serialized_target_cores(self.encoder_npu_backend)

    @encoder_target_cores.setter
    def encoder_target_cores(self, values: list) -> None:
        self.encoder_npu_backend.target_cores = values

    @property
    def encoder_target_clusters(self) -> list:
        return _serialized_target_clusters(self.encoder_npu_backend)

    @encoder_target_clusters.setter
    def encoder_target_clusters(self, values: list) -> None:
        self.encoder_npu_backend.target_clusters = values

    @property
    def decoder_mxq_path(self) -> str:
        return self.decoder_npu_backend.mxq_path

    @decoder_mxq_path.setter
    def decoder_mxq_path(self, value: str) -> None:
        self.decoder_npu_backend.mxq_path = value

    @property
    def decoder_target_device(self) -> str:
        """Board identifier used by the decoder NPU backend."""
        return self.decoder_npu_backend.target_device

    @decoder_target_device.setter
    def decoder_target_device(self, value: str) -> None:
        self.decoder_npu_backend = _rebuild_backend_for_target_device(
            self.decoder_npu_backend, value, prefix="decoder_"
        )

    @property
    def decoder_dev_no(self) -> int:
        return self.decoder_npu_backend.dev_no

    @decoder_dev_no.setter
    def decoder_dev_no(self, value: int) -> None:
        self.decoder_npu_backend.dev_no = value

    @property
    def decoder_core_mode(self) -> str:
        return self.decoder_npu_backend.core_mode

    @decoder_core_mode.setter
    def decoder_core_mode(self, value: str) -> None:
        self.decoder_npu_backend.core_mode = value

    @property
    def decoder_max_batch_size(self) -> int:
        return self.decoder_npu_backend.max_batch_size

    @decoder_max_batch_size.setter
    def decoder_max_batch_size(self, value: int) -> None:
        self.decoder_npu_backend.max_batch_size = max(1, value)

    @property
    def decoder_target_cores(self) -> list:
        return _serialized_target_cores(self.decoder_npu_backend)

    @decoder_target_cores.setter
    def decoder_target_cores(self, values: list) -> None:
        self.decoder_npu_backend.target_cores = values

    @property
    def decoder_target_clusters(self) -> list:
        return _serialized_target_clusters(self.decoder_npu_backend)

    @decoder_target_clusters.setter
    def decoder_target_clusters(self, values: list) -> None:
        self.decoder_npu_backend.target_clusters = values

    @classmethod
    def from_dict(
        cls: type[SpecificPretrainedConfigType], config_dict: dict[str, Any], **kwargs
    ) -> Union[
        "MobilintEncoderDecoderConfigMixin",
        tuple["MobilintEncoderDecoderConfigMixin", dict[str, Any]],
    ]:
        """Buffer ``encoder_*`` / ``decoder_*`` NPU kwargs across HF's ``from_dict``.

        Mirrors :meth:`MobilintConfigMixin.from_dict` for the two prefixed
        sub-backends. See that override's docstring for the hasattr-probe
        rationale.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        encoder_sub = _split_npu_backend_kwargs(kwargs, prefix="encoder_")
        decoder_sub = _split_npu_backend_kwargs(kwargs, prefix="decoder_")

        config, unused_kwargs = super().from_dict(
            config_dict, return_unused_kwargs=True, **kwargs
        )  # type: ignore[misc]

        _apply_npu_backend_kwargs(
            config,
            encoder_sub,
            prefix="encoder_",
            resolve_backend=lambda c: c.encoder_npu_backend,
        )
        _apply_npu_backend_kwargs(
            config,
            decoder_sub,
            prefix="decoder_",
            resolve_backend=lambda c: c.decoder_npu_backend,
        )

        if return_unused_kwargs:
            return config, unused_kwargs
        return config

    def _remove_keys_not_serialized(self, d: dict[str, Any]) -> None:
        if hasattr(self, "encoder_npu_backend"):
            _ = d.pop("encoder_npu_backend", None)

        if hasattr(self, "decoder_npu_backend"):
            _ = d.pop("decoder_npu_backend", None)

        super()._remove_keys_not_serialized(d)

    def to_dict(self):
        output = super().to_dict()

        if hasattr(self, "encoder_npu_backend"):
            output.update(self.encoder_npu_backend.to_dict(prefix="encoder_"))
        if hasattr(self, "decoder_npu_backend"):
            output.update(self.decoder_npu_backend.to_dict(prefix="decoder_"))

        return output

    def get_text_config(self, decoder=None, encoder=None) -> "PretrainedConfig":
        return self


class MobilintVisionTextConfigMixin(PretrainedConfig):
    sub_configs = {"vision_config": MobilintConfigMixin, "text_config": MobilintConfigMixin}

    _SUB_BACKEND_FIELDS = (
        "mxq_path",
        "dev_no",
        "max_batch_size",
        "core_mode",
        "target_device",
        "target_cores",
        "target_clusters",
        "npu_prefill_chunk_size",
    )

    @classmethod
    def _split_sub_backend_kwargs(cls, kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pop ``text_*`` / ``vision_*`` NPU keys out of ``kwargs`` in place.

        Upstream composite configs (e.g. ``Qwen2VLConfig``, ``BlipConfig``) may
        call ``PretrainedConfig.__init__(**kwargs)`` before instantiating their
        sub-configs, which triggers the prefixed property setters on this mixin
        while ``self.text_config`` / ``self.vision_config`` do not yet exist.
        Removing the keys up front and re-applying them after the sub-configs
        are built avoids that ordering hazard.
        """
        text_kwargs: dict[str, Any] = {}
        vision_kwargs: dict[str, Any] = {}
        for field in cls._SUB_BACKEND_FIELDS:
            text_key = f"text_{field}"
            if text_key in kwargs:
                text_kwargs[field] = kwargs.pop(text_key)
            vision_key = f"vision_{field}"
            if vision_key in kwargs:
                vision_kwargs[field] = kwargs.pop(vision_key)
        return text_kwargs, vision_kwargs

    def _apply_sub_backend_kwargs(
        self, text_kwargs: dict[str, Any], vision_kwargs: dict[str, Any]
    ) -> None:
        # Overrides come from two paths: the JSON-load helper that unpacks
        # top-level ``text_*`` / ``vision_*`` keys, and HF's ``from_pretrained``
        # model-kwargs application. Both routes reach here as unprefixed keys
        # (``dev_no`` etc.) that we route through the sub-config's own
        # setters. Each setter records its raw override on the sub-config
        # backend's :class:`NPUTargetSpecPending` accumulator without
        # normalizing between fields; the canonical spec is materialized once
        # on the next :attr:`MobilintNPUBackend._spec` read, so setter order
        # inside this loop is irrelevant.
        text_config = getattr(self, "text_config", None)
        if text_config is not None:
            for key, value in text_kwargs.items():
                setattr(text_config, key, value)
        vision_config = getattr(self, "vision_config", None)
        if vision_config is not None:
            for key, value in vision_kwargs.items():
                setattr(vision_config, key, value)

    @PretrainedConfig.name_or_path.setter
    def name_or_path(self, value):
        PretrainedConfig.name_or_path.fset(self, value)
        vision_config = getattr(self, "vision_config", None)
        if vision_config is not None:
            vision_config.name_or_path = value
        text_config = getattr(self, "text_config", None)
        if text_config is not None:
            text_config.name_or_path = value

    @property
    def vision_mxq_path(self) -> str:
        return self.vision_config.mxq_path

    @vision_mxq_path.setter
    def vision_mxq_path(self, value: str) -> None:
        self.vision_config.mxq_path = value

    @property
    def vision_dev_no(self) -> int:
        return self.vision_config.dev_no

    @vision_dev_no.setter
    def vision_dev_no(self, value: int) -> None:
        self.vision_config.dev_no = value

    @property
    def vision_core_mode(self) -> str:
        return self.vision_config.core_mode

    @vision_core_mode.setter
    def vision_core_mode(self, value: str) -> None:
        self.vision_config.core_mode = value

    @property
    def vision_max_batch_size(self) -> int:
        return self.vision_config.max_batch_size

    @vision_max_batch_size.setter
    def vision_max_batch_size(self, value: int) -> None:
        self.vision_config.max_batch_size = max(1, value)

    @property
    def vision_target_cores(self) -> list:
        return _serialized_target_cores(self.vision_config.npu_backend)

    @vision_target_cores.setter
    def vision_target_cores(self, values: list) -> None:
        self.vision_config.target_cores = values

    @property
    def vision_target_clusters(self) -> list:
        return _serialized_target_clusters(self.vision_config.npu_backend)

    @vision_target_clusters.setter
    def vision_target_clusters(self, values: list) -> None:
        self.vision_config.target_clusters = values

    @property
    def text_mxq_path(self) -> str:
        return self.text_config.mxq_path

    @text_mxq_path.setter
    def text_mxq_path(self, value: str) -> None:
        self.text_config.mxq_path = value

    @property
    def text_dev_no(self) -> int:
        return self.text_config.dev_no

    @text_dev_no.setter
    def text_dev_no(self, value: int) -> None:
        self.text_config.dev_no = value

    @property
    def text_core_mode(self) -> str:
        return self.text_config.core_mode

    @text_core_mode.setter
    def text_core_mode(self, value: str) -> None:
        self.text_config.core_mode = value

    @property
    def text_max_batch_size(self) -> int:
        return self.text_config.max_batch_size

    @text_max_batch_size.setter
    def text_max_batch_size(self, value: int) -> None:
        self.text_config.max_batch_size = max(1, value)

    @property
    def text_target_cores(self) -> list:
        return _serialized_target_cores(self.text_config.npu_backend)

    @text_target_cores.setter
    def text_target_cores(self, values: list) -> None:
        self.text_config.target_cores = values

    @property
    def text_target_clusters(self) -> list:
        return _serialized_target_clusters(self.text_config.npu_backend)

    @text_target_clusters.setter
    def text_target_clusters(self, values: list) -> None:
        self.text_config.target_clusters = values

    @property
    def text_npu_prefill_chunk_size(self) -> Any:
        return self.text_config.npu_prefill_chunk_size

    @text_npu_prefill_chunk_size.setter
    def text_npu_prefill_chunk_size(self, value: Any) -> None:
        self.text_config.npu_prefill_chunk_size = value

    @classmethod
    def from_dict(
        cls: type[SpecificPretrainedConfigType], config_dict: dict[str, Any], **kwargs
    ) -> Union["MobilintVisionTextConfigMixin", tuple["MobilintVisionTextConfigMixin", dict[str, Any]]]:
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # Buffer sub-backend NPU keys until after upstream's kwargs loop:
        # ``PretrainedConfig.from_dict`` ``hasattr``-probes each override, which
        # routes the prefixed NPU properties through the sub-config backends and
        # forces a lazy finalize on the pending state visible at that moment.
        # With a ``global8`` config_dict and a caller narrowing target_cores to
        # a single cluster, that partial-state finalize trips
        # ``_validate_global8_coverage`` before the matching ``core_mode='single'``
        # override lands. Applying the overrides as one group via
        # ``_apply_sub_backend_kwargs`` after upstream returns makes finalize see
        # the fully-consistent override set.
        text_sub_kwargs, vision_sub_kwargs = cls._split_sub_backend_kwargs(kwargs)

        config: MobilintVisionTextConfigMixin
        unused_kwargs: dict[str, Any]
        config, unused_kwargs = super().from_dict(config_dict, return_unused_kwargs=True, **kwargs)  # type: ignore

        config._apply_sub_backend_kwargs(text_sub_kwargs, vision_sub_kwargs)

        for sub_config in (config.text_config, config.vision_config):
            sub_config.name_or_path = config.name_or_path

            revision = getattr(config, "revision", None)
            if revision:
                sub_config.revision = revision

            commit_hash = getattr(config, "_commit_hash", None)
            if commit_hash:
                sub_config._commit_hash = commit_hash

        if return_unused_kwargs:
            return config, unused_kwargs
        else:
            return config

    @classmethod
    def from_text_vision_configs(
        cls,
        text_config: MobilintConfigMixin,
        vision_config: MobilintConfigMixin,
        **kwargs,
    ):
        return cls(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            **kwargs,
        )


class MobilintEagle3ConfigMixin(PretrainedConfig):
    """Config mixin for EAGLE-3 models with base/draft/fc backends."""

    sub_configs = {"draft_config": MobilintConfigMixin}
    _EAGLE3_BACKEND_FIELDS = (
        "mxq_path",
        "dev_no",
        "max_batch_size",
        "core_mode",
        "target_device",
        "target_cores",
        "target_clusters",
        "revision",
        "commit_hash",
    )
    # ``eagle3_tree_depth`` / ``eagle3_tree_top_k`` here are the legacy location: ``generate`` prefers the
    # same-named ``generation_config`` fields and only falls back to these for older releases. The NPU chunk
    # size uses the shared ``npu_prefill_chunk_size`` field (int or per-core-mode dict), like other LLMs.
    _EAGLE3_RUNTIME_FIELDS = (
        ("eagle3_tree_depth", 5, int),
        ("eagle3_tree_top_k", 8, int),
    )

    @classmethod
    def _get_draft_config_class(cls) -> type[PretrainedConfig]:
        return MobilintConfigMixin

    def _init_or_coerce_draft_config(self, draft_config: Any | None) -> None:
        draft_config_cls = self._get_draft_config_class()

        if draft_config is None:
            coerced_draft_config = draft_config_cls()
        elif isinstance(draft_config, dict):
            coerced_draft_config = draft_config_cls(**draft_config)
        elif isinstance(draft_config, draft_config_cls):
            coerced_draft_config = draft_config
        else:
            raise TypeError(
                f"draft_config must be None, dict, or {draft_config_cls.__name__}; got {type(draft_config).__name__}"
            )

        self.draft_config = coerced_draft_config
        self.draft_config.name_or_path = self.name_or_path

    def _ensure_eagle3_npu_backends(self, kwargs: dict[str, Any]) -> None:
        def _resolve_backend_kwargs(prefix: str) -> dict[str, Any]:
            backend_kwargs: dict[str, Any] = {}
            for field_name in self._EAGLE3_BACKEND_FIELDS:
                prefixed_key = f"{prefix}{field_name}"
                if prefixed_key in kwargs:
                    backend_kwargs[prefixed_key] = kwargs[prefixed_key]
                    continue

                if field_name in kwargs:
                    backend_kwargs[prefixed_key] = kwargs[field_name]
            return backend_kwargs

        if not hasattr(self, "base_npu_backend"):
            base_kwargs = _resolve_backend_kwargs("base_")
            _normalize_npu_target_kwargs(base_kwargs, prefix="base_")
            self.base_npu_backend = MobilintNPUBackend.from_dict(base_kwargs, prefix="base_")
        if not hasattr(self, "draft_npu_backend"):
            draft_kwargs = _resolve_backend_kwargs("draft_")
            _normalize_npu_target_kwargs(draft_kwargs, prefix="draft_")
            self.draft_npu_backend = MobilintNPUBackend.from_dict(draft_kwargs, prefix="draft_")
        if not hasattr(self, "fc_npu_backend"):
            fc_kwargs = _resolve_backend_kwargs("fc_")
            _normalize_npu_target_kwargs(fc_kwargs, prefix="fc_")
            self.fc_npu_backend = MobilintNPUBackend.from_dict(fc_kwargs, prefix="fc_")
        # Drop the base_/draft_/fc_ NPU keys and their unprefixed fallbacks
        # from ``kwargs`` so downstream ``PretrainedConfig.__init__`` does
        # not re-apply them via setattr and pollute the just-baked
        # ``NPUTargetSpecPending`` state — see the top-level
        # :func:`_pop_consumed_backend_kwargs` docstring for the underlying
        # HF init pattern.
        for prefix in ("base_", "draft_", "fc_", ""):
            _pop_consumed_backend_kwargs(kwargs, prefix=prefix)

    def _ensure_eagle3_runtime_fields(self, kwargs: dict[str, Any]) -> None:
        for field_name, default_value, _annotation in self._EAGLE3_RUNTIME_FIELDS:
            if hasattr(self, field_name):
                continue
            value = kwargs.pop(field_name, default_value)
            self.__dict__[field_name] = self._coerce_positive_runtime_int(
                field_name,
                value,
                default_value=default_value,
            )

    @staticmethod
    def _coerce_positive_runtime_int(field_name: str, value: Any, *, default_value: int) -> int:
        """Coerce runtime integer fields and reject invalid ranges."""
        candidate = default_value if value is None else value
        try:
            coerced = int(candidate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be an integer, got {candidate!r}") from exc
        if coerced <= 0:
            raise ValueError(f"{field_name} must be > 0, got {coerced}")
        return coerced

    def __init__(self, **kwargs):
        draft_config = kwargs.pop("draft_config", None)
        self._ensure_eagle3_npu_backends(kwargs)
        self._ensure_eagle3_runtime_fields(kwargs)
        super().__init__(**kwargs)
        self._init_or_coerce_draft_config(draft_config)

    def __post_init__(self, **kwargs: Any) -> None:
        draft_config = kwargs.pop("draft_config", getattr(self, "draft_config", None))
        self._ensure_eagle3_npu_backends(kwargs)
        self._ensure_eagle3_runtime_fields(kwargs)
        super().__post_init__(**kwargs)
        self._init_or_coerce_draft_config(draft_config)

    @PretrainedConfig.name_or_path.setter
    def name_or_path(self, value: str):
        PretrainedConfig.name_or_path.fset(self, value)
        draft_config = getattr(self, "draft_config", None)
        if draft_config is not None:
            draft_config.name_or_path = value

    @property
    def base_dev_no(self) -> int:
        return self.base_npu_backend.dev_no

    @base_dev_no.setter
    def base_dev_no(self, value: int) -> None:
        self.base_npu_backend.dev_no = value

    @property
    def draft_dev_no(self) -> int:
        return self.draft_npu_backend.dev_no

    @draft_dev_no.setter
    def draft_dev_no(self, value: int) -> None:
        self.draft_npu_backend.dev_no = value

    @property
    def fc_dev_no(self) -> int:
        return self.fc_npu_backend.dev_no

    @fc_dev_no.setter
    def fc_dev_no(self, value: int) -> None:
        self.fc_npu_backend.dev_no = value

    @property
    def base_target_device(self) -> str:
        return self.base_npu_backend.target_device

    @base_target_device.setter
    def base_target_device(self, value: str) -> None:
        self.base_npu_backend = _rebuild_backend_for_target_device(
            self.base_npu_backend, value, prefix="base_"
        )

    @property
    def draft_target_device(self) -> str:
        return self.draft_npu_backend.target_device

    @draft_target_device.setter
    def draft_target_device(self, value: str) -> None:
        self.draft_npu_backend = _rebuild_backend_for_target_device(
            self.draft_npu_backend, value, prefix="draft_"
        )

    @property
    def fc_target_device(self) -> str:
        return self.fc_npu_backend.target_device

    @fc_target_device.setter
    def fc_target_device(self, value: str) -> None:
        self.fc_npu_backend = _rebuild_backend_for_target_device(
            self.fc_npu_backend, value, prefix="fc_"
        )

    @property
    def base_max_batch_size(self) -> int:
        return self.base_npu_backend.max_batch_size

    @base_max_batch_size.setter
    def base_max_batch_size(self, value: int) -> None:
        self.base_npu_backend.max_batch_size = max(1, value)

    @property
    def draft_max_batch_size(self) -> int:
        return self.draft_npu_backend.max_batch_size

    @draft_max_batch_size.setter
    def draft_max_batch_size(self, value: int) -> None:
        self.draft_npu_backend.max_batch_size = max(1, value)

    @property
    def fc_max_batch_size(self) -> int:
        return self.fc_npu_backend.max_batch_size

    @fc_max_batch_size.setter
    def fc_max_batch_size(self, value: int) -> None:
        self.fc_npu_backend.max_batch_size = max(1, value)

    @property
    def base_core_mode(self) -> str:
        return self.base_npu_backend.core_mode

    @base_core_mode.setter
    def base_core_mode(self, value: str) -> None:
        self.base_npu_backend.core_mode = value

    @property
    def draft_core_mode(self) -> str:
        return self.draft_npu_backend.core_mode

    @draft_core_mode.setter
    def draft_core_mode(self, value: str) -> None:
        self.draft_npu_backend.core_mode = value

    @property
    def fc_core_mode(self) -> str:
        return self.fc_npu_backend.core_mode

    @fc_core_mode.setter
    def fc_core_mode(self, value: str) -> None:
        self.fc_npu_backend.core_mode = value

    @property
    def base_target_cores(self) -> list[str]:
        return _serialized_target_cores(self.base_npu_backend)

    @base_target_cores.setter
    def base_target_cores(self, values: list[str]) -> None:
        self.base_npu_backend.target_cores = values

    @property
    def draft_target_cores(self) -> list[str]:
        return _serialized_target_cores(self.draft_npu_backend)

    @draft_target_cores.setter
    def draft_target_cores(self, values: list[str]) -> None:
        self.draft_npu_backend.target_cores = values

    @property
    def fc_target_cores(self) -> list[str]:
        return _serialized_target_cores(self.fc_npu_backend)

    @fc_target_cores.setter
    def fc_target_cores(self, values: list[str]) -> None:
        self.fc_npu_backend.target_cores = values

    @property
    def base_target_clusters(self) -> list[int]:
        return _serialized_target_clusters(self.base_npu_backend)

    @base_target_clusters.setter
    def base_target_clusters(self, values: list[int]) -> None:
        self.base_npu_backend.target_clusters = values

    @property
    def draft_target_clusters(self) -> list[int]:
        return _serialized_target_clusters(self.draft_npu_backend)

    @draft_target_clusters.setter
    def draft_target_clusters(self, values: list[int]) -> None:
        self.draft_npu_backend.target_clusters = values

    @property
    def fc_target_clusters(self) -> list[int]:
        return _serialized_target_clusters(self.fc_npu_backend)

    @fc_target_clusters.setter
    def fc_target_clusters(self, values: list[int]) -> None:
        self.fc_npu_backend.target_clusters = values

    @property
    def base_mxq_path(self) -> str:
        return self.base_npu_backend.mxq_path

    @base_mxq_path.setter
    def base_mxq_path(self, value: str) -> None:
        self.base_npu_backend.mxq_path = value

    @property
    def draft_mxq_path(self) -> str:
        return self.draft_npu_backend.mxq_path

    @draft_mxq_path.setter
    def draft_mxq_path(self, value: str) -> None:
        self.draft_npu_backend.mxq_path = value

    @property
    def fc_mxq_path(self) -> str:
        return self.fc_npu_backend.mxq_path

    @fc_mxq_path.setter
    def fc_mxq_path(self, value: str) -> None:
        self.fc_npu_backend.mxq_path = value

    @classmethod
    def from_dict(
        cls: type[SpecificPretrainedConfigType], config_dict: dict[str, Any], **kwargs
    ) -> Union[
        "MobilintEagle3ConfigMixin",
        tuple["MobilintEagle3ConfigMixin", dict[str, Any]],
    ]:
        """Buffer ``base_*`` / ``draft_*`` / ``fc_*`` NPU kwargs across HF's ``from_dict``.

        Mirrors :meth:`MobilintConfigMixin.from_dict` for Eagle3's three
        prefixed sub-backends. See that override's docstring for the
        hasattr-probe rationale.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        base_sub = _split_npu_backend_kwargs(kwargs, prefix="base_")
        draft_sub = _split_npu_backend_kwargs(kwargs, prefix="draft_")
        fc_sub = _split_npu_backend_kwargs(kwargs, prefix="fc_")

        config, unused_kwargs = super().from_dict(
            config_dict, return_unused_kwargs=True, **kwargs
        )  # type: ignore[misc]

        _apply_npu_backend_kwargs(
            config,
            base_sub,
            prefix="base_",
            resolve_backend=lambda c: c.base_npu_backend,
        )
        _apply_npu_backend_kwargs(
            config,
            draft_sub,
            prefix="draft_",
            resolve_backend=lambda c: c.draft_npu_backend,
        )
        _apply_npu_backend_kwargs(
            config,
            fc_sub,
            prefix="fc_",
            resolve_backend=lambda c: c.fc_npu_backend,
        )

        if return_unused_kwargs:
            return config, unused_kwargs
        return config

    def _remove_keys_not_serialized(self, d: dict[str, Any]) -> None:
        _ = d.pop("base_npu_backend", None)
        _ = d.pop("draft_npu_backend", None)
        _ = d.pop("fc_npu_backend", None)
        super()._remove_keys_not_serialized(d)

    def to_dict(self):
        output = super().to_dict()
        output.update(self.base_npu_backend.to_dict(prefix="base_"))
        output.update(self.draft_npu_backend.to_dict(prefix="draft_"))
        output.update(self.fc_npu_backend.to_dict(prefix="fc_"))
        return output


class MobilintMixtureOfExpertsConfigMixin(PretrainedConfig):
    """Config mixin for Mixture-of-Experts models split into shared/expert/lm_head MXQ roles.

    A MoE release ships one ``shared`` MXQ per decoder layer (attention, router gate and residual), one MXQ per
    ``(layer, expert)`` pair, and a single ``lm_head`` MXQ. Each role owns its own NPU target spec under the
    ``shared_`` / ``expert_`` / ``lm_head_`` prefixes, following :class:`MobilintEagle3ConfigMixin`. The role
    ``mxq_path`` fields are ``str.format`` templates: ``shared_mxq_path`` receives ``layer`` and
    ``expert_mxq_path`` receives ``layer`` and ``expert``.

    Placement is round-robin over the devices each role's ``target_cores`` cover: ``shared[layer]`` runs on the
    ``layer % n``-th shared device, ``expert[layer][expert]`` on the ``expert % n``-th expert device, and the
    ``lm_head`` on its single device. Only ``single`` core mode is supported; the cores listed for a device form
    that device's ``ModelConfig``.

    Unlike EAGLE-3, unprefixed NPU kwargs are **not** fanned out to every role: the roles need very different
    core sets (one core per shared MXQ versus eight per expert MXQ), so an unprefixed ``dev_no`` would expand
    every role to every core of the device and multiply the shared/lm_head footprint by eight.

    ``moe_mxq_pack_index`` points at the index JSON of a packed release (see
    :class:`~transformers_mblt.utils.moe_utils.MobilintMixtureOfExpertsPackIndex`); the shared and expert MXQs then
    come from the per-layer packs and ``shared_mxq_path`` / ``expert_mxq_path`` are unused. Without it the role
    templates name individual MXQ files.

    ``moe_shared_output_order`` lists the role of each shared MXQ output (``residual``, ``expert_in``,
    ``gate``). The ``"default"`` entry applies to every layer without a per-layer override keyed by the layer
    index as a string. Like ``vision_output_order`` on Qwen3-VL, the order is a property of the compiled
    artifact and travels with it in ``config.json``.
    """

    _MOE_BACKEND_PREFIXES = ("shared_", "expert_", "lm_head_")
    _MOE_BACKEND_FIELDS = MobilintEagle3ConfigMixin._EAGLE3_BACKEND_FIELDS
    MOE_SHARED_OUTPUT_ROLES = ("residual", "expert_in", "gate")
    _MOE_DEFAULT_TARGET_CORES = {
        "shared_": ["0:0:0"],
        "expert_": [f"0:{cluster}:{core}" for cluster in range(2) for core in range(4)],
        "lm_head_": ["0:0:0"],
    }
    _MOE_RUNTIME_FIELDS = (("num_expert_workers", 24, int),)

    def _ensure_moe_npu_backends(self, kwargs: dict[str, Any]) -> None:
        for prefix in self._MOE_BACKEND_PREFIXES:
            if hasattr(self, f"{prefix}npu_backend"):
                continue
            backend_kwargs = {
                f"{prefix}{field_name}": kwargs[f"{prefix}{field_name}"]
                for field_name in self._MOE_BACKEND_FIELDS
                if f"{prefix}{field_name}" in kwargs
            }
            backend_kwargs.setdefault(f"{prefix}core_mode", "single")
            has_targets = any(
                backend_kwargs.get(f"{prefix}{field_name}") is not None
                for field_name in ("dev_no", "target_cores", "target_clusters")
            )
            if not has_targets:
                backend_kwargs[f"{prefix}target_cores"] = list(self._MOE_DEFAULT_TARGET_CORES[prefix])
            _normalize_npu_target_kwargs(backend_kwargs, prefix=prefix)
            setattr(self, f"{prefix}npu_backend", MobilintNPUBackend.from_dict(backend_kwargs, prefix=prefix))
        # Drop the role NPU keys from ``kwargs`` so ``PretrainedConfig.__init__`` does not re-apply them via
        # setattr — see :func:`_pop_consumed_backend_kwargs`.
        for prefix in self._MOE_BACKEND_PREFIXES:
            _pop_consumed_backend_kwargs(kwargs, prefix=prefix)

    def _ensure_moe_runtime_fields(self, kwargs: dict[str, Any]) -> None:
        for field_name, default_value, _annotation in self._MOE_RUNTIME_FIELDS:
            if hasattr(self, field_name):
                continue
            value = kwargs.pop(field_name, default_value)
            self.__dict__[field_name] = MobilintEagle3ConfigMixin._coerce_positive_runtime_int(
                field_name,
                value,
                default_value=default_value,
            )
        if "moe_mxq_pack_index" not in self.__dict__:
            pack_index = kwargs.pop("moe_mxq_pack_index", None)
            self.__dict__["moe_mxq_pack_index"] = str(pack_index) if pack_index else None
        if "moe_shared_output_order" not in self.__dict__:
            self.__dict__["moe_shared_output_order"] = self._coerce_moe_shared_output_order(
                kwargs.pop("moe_shared_output_order", None)
            )

    @classmethod
    def _coerce_moe_shared_output_order(cls, value: Any) -> Optional[dict[str, list[str]]]:
        """Validate ``moe_shared_output_order`` and return a JSON-safe copy.

        ``None`` means "not shipped"; the backend then detects the order from the MXQ quantization scales. A
        malformed value raises: it is a packaging bug in a published artifact, and a wrong order corrupts the
        output silently.
        """
        if value is None:
            return None
        if not isinstance(value, dict) or not value:
            raise ValueError(f"moe_shared_output_order must be a non-empty dict of role lists, got {value!r}")
        expected = sorted(cls.MOE_SHARED_OUTPUT_ROLES)
        result: dict[str, list[str]] = {}
        for key, roles in value.items():
            key = str(key)
            if key != "default" and not key.isdigit():
                raise ValueError(f"moe_shared_output_order keys must be 'default' or a layer index, got {key!r}")
            if not isinstance(roles, (list, tuple)) or sorted(roles) != expected:
                raise ValueError(
                    f"moe_shared_output_order[{key!r}] must be a permutation of "
                    f"{list(cls.MOE_SHARED_OUTPUT_ROLES)}, got {roles!r}"
                )
            result[key] = list(roles)
        return result

    def get_moe_shared_output_indices(self, layer_idx: int) -> Optional[tuple[int, int, int]]:
        """Return the ``(residual, expert_in, gate)`` output indices of ``shared[layer_idx]``.

        Returns ``None`` when the config ships no ``moe_shared_output_order`` entry for this layer (neither a
        per-layer override nor ``"default"``).
        """
        order = self.moe_shared_output_order
        if not order:
            return None
        roles = order.get(str(layer_idx), order.get("default"))
        if roles is None:
            return None
        residual, expert_in, gate = (roles.index(role) for role in self.MOE_SHARED_OUTPUT_ROLES)
        return residual, expert_in, gate

    @property
    def max_batch_size(self) -> int:
        """MoE releases run one sequence at a time: every shared MXQ holds a single KV cache entry."""
        return 1

    @max_batch_size.setter
    def max_batch_size(self, value: int) -> None:
        if int(value) > 1:
            raise ValueError(f"Mixture-of-Experts releases support only max_batch_size=1, got {value}.")

    def __init__(self, **kwargs):
        self._ensure_moe_npu_backends(kwargs)
        self._ensure_moe_runtime_fields(kwargs)
        super().__init__(**kwargs)

    def __post_init__(self, **kwargs: Any) -> None:
        self._ensure_moe_npu_backends(kwargs)
        self._ensure_moe_runtime_fields(kwargs)
        super().__post_init__(**kwargs)

    @property
    def shared_mxq_path(self) -> str:
        return self.shared_npu_backend.mxq_path

    @shared_mxq_path.setter
    def shared_mxq_path(self, value: str) -> None:
        self.shared_npu_backend.mxq_path = value

    @property
    def shared_target_device(self) -> str:
        return self.shared_npu_backend.target_device

    @shared_target_device.setter
    def shared_target_device(self, value: str) -> None:
        self.shared_npu_backend = _rebuild_backend_for_target_device(
            self.shared_npu_backend, value, prefix="shared_"
        )

    @property
    def shared_dev_no(self) -> Union[int, list[int]]:
        return self.shared_npu_backend.dev_no

    @shared_dev_no.setter
    def shared_dev_no(self, value: Union[int, list[int]]) -> None:
        self.shared_npu_backend.dev_no = value

    @property
    def shared_core_mode(self) -> str:
        return self.shared_npu_backend.core_mode

    @shared_core_mode.setter
    def shared_core_mode(self, value: str) -> None:
        self.shared_npu_backend.core_mode = value

    @property
    def shared_max_batch_size(self) -> int:
        return self.shared_npu_backend.max_batch_size

    @shared_max_batch_size.setter
    def shared_max_batch_size(self, value: int) -> None:
        self.shared_npu_backend.max_batch_size = max(1, value)

    @property
    def shared_target_cores(self) -> list[str]:
        return _serialized_target_cores(self.shared_npu_backend)

    @shared_target_cores.setter
    def shared_target_cores(self, values: list[str]) -> None:
        self.shared_npu_backend.target_cores = values

    @property
    def shared_target_clusters(self) -> list[str]:
        return _serialized_target_clusters(self.shared_npu_backend)

    @shared_target_clusters.setter
    def shared_target_clusters(self, values: list[str]) -> None:
        self.shared_npu_backend.target_clusters = values

    @property
    def expert_mxq_path(self) -> str:
        return self.expert_npu_backend.mxq_path

    @expert_mxq_path.setter
    def expert_mxq_path(self, value: str) -> None:
        self.expert_npu_backend.mxq_path = value

    @property
    def expert_target_device(self) -> str:
        return self.expert_npu_backend.target_device

    @expert_target_device.setter
    def expert_target_device(self, value: str) -> None:
        self.expert_npu_backend = _rebuild_backend_for_target_device(
            self.expert_npu_backend, value, prefix="expert_"
        )

    @property
    def expert_dev_no(self) -> Union[int, list[int]]:
        return self.expert_npu_backend.dev_no

    @expert_dev_no.setter
    def expert_dev_no(self, value: Union[int, list[int]]) -> None:
        self.expert_npu_backend.dev_no = value

    @property
    def expert_core_mode(self) -> str:
        return self.expert_npu_backend.core_mode

    @expert_core_mode.setter
    def expert_core_mode(self, value: str) -> None:
        self.expert_npu_backend.core_mode = value

    @property
    def expert_max_batch_size(self) -> int:
        return self.expert_npu_backend.max_batch_size

    @expert_max_batch_size.setter
    def expert_max_batch_size(self, value: int) -> None:
        self.expert_npu_backend.max_batch_size = max(1, value)

    @property
    def expert_target_cores(self) -> list[str]:
        return _serialized_target_cores(self.expert_npu_backend)

    @expert_target_cores.setter
    def expert_target_cores(self, values: list[str]) -> None:
        self.expert_npu_backend.target_cores = values

    @property
    def expert_target_clusters(self) -> list[str]:
        return _serialized_target_clusters(self.expert_npu_backend)

    @expert_target_clusters.setter
    def expert_target_clusters(self, values: list[str]) -> None:
        self.expert_npu_backend.target_clusters = values

    @property
    def lm_head_mxq_path(self) -> str:
        return self.lm_head_npu_backend.mxq_path

    @lm_head_mxq_path.setter
    def lm_head_mxq_path(self, value: str) -> None:
        self.lm_head_npu_backend.mxq_path = value

    @property
    def lm_head_target_device(self) -> str:
        return self.lm_head_npu_backend.target_device

    @lm_head_target_device.setter
    def lm_head_target_device(self, value: str) -> None:
        self.lm_head_npu_backend = _rebuild_backend_for_target_device(
            self.lm_head_npu_backend, value, prefix="lm_head_"
        )

    @property
    def lm_head_dev_no(self) -> Union[int, list[int]]:
        return self.lm_head_npu_backend.dev_no

    @lm_head_dev_no.setter
    def lm_head_dev_no(self, value: Union[int, list[int]]) -> None:
        self.lm_head_npu_backend.dev_no = value

    @property
    def lm_head_core_mode(self) -> str:
        return self.lm_head_npu_backend.core_mode

    @lm_head_core_mode.setter
    def lm_head_core_mode(self, value: str) -> None:
        self.lm_head_npu_backend.core_mode = value

    @property
    def lm_head_max_batch_size(self) -> int:
        return self.lm_head_npu_backend.max_batch_size

    @lm_head_max_batch_size.setter
    def lm_head_max_batch_size(self, value: int) -> None:
        self.lm_head_npu_backend.max_batch_size = max(1, value)

    @property
    def lm_head_target_cores(self) -> list[str]:
        return _serialized_target_cores(self.lm_head_npu_backend)

    @lm_head_target_cores.setter
    def lm_head_target_cores(self, values: list[str]) -> None:
        self.lm_head_npu_backend.target_cores = values

    @property
    def lm_head_target_clusters(self) -> list[str]:
        return _serialized_target_clusters(self.lm_head_npu_backend)

    @lm_head_target_clusters.setter
    def lm_head_target_clusters(self, values: list[str]) -> None:
        self.lm_head_npu_backend.target_clusters = values

    @classmethod
    def from_dict(
        cls: type[SpecificPretrainedConfigType], config_dict: dict[str, Any], **kwargs
    ) -> Union[
        "MobilintMixtureOfExpertsConfigMixin",
        tuple["MobilintMixtureOfExpertsConfigMixin", dict[str, Any]],
    ]:
        """Buffer ``shared_*`` / ``expert_*`` / ``lm_head_*`` NPU kwargs across HF's ``from_dict``.

        Mirrors :meth:`MobilintEagle3ConfigMixin.from_dict` for the three MoE role backends.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        role_subs = {
            prefix: _split_npu_backend_kwargs(kwargs, prefix=prefix)
            for prefix in MobilintMixtureOfExpertsConfigMixin._MOE_BACKEND_PREFIXES
        }

        config, unused_kwargs = super().from_dict(
            config_dict, return_unused_kwargs=True, **kwargs
        )  # type: ignore[misc]

        for prefix, sub in role_subs.items():
            _apply_npu_backend_kwargs(
                config,
                sub,
                prefix=prefix,
                resolve_backend=lambda c, p=prefix: getattr(c, f"{p}npu_backend"),
            )

        if return_unused_kwargs:
            return config, unused_kwargs
        return config

    def _remove_keys_not_serialized(self, d: dict[str, Any]) -> None:
        for prefix in self._MOE_BACKEND_PREFIXES:
            _ = d.pop(f"{prefix}npu_backend", None)
        super()._remove_keys_not_serialized(d)

    def to_dict(self):
        output = super().to_dict()
        for prefix in self._MOE_BACKEND_PREFIXES:
            output.update(getattr(self, f"{prefix}npu_backend").to_dict(prefix=prefix))
        return output
