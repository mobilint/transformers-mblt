"""NPU backend exports used by the Transformers integration.

The backend implementation is maintained by ``mblt-npu-python`` (:mod:`mblt_npu`). This module adds the
multi-slot ``dispatcher`` property that the Transformers models rely on, without coupling the NPU wheel to
``transformers``.
"""

from mblt_npu import (
    BACKEND_CLASSES,
    DEFAULT_TARGET_DEVICE,
    MobilintAriesBackend,
    MobilintBackendAllocError,
    MobilintNPUBackend,
    MobilintRegulusBackend,
    backend_class_for,
)


def _get_transformers_dispatcher(backend: MobilintNPUBackend):
    """Return the backend's lazily created :class:`MultiSlotDispatcher`."""
    # Attribute name is shared with mblt-model-zoo so a backend keeps one dispatcher when both packages are installed.
    dispatcher = getattr(backend, "_mblt_model_zoo_dispatcher", None)
    if dispatcher is None:
        from .utils.multi_slot_dispatch import MultiSlotDispatcher

        dispatcher = MultiSlotDispatcher(backend)
        backend._mblt_model_zoo_dispatcher = dispatcher
    return dispatcher


# Guarded so installing alongside mblt-model-zoo (which installs the same property) never overrides an existing one.
if not hasattr(MobilintNPUBackend, "dispatcher"):
    MobilintNPUBackend.dispatcher = property(_get_transformers_dispatcher)

__all__ = [
    "BACKEND_CLASSES",
    "DEFAULT_TARGET_DEVICE",
    "MobilintBackendAllocError",
    "MobilintAriesBackend",
    "MobilintNPUBackend",
    "MobilintRegulusBackend",
    "backend_class_for",
]
