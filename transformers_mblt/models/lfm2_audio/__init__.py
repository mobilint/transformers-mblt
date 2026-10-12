from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configuration_lfm2_audio import (
        MobilintLfm2AudioBackboneConfig,
        MobilintLfm2AudioConfig,
        MobilintLfm2AudioDepthformerConfig,
        MobilintLfm2AudioEncoderConfig,
    )
    from .modeling_lfm2_audio import MobilintLfm2AudioForConditionalGeneration
    from .processing_lfm2_audio import MobilintLfm2AudioFeatureExtractor, MobilintLfm2AudioProcessor


_CONFIGURATION_NAMES = {
    "MobilintLfm2AudioConfig",
    "MobilintLfm2AudioEncoderConfig",
    "MobilintLfm2AudioBackboneConfig",
    "MobilintLfm2AudioDepthformerConfig",
}
_PROCESSING_NAMES = {"MobilintLfm2AudioFeatureExtractor", "MobilintLfm2AudioProcessor"}


def __getattr__(name: str):
    import importlib

    if name in _CONFIGURATION_NAMES:
        module = importlib.import_module(".configuration_lfm2_audio", __package__)
        return getattr(module, name)

    if name == "MobilintLfm2AudioForConditionalGeneration":
        module = importlib.import_module(".modeling_lfm2_audio", __package__)
        return module.MobilintLfm2AudioForConditionalGeneration

    if name in _PROCESSING_NAMES:
        module = importlib.import_module(".processing_lfm2_audio", __package__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MobilintLfm2AudioConfig",
    "MobilintLfm2AudioEncoderConfig",
    "MobilintLfm2AudioBackboneConfig",
    "MobilintLfm2AudioDepthformerConfig",
    "MobilintLfm2AudioForConditionalGeneration",
    "MobilintLfm2AudioFeatureExtractor",
    "MobilintLfm2AudioProcessor",
]
