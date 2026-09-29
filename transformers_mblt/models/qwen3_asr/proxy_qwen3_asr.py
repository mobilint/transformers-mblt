from packaging.version import Version

try:
    from transformers import __version__ as transformers_version
except ImportError as exc:
    raise ImportError(
        "Mobilint Qwen3-ASR models require 'transformers>=4.57.0'. "
        'Please install or upgrade with: pip install -U transformers-mblt'
    ) from exc

_MIN_TRANSFORMERS_VERSION = Version("4.57.0")


def _ensure_supported_transformers_version() -> None:
    """Raise an informative error when upstream Qwen3-ASR support is unavailable."""
    if Version(transformers_version) < _MIN_TRANSFORMERS_VERSION:
        raise ImportError(
            "Mobilint Qwen3-ASR models require 'transformers>=4.57.0'. "
            f"Found transformers=={transformers_version}. "
            'Please upgrade transformers or reinstall with: pip install -U transformers-mblt'
        )


_ensure_supported_transformers_version()

try:
    from transformers_mblt.models.qwen3_asr.configuration_qwen3_asr import (
        MobilintQwen3ASRConfig,
    )
    from transformers_mblt.models.qwen3_asr.modeling_qwen3_asr import (
        MobilintQwen3ASRForConditionalGeneration,
    )
    from transformers_mblt.models.qwen3_asr.processing_qwen3_asr import (
        MobilintQwen3ASRProcessor,
    )
except ModuleNotFoundError as exc:
    if exc.name != "transformers_mblt":
        raise
    # Hub repositories are shared with mblt-model-zoo, which exposes the same classes under its legacy path.
    try:
        from mblt_model_zoo.hf_transformers.models.qwen3_asr.configuration_qwen3_asr import (
            MobilintQwen3ASRConfig,
        )
        from mblt_model_zoo.hf_transformers.models.qwen3_asr.modeling_qwen3_asr import (
            MobilintQwen3ASRForConditionalGeneration,
        )
        from mblt_model_zoo.hf_transformers.models.qwen3_asr.processing_qwen3_asr import (
            MobilintQwen3ASRProcessor,
        )
    except ModuleNotFoundError as legacy_exc:
        if legacy_exc.name not in {"mblt_model_zoo", "mblt_model_zoo.hf_transformers"}:
            raise
        raise ImportError(
            "This model requires 'transformers-mblt' to be installed. "
            "Please run: pip install transformers-mblt"
        ) from legacy_exc

__all__ = [
    "MobilintQwen3ASRConfig",
    "MobilintQwen3ASRForConditionalGeneration",
    "MobilintQwen3ASRProcessor",
]
