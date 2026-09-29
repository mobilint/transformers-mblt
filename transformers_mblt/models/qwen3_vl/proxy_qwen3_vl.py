"""Remote-code proxy exports for Mobilint Qwen3-VL models."""

from packaging.version import Version

try:
    from transformers import __version__ as transformers_version
except ImportError as exc:
    raise ImportError(
        "Mobilint Qwen3-VL models require 'transformers>=4.57.0'. "
        'Please install or upgrade with: pip install -U transformers-mblt'
    ) from exc

_MIN_TRANSFORMERS_VERSION = Version("4.57.0")


def _ensure_supported_transformers_version() -> None:
    """Raise an informative error when upstream Qwen3-VL support is unavailable."""
    if Version(transformers_version) < _MIN_TRANSFORMERS_VERSION:
        raise ImportError(
            "Mobilint Qwen3-VL models require 'transformers>=4.57.0'. "
            f"Found transformers=={transformers_version}. "
            'Please upgrade transformers or reinstall with: pip install -U transformers-mblt'
        )


_ensure_supported_transformers_version()

try:
    from transformers_mblt.models.qwen3_vl.configuration_qwen3_vl import (
        MobilintQwen3VLConfig,
    )
    from transformers_mblt.models.qwen3_vl.modeling_qwen3_vl import (
        MobilintQwen3VLForConditionalGeneration,
    )
    from transformers_mblt.models.qwen3_vl.processing_qwen3_vl import (
        MobilintQwen3VLProcessor,
    )
except ModuleNotFoundError as exc:
    if exc.name != "transformers_mblt":
        raise
    # Hub repositories are shared with mblt-model-zoo, which exposes the same classes under its legacy path.
    try:
        from mblt_model_zoo.hf_transformers.models.qwen3_vl.configuration_qwen3_vl import (
            MobilintQwen3VLConfig,
        )
        from mblt_model_zoo.hf_transformers.models.qwen3_vl.modeling_qwen3_vl import (
            MobilintQwen3VLForConditionalGeneration,
        )
        from mblt_model_zoo.hf_transformers.models.qwen3_vl.processing_qwen3_vl import (
            MobilintQwen3VLProcessor,
        )
    except ModuleNotFoundError as legacy_exc:
        if legacy_exc.name not in {"mblt_model_zoo", "mblt_model_zoo.hf_transformers"}:
            raise
        raise ImportError(
            "This model requires 'transformers-mblt' to be installed. "
            "Please run: pip install transformers-mblt"
        ) from legacy_exc

__all__ = ["MobilintQwen3VLConfig", "MobilintQwen3VLForConditionalGeneration", "MobilintQwen3VLProcessor"]
