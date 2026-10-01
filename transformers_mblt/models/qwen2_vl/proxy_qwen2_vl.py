try:
    from transformers_mblt.models.qwen2_vl.configuration_qwen2_vl import (
        MobilintQwen2VLConfig,
    )
    from transformers_mblt.models.qwen2_vl.modeling_qwen2_vl import (
        MobilintQwen2VLForConditionalGeneration,
    )
    from transformers_mblt.models.qwen2_vl.processing_qwen2_vl import (
        MobilintQwen2VLProcessor,
    )
except ModuleNotFoundError as exc:
    if exc.name != "transformers_mblt":
        raise
    # Hub repositories are shared with mblt-model-zoo, which exposes the same classes under its legacy path.
    try:
        from mblt_model_zoo.hf_transformers.models.qwen2_vl.configuration_qwen2_vl import (
            MobilintQwen2VLConfig,
        )
        from mblt_model_zoo.hf_transformers.models.qwen2_vl.modeling_qwen2_vl import (
            MobilintQwen2VLForConditionalGeneration,
        )
        from mblt_model_zoo.hf_transformers.models.qwen2_vl.processing_qwen2_vl import (
            MobilintQwen2VLProcessor,
        )
    except ModuleNotFoundError as legacy_exc:
        if legacy_exc.name not in {"mblt_model_zoo", "mblt_model_zoo.hf_transformers"}:
            raise
        raise ImportError(
            "This model requires 'transformers-mblt' to be installed. "
            "Please run: pip install transformers-mblt"
        ) from legacy_exc

__all__ = ["MobilintQwen2VLConfig", "MobilintQwen2VLForConditionalGeneration", "MobilintQwen2VLProcessor"]