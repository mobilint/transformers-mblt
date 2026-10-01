try:
    from transformers_mblt.models.qwen2_eagle3.configuration_qwen2_eagle3 import (
        MobilintQwen2Eagle3Config,
    )
    from transformers_mblt.models.qwen2_eagle3.modeling_qwen2_eagle3 import (
        MobilintQwen2Eagle3ForCausalLM,
    )
except ModuleNotFoundError as exc:
    if exc.name != "transformers_mblt":
        raise
    # Hub repositories are shared with mblt-model-zoo, which exposes the same classes under its legacy path.
    try:
        from mblt_model_zoo.hf_transformers.models.qwen2_eagle3.configuration_qwen2_eagle3 import (
            MobilintQwen2Eagle3Config,
        )
        from mblt_model_zoo.hf_transformers.models.qwen2_eagle3.modeling_qwen2_eagle3 import (
            MobilintQwen2Eagle3ForCausalLM,
        )
    except ModuleNotFoundError as legacy_exc:
        if legacy_exc.name not in {"mblt_model_zoo", "mblt_model_zoo.hf_transformers"}:
            raise
        raise ImportError(
            "This model requires 'transformers-mblt' to be installed. "
            "Please run: pip install transformers-mblt"
        ) from legacy_exc


__all__ = ["MobilintQwen2Eagle3Config", "MobilintQwen2Eagle3ForCausalLM"]
