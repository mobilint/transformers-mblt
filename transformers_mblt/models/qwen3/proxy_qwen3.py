try:
    from transformers_mblt.models.qwen3.configuration_qwen3 import (
        MobilintQwen3Config,
    )
    from transformers_mblt.models.qwen3.modeling_qwen3 import (
        MobilintQwen3ForCausalLM,
    )
except ModuleNotFoundError as exc:
    if exc.name != "transformers_mblt":
        raise
    # Hub repositories are shared with mblt-model-zoo, which exposes the same classes under its legacy path.
    try:
        from mblt_model_zoo.hf_transformers.models.qwen3.configuration_qwen3 import (
            MobilintQwen3Config,
        )
        from mblt_model_zoo.hf_transformers.models.qwen3.modeling_qwen3 import (
            MobilintQwen3ForCausalLM,
        )
    except ModuleNotFoundError as legacy_exc:
        if legacy_exc.name not in {"mblt_model_zoo", "mblt_model_zoo.hf_transformers"}:
            raise
        raise ImportError(
            "This model requires 'transformers-mblt' to be installed. "
            "Please run: pip install transformers-mblt"
        ) from legacy_exc

__all__ = ["MobilintQwen3Config", "MobilintQwen3ForCausalLM"]