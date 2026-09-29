try:
    from transformers_mblt.models.exaone4.configuration_exaone4 import (
        MobilintExaone4Config,
    )
    from transformers_mblt.models.exaone4.modeling_exaone4 import (
        MobilintExaone4ForCausalLM,
    )
except ModuleNotFoundError as exc:
    if exc.name != "transformers_mblt":
        raise
    # Hub repositories are shared with mblt-model-zoo, which exposes the same classes under its legacy path.
    try:
        from mblt_model_zoo.hf_transformers.models.exaone4.configuration_exaone4 import (
            MobilintExaone4Config,
        )
        from mblt_model_zoo.hf_transformers.models.exaone4.modeling_exaone4 import (
            MobilintExaone4ForCausalLM,
        )
    except ModuleNotFoundError as legacy_exc:
        if legacy_exc.name not in {"mblt_model_zoo", "mblt_model_zoo.hf_transformers"}:
            raise
        raise ImportError(
            "This model requires 'transformers-mblt' to be installed. "
            "Please run: pip install transformers-mblt"
        ) from legacy_exc

__all__ = ["MobilintExaone4Config", "MobilintExaone4ForCausalLM"]