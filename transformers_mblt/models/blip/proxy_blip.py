try:
    from transformers_mblt.models.blip.configuration_blip import (
        MobilintBlipConfig,
    )
    from transformers_mblt.models.blip.modeling_blip import (
        MobilintBlipForConditionalGeneration,
    )
except ModuleNotFoundError as exc:
    if exc.name != "transformers_mblt":
        raise
    # Hub repositories are shared with mblt-model-zoo, which exposes the same classes under its legacy path.
    try:
        from mblt_model_zoo.hf_transformers.models.blip.configuration_blip import (
            MobilintBlipConfig,
        )
        from mblt_model_zoo.hf_transformers.models.blip.modeling_blip import (
            MobilintBlipForConditionalGeneration,
        )
    except ModuleNotFoundError as legacy_exc:
        if legacy_exc.name not in {"mblt_model_zoo", "mblt_model_zoo.hf_transformers"}:
            raise
        raise ImportError(
            "This model requires 'transformers-mblt' to be installed. "
            "Please run: pip install transformers-mblt"
        ) from legacy_exc

__all__ = ["MobilintBlipConfig", "MobilintBlipForConditionalGeneration"]
