"""Proxy exports for Mobilint XLM-RoBERTa Hub repositories."""

_MISSING = {
    "transformers_mblt",
    "transformers_mblt.models",
    "transformers_mblt.models.xlm_roberta",
    "mblt_model_zoo",
    "mblt_model_zoo.hf_transformers",
    "mblt_model_zoo.hf_transformers.models",
    "mblt_model_zoo.hf_transformers.models.xlm_roberta",
}


try:
    from transformers_mblt.models.xlm_roberta.configuration_xlm_roberta import (
        MobilintXLMRobertaConfig,
    )
    from transformers_mblt.models.xlm_roberta.modeling_xlm_roberta import (
        MobilintXLMRobertaForSequenceClassification,
    )
except ModuleNotFoundError as exc:
    # No transformers-mblt, or a release from before XLM-RoBERTa (0.7.2 and earlier).
    if exc.name not in _MISSING:
        raise
    # Hub repositories are shared with mblt-model-zoo, which exposes the same classes under its legacy path.
    try:
        from mblt_model_zoo.hf_transformers.models.xlm_roberta.configuration_xlm_roberta import (
            MobilintXLMRobertaConfig,
        )
        from mblt_model_zoo.hf_transformers.models.xlm_roberta.modeling_xlm_roberta import (
            MobilintXLMRobertaForSequenceClassification,
        )
    except ModuleNotFoundError as legacy_exc:
        if legacy_exc.name not in _MISSING:
            raise
        raise ImportError(
            "This model requires transformers-mblt 0.8.0 or later, which adds XLM-RoBERTa. Please run: "
            "pip install -U 'transformers-mblt>=0.8.0'"
        ) from legacy_exc


__all__ = ["MobilintXLMRobertaConfig", "MobilintXLMRobertaForSequenceClassification"]
