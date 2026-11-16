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


def _missing(exc: ImportError) -> bool:
    """A missing package, or an older release without these classes; any other import failure is re-raised."""
    if isinstance(exc, ModuleNotFoundError):
        return exc.name in _MISSING
    return "cannot import name" in str(exc)


try:
    from transformers_mblt.models.xlm_roberta.configuration_xlm_roberta import (
        MobilintXLMRobertaConfig,
    )
    from transformers_mblt.models.xlm_roberta.modeling_xlm_roberta import (
        MobilintXLMRobertaForSequenceClassification,
    )
except ImportError as exc:
    # No transformers-mblt, or a release from before XLM-RoBERTa (0.7.2 and earlier).
    if not _missing(exc):
        raise
    # Hub repositories are shared with mblt-model-zoo, which exposes the same classes under its legacy path.
    try:
        from mblt_model_zoo.hf_transformers.models.xlm_roberta.configuration_xlm_roberta import (
            MobilintXLMRobertaConfig,
        )
        from mblt_model_zoo.hf_transformers.models.xlm_roberta.modeling_xlm_roberta import (
            MobilintXLMRobertaForSequenceClassification,
        )
    except ImportError as legacy_exc:
        if not _missing(legacy_exc):
            raise
        raise ImportError(
            "This model requires transformers-mblt 0.7.3 or later, which adds XLM-RoBERTa. Please run: "
            "pip install -U 'transformers-mblt>=0.7.3'"
        ) from legacy_exc


__all__ = ["MobilintXLMRobertaConfig", "MobilintXLMRobertaForSequenceClassification"]
