"""Proxy exports for Mobilint BERT embedding Hub repositories."""

_MISSING = {
    "transformers_mblt",
    "transformers_mblt.models",
    "transformers_mblt.models.bert",
    "mblt_model_zoo",
    "mblt_model_zoo.hf_transformers",
    "mblt_model_zoo.hf_transformers.models",
    "mblt_model_zoo.hf_transformers.models.bert",
}

try:
    from transformers_mblt.models.bert.configuration_bert import (
        MobilintBertEmbeddingConfig,
    )
    from transformers_mblt.models.bert.modeling_bert import (
        MobilintBertEmbeddingModel,
    )
except ImportError as exc:
    # No transformers-mblt, or a release from before BERT embedding models (0.7.2 and earlier).
    if isinstance(exc, ModuleNotFoundError) and exc.name not in _MISSING:
        raise
    # Hub repositories are shared with mblt-model-zoo, which exposes the same classes under its legacy path.
    try:
        from mblt_model_zoo.hf_transformers.models.bert.configuration_bert import (
            MobilintBertEmbeddingConfig,
        )
        from mblt_model_zoo.hf_transformers.models.bert.modeling_bert import (
            MobilintBertEmbeddingModel,
        )
    except ImportError as legacy_exc:
        if isinstance(legacy_exc, ModuleNotFoundError) and legacy_exc.name not in _MISSING:
            raise
        raise ImportError(
            "This model requires transformers-mblt 0.7.3 or later, which adds BERT embedding models. Please run: "
            "pip install -U 'transformers-mblt>=0.7.3'"
        ) from legacy_exc


__all__ = ["MobilintBertEmbeddingConfig", "MobilintBertEmbeddingModel"]
