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
# Releases before 0.8.0 have these modules without the embedding classes.
_OLD_MODULES = {
    f"{package}.models.bert.{module}"
    for package in ("transformers_mblt", "mblt_model_zoo.hf_transformers")
    for module in ("configuration_bert", "modeling_bert")
}


def _missing(exc: ImportError) -> bool:
    """A missing package, or an older release without these classes; any other import failure is re-raised."""
    if isinstance(exc, ModuleNotFoundError):
        return exc.name in _MISSING
    return exc.name in _OLD_MODULES and "MobilintBertEmbedding" in str(exc)


try:
    from transformers_mblt.models.bert.configuration_bert import (
        MobilintBertEmbeddingConfig,
    )
    from transformers_mblt.models.bert.modeling_bert import (
        MobilintBertEmbeddingModel,
    )
except ImportError as exc:
    # No transformers-mblt, or a release from before BERT embedding models (0.7.2 and earlier).
    if not _missing(exc):
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
        if not _missing(legacy_exc):
            raise
        raise ImportError(
            "This model requires transformers-mblt 0.8.0 or later, which adds BERT embedding models. Please run: "
            "pip install -U 'transformers-mblt>=0.8.0'"
        ) from legacy_exc


__all__ = ["MobilintBertEmbeddingConfig", "MobilintBertEmbeddingModel"]
