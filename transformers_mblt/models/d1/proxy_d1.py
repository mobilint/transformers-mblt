"""Proxy exports for the Mobilint D1 Hub repository."""

_MISSING = {
    "transformers_mblt",
    "transformers_mblt.models",
    "transformers_mblt.models.d1",
    "mblt_model_zoo",
    "mblt_model_zoo.hf_transformers",
    "mblt_model_zoo.hf_transformers.models",
    "mblt_model_zoo.hf_transformers.models.d1",
}

try:
    from transformers_mblt.models.d1.configuration_d1 import (
        MobilintD1Config,
    )
    from transformers_mblt.models.d1.modeling_d1 import (
        MobilintD1Model,
    )
except ModuleNotFoundError as exc:
    # No transformers-mblt, or a release from before D1 (0.7.2 and earlier).
    if exc.name not in _MISSING:
        raise
    # Hub repositories are shared with mblt-model-zoo, which exposes the same classes under its legacy path.
    try:
        from mblt_model_zoo.hf_transformers.models.d1.configuration_d1 import (
            MobilintD1Config,
        )
        from mblt_model_zoo.hf_transformers.models.d1.modeling_d1 import (
            MobilintD1Model,
        )
    except ModuleNotFoundError as legacy_exc:
        if legacy_exc.name not in _MISSING:
            raise
        raise ImportError(
            "This model requires transformers-mblt 0.7.3 or later, which adds D1. Please run: "
            "pip install -U 'transformers-mblt>=0.7.3'"
        ) from legacy_exc


__all__ = ["MobilintD1Config", "MobilintD1Model"]
