"""Proxy exports for the Mobilint D1 Hub repository."""

try:
    from transformers_mblt.models.d1.configuration_d1 import (
        MobilintD1Config,
    )
    from transformers_mblt.models.d1.modeling_d1 import (
        MobilintD1Model,
    )
except ModuleNotFoundError as exc:
    if exc.name != "transformers_mblt":
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
        if legacy_exc.name not in {
            "mblt_model_zoo",
            "mblt_model_zoo.hf_transformers",
            "mblt_model_zoo.hf_transformers.models.d1",
        }:
            raise
        raise ImportError(
            "This model requires 'transformers-mblt' to be installed. Please run: pip install transformers-mblt"
        ) from legacy_exc


__all__ = ["MobilintD1Config", "MobilintD1Model"]
