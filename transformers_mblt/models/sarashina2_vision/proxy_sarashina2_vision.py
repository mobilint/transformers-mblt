"""Remote-code exports for the Mobilint Sarashina2.2-Vision release."""
try:
    from transformers_mblt.models.sarashina2_vision import (
        MobilintSarashina2VisionConfig,
        MobilintSarashina2VisionForCausalLM,
        MobilintSarashina2VisionProcessor,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"transformers_mblt", "transformers_mblt.models",
                        "transformers_mblt.models.sarashina2_vision"}:
        raise
    try:
        from mblt_model_zoo.hf_transformers.models.sarashina2_vision import (
            MobilintSarashina2VisionConfig,
            MobilintSarashina2VisionForCausalLM,
            MobilintSarashina2VisionProcessor,
        )
    except ModuleNotFoundError as legacy_exc:
        if legacy_exc.name not in {"mblt_model_zoo", "mblt_model_zoo.hf_transformers",
                                   "mblt_model_zoo.hf_transformers.models",
                                   "mblt_model_zoo.hf_transformers.models.sarashina2_vision"}:
            raise
        raise ImportError("Sarashina2-Vision requires a transformers-mblt release with Sarashina support. "
                          "Install the updated package with: pip install -U transformers-mblt") from legacy_exc

__all__ = ["MobilintSarashina2VisionConfig", "MobilintSarashina2VisionForCausalLM", "MobilintSarashina2VisionProcessor"]
