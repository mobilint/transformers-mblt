from argparse import Namespace


def register_mobilint_models(args: Namespace, transformers):
    """Register Mobilint model classes using the active CLI trust and revision settings."""
    revision = getattr(args, "model_revision", None)
    trust_remote_code = getattr(args, "trust_remote_code", False)
    # Without remote code, AutoConfig can only resolve `mobilint-*` model types that are registered locally.
    from .._registry import register

    register()
    config = transformers.AutoConfig.from_pretrained(
        args.model_name_or_path_or_address,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )

    model_type = getattr(config, "model_type", "")
    arch_name = config.architectures[0] if getattr(config, "architectures", None) else ""

    if model_type.startswith("mobilint-") or arch_name.startswith("Mobilint"):
        original_model_type = model_type[len("mobilint-") :] if model_type.startswith("mobilint-") else model_type
        module_model_type = original_model_type.replace("-", "_")

        import importlib

        module = importlib.import_module(f"transformers_mblt.models.{module_model_type}.modeling_{module_model_type}")
        # Importing the modeling module already registers its Auto classes. The class injection and task-mapping
        # patch below need a concrete architecture name, so skip them when the config omits `architectures`.
        if not arch_name:
            return
        setattr(transformers, arch_name, module.__dict__[arch_name])

        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = transformers.models.auto.modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = (
            transformers.models.auto.modeling_auto.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
        )

        if arch_name.endswith("CausalLM"):
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[model_type] = arch_name
        elif arch_name.endswith("ConditionalGeneration"):
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES[model_type] = arch_name
