"""Re-export of upstream ``Qwen3ASRProcessor`` under a Mobilint-friendly name."""

from transformers import AutoProcessor

from .configuration_qwen3_asr import MobilintQwen3ASRConfig

from ._upstream.processing_qwen3_asr import Qwen3ASRProcessor

MobilintQwen3ASRProcessor = Qwen3ASRProcessor

# Upstream registers the processor only for its own ``Qwen3ASRConfig``; map the Mobilint config too so
# ``AutoProcessor`` resolves it after ``transformers_mblt.register()`` without Hub remote code.
AutoProcessor.register(MobilintQwen3ASRConfig, MobilintQwen3ASRProcessor, exist_ok=True)

__all__ = ["MobilintQwen3ASRProcessor"]
