"""Re-export of upstream ``Qwen3ASRProcessor`` under a Mobilint-friendly name."""

from transformers import AutoProcessor

from ._errors import guard_qwen_asr_import
from .configuration_qwen3_asr import MobilintQwen3ASRConfig

with guard_qwen_asr_import():
    from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

MobilintQwen3ASRProcessor = Qwen3ASRProcessor

# Upstream registers the processor only for its own ``Qwen3ASRConfig``; map the Mobilint config too so
# ``AutoProcessor`` resolves it after ``transformers_mblt.register()`` without Hub remote code.
AutoProcessor.register(MobilintQwen3ASRConfig, MobilintQwen3ASRProcessor, exist_ok=True)

__all__ = ["MobilintQwen3ASRProcessor"]
