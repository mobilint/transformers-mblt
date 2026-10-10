"""Qwen3-VL's own single-row llm_forward binds MobilintDeepStackCache rows like the shared LLM core."""

from __future__ import annotations

import numpy as np
import torch

from tests.transformers.image_text_to_text.qwen3_vl_compat import (
    skip_if_transformers_lacks_qwen3_vl_support,
)

skip_if_transformers_lacks_qwen3_vl_support()

from tests.transformers.image_text_to_text.test_qwen3_vl_logits_to_keep import (  # noqa: E402
    _BareQwen3VLTextModel,
)
from transformers_mblt.utils.cache_utils import MobilintDeepStackCache  # noqa: E402


class _TokenKvMxq:
    """Slot KV is the token ids written so far; logits encode the visible prefix."""

    def __init__(self) -> None:
        self.kv: list[int] = []
        self.calls: list[str] = []

    def get_model_output_shape(self):
        return [(1, 1, 3)]

    def dump_cache_memory(self, cache_id: int = 0) -> list[bytes]:
        self.calls.append("dump")
        return [bytes(self.kv)]

    def load_cache_memory(self, buffer: list[bytes], cache_id: int = 0) -> None:
        self.calls.append("load")
        self.kv = list(buffer[0])

    def infer(self, inputs, _extra, cache_size):
        tokens = [int(v) for v in np.asarray(inputs[0])[0, :, 0]]
        self.kv = self.kv[: int(cache_size)] + tokens
        visible = self.kv
        return [np.array([[[len(visible), sum(visible), sum(i * t for i, t in enumerate(visible))]]], dtype=np.float32)]


def _forward(model, cache, tokens: list[int]) -> list[float]:
    start = 0 if cache is None else cache.get_seq_length()
    embeds = torch.tensor(tokens, dtype=torch.float32).reshape(1, len(tokens), 1)
    logits = model.llm_forward(
        inputs_embeds=embeds,
        deepstack_visual_embeds=None,
        visual_pos_masks=None,
        past_key_values=cache,
        cache_position=torch.arange(start, start + len(tokens)),
        npu_prefill_chunk_size=64,
    )
    return logits.reshape(-1).tolist()


def _cache(model) -> MobilintDeepStackCache:
    return MobilintDeepStackCache(model.npu_backend.mxq_models, num_deepstack_layers=0, hidden_size=1)


def test_single_row_qwen3_vl_resumes_each_cache_on_its_own_kv() -> None:
    model = _BareQwen3VLTextModel(_TokenKvMxq(), hidden_size=1)
    a, b = _cache(model), _cache(model)

    _forward(model, a, [1, 2, 3])
    _forward(model, b, [7, 8])
    _forward(model, None, [9, 9, 9, 9])

    assert _forward(model, a, [4]) == [4.0, 10.0, 20.0]
    assert _forward(model, b, [5]) == [3.0, 20.0, 18.0]
