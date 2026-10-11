"""Qwen-VL refuses to guess M-RoPE positions for a ``MobilintCache.from_rows`` batch that holds tokens."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.transformers.image_text_to_text.qwen3_vl_compat import transformers_supports_qwen3_vl
from transformers_mblt.models.qwen2_vl.modeling_qwen2_vl import MobilintQwen2VLForConditionalGeneration
from transformers_mblt.utils.cache_utils import MobilintCache

if transformers_supports_qwen3_vl():
    from transformers_mblt.models.qwen3_vl.modeling_qwen3_vl import MobilintQwen3VLForConditionalGeneration

    _QWEN3_VL = pytest.param(MobilintQwen3VLForConditionalGeneration, id="qwen3_vl")
else:
    _QWEN3_VL = pytest.param(None, id="qwen3_vl", marks=pytest.mark.skip(reason="transformers lacks Qwen3-VL"))
_VL_CLASSES = [pytest.param(MobilintQwen2VLForConditionalGeneration, id="qwen2_vl"), _QWEN3_VL]


class _Mxq:
    """Just enough of a qbruntime Model for caches that never bind."""


def _joined_batch(rows: int = 2) -> MobilintCache:
    models = [_Mxq()]
    caches = [MobilintCache(models, per_model_batch=2, slots=[(0, row)]) for row in range(rows)]
    for cache, length in zip(caches, (10, 4)):
        cache.set_seq_length(length)
    return MobilintCache.from_rows(caches)


def _wrapper(cls, uses_rope_input: bool):
    """The public VL model around a stub inner model that records what it was called with."""
    wrapper = cls.__new__(cls)
    torch.nn.Module.__init__(wrapper)
    calls = []

    class _Inner:
        rope_deltas = None
        language_model = SimpleNamespace(_uses_rope_input=uses_rope_input)

        def __call__(self, **kwargs):
            calls.append(kwargs)
            batch, length = kwargs["inputs_embeds"].shape[:2]
            return SimpleNamespace(
                last_hidden_state=torch.zeros(batch, length, 2), past_key_values=kwargs["past_key_values"]
            )

    wrapper.__dict__["model"] = _Inner()
    wrapper.lm_head = torch.nn.Identity()
    wrapper.config = SimpleNamespace(use_return_dict=True, return_dict=True)
    return wrapper, calls


@pytest.mark.parametrize(
    ("cls", "rows"),
    [
        pytest.param(MobilintQwen2VLForConditionalGeneration, 1, id="qwen2_vl"),
        pytest.param(*_QWEN3_VL.values, 2, id="qwen3_vl", marks=_QWEN3_VL.marks),
    ],
)
def test_joined_batch_needs_position_ids_on_an_external_rope_mxq(cls, rows) -> None:
    wrapper, calls = _wrapper(cls, uses_rope_input=True)
    batch = _joined_batch(rows)

    with pytest.raises(ValueError, match="pass position_ids"):
        wrapper(inputs_embeds=torch.zeros(rows, 1, 1), past_key_values=batch)

    positions = torch.tensor([[[4], [7]][:rows]] * 3)
    wrapper(inputs_embeds=torch.zeros(rows, 1, 1), past_key_values=batch, position_ids=positions)
    assert calls[-1]["position_ids"] is positions


def test_qwen2_vl_external_rope_mxq_refuses_a_multi_row_joined_batch() -> None:
    wrapper, _ = _wrapper(MobilintQwen2VLForConditionalGeneration, uses_rope_input=True)

    with pytest.raises(NotImplementedError, match="one sequence at a time"):
        wrapper(
            inputs_embeds=torch.zeros(2, 1, 1),
            past_key_values=_joined_batch(2),
            position_ids=torch.zeros(3, 2, 1, dtype=torch.long),
        )


@pytest.mark.parametrize("cls", _VL_CLASSES)
def test_joined_batch_runs_without_positions_on_a_static_mxq(cls) -> None:
    wrapper, calls = _wrapper(cls, uses_rope_input=False)

    wrapper(inputs_embeds=torch.zeros(2, 1, 1), past_key_values=_joined_batch())

    assert calls[-1].get("position_ids") is None
