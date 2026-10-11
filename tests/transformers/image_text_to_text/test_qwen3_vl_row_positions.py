"""Qwen3-VL gives each row of a ``MobilintCache.from_rows`` batch its own M-RoPE text positions.

Upstream builds one start for the whole batch, which is right only while every row holds the same number of tokens.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.transformers.image_text_to_text.qwen3_vl_compat import transformers_supports_qwen3_vl
from transformers_mblt.utils.cache_utils import MobilintCache

if not transformers_supports_qwen3_vl():
    pytest.skip("transformers lacks Qwen3-VL", allow_module_level=True)

from transformers_mblt.models.qwen3_vl.modeling_qwen3_vl import (  # noqa: E402
    MobilintQwen3VLForConditionalGeneration,
    _row_text_position_ids,
)


class _Mxq:
    """Just enough of a qbruntime Model for caches that never bind."""


class _Backend:
    def __init__(self, k: int = 2) -> None:
        self.mxq_models = [_Mxq()]
        self.k_per_model = k


def _request(backend: _Backend, slot: tuple[int, int]) -> MobilintCache:
    return MobilintCache(backend.mxq_models, per_model_batch=backend.k_per_model, slots=[slot])


def test_qwen3_vl_text_positions_continue_each_row_from_its_own_length() -> None:
    backend = _Backend(k=2)
    a, b = _request(backend, (0, 0)), _request(backend, (0, 1))
    a.set_seq_length(10)
    b.set_seq_length(4)
    a.rope_deltas = torch.tensor([[-6]])  # a's prompt had an image
    batch = MobilintCache.from_rows([a, b])

    right_padded = _row_text_position_ids(batch, torch.tensor([[1, 0, 0], [1, 1, 1]]), 2, 3)
    left_padded = _row_text_position_ids(batch, torch.tensor([[0, 0, 1], [1, 1, 1]]), 2, 3)
    with_history = _row_text_position_ids(batch, torch.ones(2, 13, dtype=torch.long), 2, 3)  # mask covers the cache

    assert right_padded[0, 0].item() == 4 and right_padded[1].tolist() == [4, 5, 6]
    assert left_padded[0, 2].item() == 4 and left_padded[1].tolist() == [4, 5, 6]
    assert with_history.tolist() == [[4, 5, 6], [4, 5, 6]]
    b.set_seq_length(10)
    assert _row_text_position_ids(MobilintCache.from_rows([a, b]), None, 2, 3) is None  # rows agree
    assert _row_text_position_ids(a, None, 1, 3) is None  # one row


def _qwen3_vl_wrapper():
    """The public Qwen3-VL model around a stub inner model that records what it was called with."""
    wrapper = MobilintQwen3VLForConditionalGeneration.__new__(MobilintQwen3VLForConditionalGeneration)
    torch.nn.Module.__init__(wrapper)
    calls = []

    class _Inner:
        rope_deltas = None

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


def test_qwen3_vl_model_gives_mixed_length_rows_their_own_positions() -> None:
    backend = _Backend(k=2)
    a, b = _request(backend, (0, 0)), _request(backend, (0, 1))
    a.set_seq_length(10)
    b.set_seq_length(4)
    a.rope_deltas = torch.tensor([[-6]])
    wrapper, calls = _qwen3_vl_wrapper()

    wrapper(
        inputs_embeds=torch.zeros(2, 2, 1),
        attention_mask=torch.tensor([[1, 0], [1, 1]]),
        past_key_values=MobilintCache.from_rows([a, b]),
        logits_to_keep=1,
    )

    position_ids = calls[0]["position_ids"]
    assert tuple(position_ids.shape) == (3, 2, 2)
    assert position_ids[:, 0, 0].tolist() == [4, 4, 4] and position_ids[:, 1].tolist() == [[4, 5]] * 3


def test_qwen3_vl_model_asks_for_positions_when_mixed_rows_carry_images() -> None:
    backend = _Backend(k=2)
    a, b = _request(backend, (0, 0)), _request(backend, (0, 1))
    a.set_seq_length(10)
    wrapper, _ = _qwen3_vl_wrapper()

    with pytest.raises(ValueError, match="pass position_ids"):
        wrapper(
            inputs_embeds=torch.zeros(2, 2, 1),
            past_key_values=MobilintCache.from_rows([a, b]),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        )
