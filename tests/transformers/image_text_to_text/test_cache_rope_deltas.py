"""M-RoPE ``rope_deltas`` live on the cache, so a conversation that resumes after another one keeps its own offset.

Upstream Qwen-VL keeps ``rope_deltas`` on ``self.model``: the last sequence to run sets it, and every cache that
continues afterwards adds that offset to its text positions.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from transformers_mblt.models.qwen2_vl.modeling_qwen2_vl import MobilintQwen2VLForConditionalGeneration
from transformers_mblt.models.qwen3_vl.modeling_qwen3_vl import MobilintQwen3VLForConditionalGeneration
from transformers_mblt.utils.cache_utils import (
    MobilintCache,
    MobilintDeepStackCache,
    MobilintWhisperCache,
    use_cache_rope_deltas,
)

_PROMPT_DELTAS = {5: -3, 7: -11}  # prompt length -> offset its image layout produces


class _PromptModel:
    """``self.model`` double: a prompt on an empty cache computes its offset, a continuation reads ``rope_deltas``."""

    def __init__(self) -> None:
        self.rope_deltas: torch.Tensor | None = None
        self.seen: list[int | None] = []

    def get_rope_index(self, input_ids, **kwargs):
        del kwargs
        delta = _PROMPT_DELTAS[int(input_ids.shape[1])]
        positions = torch.arange(input_ids.shape[1]).view(1, 1, -1).expand(3, 1, -1) + delta
        return positions, torch.tensor([[delta]])

    def __call__(self, *, input_ids, past_key_values=None, **kwargs):
        del kwargs
        if past_key_values is None:  # like the text model, build a cache and return it
            past_key_values = MobilintCache(object())
        if past_key_values.get_seq_length() == 0:
            _, self.rope_deltas = self.get_rope_index(input_ids)
        self.seen.append(None if self.rope_deltas is None else int(self.rope_deltas))
        past_key_values.update_seen_tokens(past_key_values.get_seq_length() + int(input_ids.shape[1]))
        return SimpleNamespace(
            last_hidden_state=torch.zeros(1, int(input_ids.shape[1]), 4),
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
            rope_deltas=self.rope_deltas,
        )


def _wrapper(cls):
    wrapper = cls.__new__(cls)
    torch.nn.Module.__init__(wrapper)
    wrapper.config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=4), return_dict=True)
    wrapper.model = _PromptModel()
    wrapper.lm_head = torch.nn.Identity()
    return wrapper


def _ids(length: int) -> torch.Tensor:
    return torch.arange(length).unsqueeze(0)


@pytest.mark.parametrize("cls", [MobilintQwen2VLForConditionalGeneration, MobilintQwen3VLForConditionalGeneration])
def test_forward_resumes_a_cache_with_its_own_offset(cls) -> None:
    wrapper = _wrapper(cls)
    a, b = MobilintCache(object()), MobilintCache(object())

    wrapper.forward(input_ids=_ids(5), past_key_values=a)
    wrapper.forward(input_ids=_ids(7), past_key_values=b)
    wrapper.forward(input_ids=_ids(1), past_key_values=a)  # upstream alone would add b's -11
    wrapper.forward(input_ids=_ids(1), past_key_values=b)

    assert wrapper.model.seen == [-3, -11, -3, -11]
    assert a.rope_deltas.tolist() == [[-3]] and b.rope_deltas.tolist() == [[-11]]


@pytest.mark.parametrize("cls", [MobilintQwen2VLForConditionalGeneration, MobilintQwen3VLForConditionalGeneration])
def test_generate_continues_a_cache_with_its_own_offset(cls) -> None:
    wrapper = _wrapper(cls)
    multimodal = {"mm_token_type_ids": torch.zeros(1, 1), "image_grid_thw": torch.ones(1, 3)}
    a, b = MobilintCache(object()), MobilintCache(object())

    wrapper._prepare_position_ids_for_generation(_ids(5), {"past_key_values": a, **multimodal})
    a.update_seen_tokens(5)
    wrapper._prepare_position_ids_for_generation(_ids(7), {"past_key_values": b, **multimodal})
    b.update_seen_tokens(7)

    # Second turn on a: the full conversation is passed again, the cache already holds the first 5 tokens.
    position_ids = wrapper._prepare_position_ids_for_generation(_ids(6), {"past_key_values": a})

    text_positions = torch.arange(6 + 5)  # upstream counts the cache length on top
    assert position_ids[0, 0].tolist() == (text_positions - 3).tolist()  # a's offset, not b's -11
    assert a.rope_deltas.tolist() == [[-3]] and b.rope_deltas.tolist() == [[-11]]


def test_generate_without_a_cache_hands_the_prompt_offset_to_the_cache_it_builds() -> None:
    wrapper = _wrapper(MobilintQwen2VLForConditionalGeneration)
    multimodal = {"mm_token_type_ids": torch.zeros(1, 1), "image_grid_thw": torch.ones(1, 3)}
    wrapper._prepare_position_ids_for_generation(_ids(5), dict(multimodal))  # generate builds the cache afterwards
    cache = MobilintCache(object())

    wrapper.forward(input_ids=_ids(5), past_key_values=cache, position_ids=torch.zeros(3, 1, 5, dtype=torch.long))

    assert cache.rope_deltas.tolist() == [[-3]]


def test_a_filled_cache_without_an_offset_does_not_borrow_another_one() -> None:
    wrapper = _wrapper(MobilintQwen2VLForConditionalGeneration)
    wrapper.model.rope_deltas = torch.tensor([[-11]])  # left by another conversation
    restored = MobilintCache(object())
    restored.update_seen_tokens(5)  # e.g. KV loaded from an older release's files

    wrapper.forward(input_ids=_ids(1), past_key_values=restored)

    assert wrapper.model.seen == [None]


def test_a_later_filled_row_without_an_offset_does_not_borrow_another_one() -> None:
    wrapper = _wrapper(MobilintQwen2VLForConditionalGeneration)
    wrapper.model.rope_deltas = torch.tensor([[-11], [-11]])  # left by another conversation
    restored = MobilintCache(object(), per_model_batch=2)
    restored.update_seen_tokens({1: 5})  # row 0 empty, row 1 filled with no recorded offset

    use_cache_rope_deltas(wrapper.model, restored)

    assert wrapper.model.rope_deltas is None


@pytest.mark.parametrize(
    "make",
    [
        lambda: MobilintCache(object()),
        lambda: MobilintDeepStackCache(object(), num_deepstack_layers=1, hidden_size=2),
        lambda: MobilintWhisperCache(object()),
    ],
)
def test_copy_and_reset_carry_the_offset(make) -> None:
    cache = make()
    cache.rope_deltas = torch.tensor([[-3]])

    copied = cache.copy()
    cache.rope_deltas.add_(1)
    assert copied.rope_deltas.tolist() == [[-3]]

    copied.reset()
    assert copied.rope_deltas is None


class _FileMxq:
    def dump_cache_memory_to(self, cache_dir, cache_id=0):
        del cache_dir, cache_id

    def load_cache_memory_from(self, cache_dir, cache_id=0):
        del cache_dir, cache_id


@pytest.mark.parametrize("cls", [MobilintQwen2VLForConditionalGeneration, MobilintQwen3VLForConditionalGeneration])
def test_a_forward_without_a_cache_keeps_the_offset_on_the_cache_it_returns(cls) -> None:
    wrapper = _wrapper(cls)

    built = wrapper.forward(input_ids=_ids(5)).past_key_values
    wrapper.forward(input_ids=_ids(7), past_key_values=MobilintCache(object()))
    wrapper.forward(input_ids=_ids(1), past_key_values=built)

    assert built.rope_deltas.tolist() == [[-3]] and wrapper.model.seen[-1] == -3


def test_batched_files_keep_each_row_offset(tmp_path) -> None:
    cache = MobilintCache(_FileMxq(), per_model_batch=2)
    cache.update_seen_tokens({0: 5, 1: 7})
    cache.rope_deltas = torch.tensor([[-3], [-11]])
    for row in (0, 1):
        (tmp_path / str(row)).mkdir()
        cache.dump_cache_memory_to(str(tmp_path / str(row)), index=row)

    restored = MobilintCache(_FileMxq(), per_model_batch=2)
    for row in (0, 1):
        restored.load_cache_memory_from(str(tmp_path / str(row)), index=row)

    assert restored.rope_deltas.tolist() == [[-3], [-11]]


def test_a_row_restored_next_to_a_filled_row_of_unknown_offset_recomputes(tmp_path) -> None:
    cache = MobilintCache(_FileMxq(), per_model_batch=2)
    cache.update_seen_tokens({0: 5})
    cache.rope_deltas = torch.tensor([[-3], [0]])
    cache.dump_cache_memory_to(str(tmp_path), index=0)

    restored = MobilintCache(_FileMxq(), per_model_batch=2)
    restored.update_seen_tokens({1: 4})  # row 1 holds a sequence whose offset nobody recorded
    restored.load_cache_memory_from(str(tmp_path), index=0)

    assert restored.rope_deltas is None


def test_restoring_an_empty_row_keeps_the_offset_of_a_filled_one(tmp_path) -> None:
    cache = MobilintCache(_FileMxq(), per_model_batch=2)
    cache.update_seen_tokens({0: 5})
    cache.rope_deltas = torch.tensor([[-3], [0]])
    for row in (0, 1):
        (tmp_path / str(row)).mkdir()
        cache.dump_cache_memory_to(str(tmp_path / str(row)), index=row)
    (tmp_path / "1" / "rope_deltas.json").unlink()  # an empty row saved without an offset

    restored = MobilintCache(_FileMxq(), per_model_batch=2)
    for row in (0, 1):
        restored.load_cache_memory_from(str(tmp_path / str(row)), index=row)

    assert restored.rope_deltas.tolist() == [[-3]]  # the empty row 1 is past the active batch


def test_rows_past_the_active_batch_are_left_out_on_restore(tmp_path) -> None:
    cache = MobilintCache(_FileMxq(), per_model_batch=3)
    cache.update_seen_tokens({0: 5})
    cache.rope_deltas = torch.tensor([[-3]])  # one active row in a cache with room for three
    for row in range(3):
        (tmp_path / str(row)).mkdir()
        cache.dump_cache_memory_to(str(tmp_path / str(row)), index=row)

    restored = MobilintCache(_FileMxq(), per_model_batch=3)
    for row in range(3):
        restored.load_cache_memory_from(str(tmp_path / str(row)), index=row)

    assert restored.rope_deltas.tolist() == [[-3]]  # sized to the active batch, not the cache's capacity

    wrapper = _wrapper(MobilintQwen2VLForConditionalGeneration)
    wrapper.forward(input_ids=_ids(1), past_key_values=restored)  # a one-row continuation
    assert wrapper.model.seen == [-3]


def test_files_keep_the_offset(tmp_path) -> None:
    cache = MobilintCache(_FileMxq())
    cache.update_seen_tokens(5)
    cache.rope_deltas = torch.tensor([[-3]])
    cache.dump_cache_memory_to(str(tmp_path))

    restored = MobilintCache(_FileMxq())
    restored.load_cache_memory_from(str(tmp_path))

    assert restored.get_seq_length() == 5 and restored.rope_deltas.tolist() == [[-3]]
