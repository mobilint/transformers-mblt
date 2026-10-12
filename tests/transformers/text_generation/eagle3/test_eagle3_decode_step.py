"""Unit tests for driving EAGLE-3 one round at a time (``decode_step``, ``eagle3_prefill``, ``eagle3_decode_step``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from transformers_mblt.models.qwen2_eagle3.modeling_qwen2_eagle3 import MobilintQwen2Eagle3ForCausalLM
from transformers_mblt.utils.cache_utils import MobilintEagle3Cache
from transformers_mblt.utils.eagle3 import decoding as decoding_module
from transformers_mblt.utils.eagle3.tree_decoding import decode_step

VOCAB = 16


class _Cache:
    """The cache state a round reads and writes, without MXQs."""

    def __init__(self, base_seq_length: int) -> None:
        self.base_seq_length = base_seq_length
        self.accept_tokens = None
        self.unfed_tokens = None
        self.fed_token_ids = None
        self.pending_draft_tokens = None
        self.retrieve_indices = None
        self.tree_mask = None
        self.tree_position_ids = None

    def get_base_seq_length(self) -> int:
        return self.base_seq_length

    def update_base_seen_tokens(self, num_tokens: int) -> None:
        self.base_seq_length += int(num_tokens)

    def clear_tree_state(self) -> None:
        self.accept_tokens = None
        self.pending_draft_tokens = None
        self.retrieve_indices = None
        self.tree_mask = None
        self.tree_position_ids = None


class _Base:
    """Base model whose greedy pick at each fed position is ``next_ids[i]``."""

    def __init__(self, next_ids: list[int]) -> None:
        self.next_ids = next_ids
        self.calls: list[dict] = []

    def __call__(self, input_ids, *, cache, position_ids=None, **kwargs):
        del cache, kwargs
        self.calls.append({"input_ids": input_ids.clone(), "position_ids": position_ids})
        logits = torch.zeros((1, input_ids.shape[1], VOCAB))
        for position, token in enumerate(self.next_ids[: input_ids.shape[1]]):
            logits[0, position, token] = 10.0
        hidden = torch.arange(input_ids.shape[1], dtype=torch.float32).reshape(1, -1, 1)
        return {"hidden_states": [hidden]}, logits


class _Draft:
    """Draft model that proposes ``root -> root + 1 -> root + 2`` after each sampled root."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def topk_generate(self, hidden_states, *, input_ids, cache, input_ids_delta, max_draft_tokens, **kwargs):
        del kwargs
        self.calls.append(
            {"hidden": hidden_states.clone(), "delta": input_ids_delta.clone(), "max_draft_tokens": max_draft_tokens}
        )
        root = int(input_ids[0, -1])
        draft_tokens = torch.tensor([[root, root + 1, root + 2]])
        retrieve_indices = torch.tensor([[0, 1, 2]])
        tree_mask = torch.ones((1, 1, 3, 3))
        tree_position_ids = torch.tensor([0, 1, 2])
        cache.pending_draft_tokens = draft_tokens
        cache.retrieve_indices = retrieve_indices
        cache.tree_mask = tree_mask
        cache.tree_position_ids = tree_position_ids
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids


def _model_with_tree(base_next_ids: list[int], *, base_seq_length: int = 4):
    """A model and a cache holding the draft tree ``5 -> 6 -> 7`` over ``base_seq_length`` cached tokens."""
    model = SimpleNamespace(eagle3_base_model=_Base(base_next_ids), eagle3_draft_model=_Draft())
    cache = _Cache(base_seq_length)
    cache.pending_draft_tokens = torch.tensor([[5, 6, 7]])
    cache.retrieve_indices = torch.tensor([[0, 1, 2]])
    cache.tree_mask = torch.ones((1, 1, 3, 3))
    cache.tree_position_ids = torch.tensor([0, 1, 2])
    return model, cache


def test_decode_step_accepts_the_whole_path_and_samples_the_next_root() -> None:
    model, cache = _model_with_tree([6, 7, 9])

    step = decode_step(model, cache, None)

    assert torch.equal(step.tokens, torch.tensor([[5, 6, 7]]))
    assert torch.equal(step.next_token, torch.tensor([[9]]))
    assert not step.stopped
    assert step.candidate_draft_tokens == 2
    base_call = model.eagle3_base_model.calls[0]
    assert torch.equal(base_call["input_ids"], torch.tensor([[5, 6, 7]]))
    # The tree sits right after the cached base tokens.
    assert torch.equal(base_call["position_ids"], torch.tensor([[4, 5, 6]]))
    # The draft reads each accepted token after the first, then the new root.
    assert torch.equal(model.eagle3_draft_model.calls[0]["delta"], torch.tensor([[6, 7, 9]]))
    # The accepted tokens are fed to the base with the next round; the next tree waits in the cache.
    assert torch.equal(cache.accept_tokens, torch.tensor([[5, 6, 7]]))
    assert torch.equal(cache.pending_draft_tokens, torch.tensor([[9, 10, 11]]))


def test_decode_step_stops_accepting_at_the_first_rejected_draft() -> None:
    model, cache = _model_with_tree([6, 8, 0])

    step = decode_step(model, cache, None)

    assert torch.equal(step.tokens, torch.tensor([[5, 6]]))
    assert torch.equal(step.next_token, torch.tensor([[8]]))


def test_consecutive_decode_steps_feed_the_accepted_tokens_first() -> None:
    model, cache = _model_with_tree([6, 7, 9])
    decode_step(model, cache, None)
    model.eagle3_base_model.next_ids = [0, 0, 0, 10, 11, 1]

    step = decode_step(model, cache, None)

    base_call = model.eagle3_base_model.calls[1]
    assert torch.equal(base_call["input_ids"], torch.tensor([[5, 6, 7, 9, 10, 11]]))
    assert torch.equal(base_call["position_ids"], torch.tensor([[4, 5, 6, 7, 8, 9]]))
    assert cache.get_base_seq_length() == 7
    assert torch.equal(step.tokens, torch.tensor([[9, 10, 11]]))
    assert torch.equal(step.next_token, torch.tensor([[1]]))


def test_decode_step_that_ends_generation_keeps_the_tokens_unfed_and_drops_the_tree() -> None:
    model, cache = _model_with_tree([6, 7, 9])

    step = decode_step(model, cache, None, remaining_tokens=2)

    assert torch.equal(step.tokens, torch.tensor([[5, 6]]))
    assert step.next_token is None
    assert step.stopped
    assert torch.equal(cache.unfed_tokens, torch.tensor([[5, 6]]))
    assert cache.pending_draft_tokens is None
    assert model.eagle3_draft_model.calls == []
    with pytest.raises(RuntimeError, match="no draft tree"):
        decode_step(model, cache, None)


def test_decode_step_stops_at_eos() -> None:
    model, cache = _model_with_tree([6, 7, 9])

    step = decode_step(model, cache, None, eos_token_id=[6])

    assert torch.equal(step.tokens, torch.tensor([[5, 6]]))
    assert step.stopped


class _FakeEagle3Cache(MobilintEagle3Cache):
    """Satisfies the type check without MXQs."""

    def __init__(self) -> None:  # noqa: D107 - no MXQs to bind
        self.cleared = 0
        self.synced = 0

    def clear_tree_state(self) -> None:
        self.cleared += 1

    def sync_draft_seq_length_to_base(self) -> None:
        self.synced += 1


def _eagle3_model() -> MobilintQwen2Eagle3ForCausalLM:
    model = object.__new__(MobilintQwen2Eagle3ForCausalLM)
    draft = SimpleNamespace(max_draft_tokens=None, reset_draft_mask_state=lambda: None)
    object.__setattr__(
        model,
        "_modules",
        {
            "model": SimpleNamespace(
                _modules={"base_model": SimpleNamespace(), "draft_model": draft, "fc_projector": SimpleNamespace()},
                draft_model=draft,
                fc_projector=SimpleNamespace(),
            )
        },
    )
    return model


def test_eagle3_prefill_starts_a_clean_round_on_the_cache_and_returns_the_root(monkeypatch) -> None:
    model = _eagle3_model()
    cache = _FakeEagle3Cache()
    seen: dict = {}

    def _initialize_tree(input_ids, model_arg, cache_arg, logits_processor, *, remaining_tokens, count_npu_time):
        seen.update(input_ids=input_ids, model=model_arg, cache=cache_arg, remaining=remaining_tokens)
        # A clean start: tree state dropped and the draft realigned to the base before feeding.
        assert cache_arg.cleared == 1 and cache_arg.synced == 1
        return torch.tensor([[7, 8, 9]]), None, None, None, None

    monkeypatch.setattr(decoding_module, "initialize_tree", _initialize_tree)

    root = model.eagle3_prefill(torch.tensor([[1, 2, 3]]), cache, max_new_tokens=5)

    assert torch.equal(root, torch.tensor([[7]]))
    assert seen["model"] is model and seen["cache"] is cache and seen["remaining"] == 5
    with pytest.raises(TypeError, match="MobilintEagle3Cache"):
        model.eagle3_prefill(torch.tensor([[1]]), None)


def test_eagle3_decode_step_runs_one_round_on_the_cache(monkeypatch) -> None:
    model = _eagle3_model()
    cache = _FakeEagle3Cache()
    calls: list = []
    monkeypatch.setattr(decoding_module, "decode_step", lambda *args, **kwargs: calls.append((args, kwargs)) or "step")

    result = model.eagle3_decode_step(cache, max_new_tokens=3, eos_token_id=2)

    assert result == "step"
    ((args, kwargs),) = calls
    assert args == (model, cache, None)
    assert kwargs == {"remaining_tokens": 3, "eos_token_id": 2, "count_npu_time": False}


def test_configure_eagle3_tree_sets_the_draft_tree_from_arguments_over_generation_config() -> None:
    model = _eagle3_model()
    model.config = SimpleNamespace(eagle3_tree_depth=None, eagle3_tree_top_k=None)
    model.generation_config = SimpleNamespace(num_assistant_tokens=9, eagle3_tree_depth=4, eagle3_tree_top_k=8)
    shapes: list = []
    model.eagle3_draft_model.set_tree_shape = lambda **kwargs: shapes.append(kwargs)

    model.configure_eagle3_tree()
    model.configure_eagle3_tree(num_assistant_tokens=5, eagle3_tree_top_k=2)

    assert model.eagle3_draft_model.max_draft_tokens == 4
    assert shapes == [{"depth": 4, "top_k": 8}, {"depth": 4, "top_k": 2}]


class _RoundCache(MobilintEagle3Cache):
    """A cache left between rounds: accepted tokens pending and the next tree drafted, without MXQs."""

    def __init__(self) -> None:  # noqa: D107 - no MXQs to bind
        self.accept_tokens = torch.tensor([[5, 6, 7]])
        self.pending_draft_tokens = torch.tensor([[9, 10, 11]])
        self.retrieve_indices = torch.tensor([[0, 1, 2]])
        self.tree_mask = torch.ones((1, 1, 3, 3))
        self.tree_position_ids = torch.tensor([0, 1, 2])
        self.unfed_tokens = None

    def clear_tree_state(self) -> None:
        self.accept_tokens = None
        self.pending_draft_tokens = None
        self.retrieve_indices = None
        self.tree_mask = None
        self.tree_position_ids = None

    def sync_draft_seq_length_to_base(self) -> None:
        pass


def test_eagle3_prefill_after_rounds_feeds_the_tokens_they_returned_first(monkeypatch) -> None:
    """The next turn on a cache left between rounds must not lose the accepted tokens and the next token."""
    model = _eagle3_model()
    cache = _RoundCache()
    seen: dict = {}

    def _initialize_tree(input_ids, model_arg, cache_arg, logits_processor, *, remaining_tokens, count_npu_time):
        seen["unfed"] = cache_arg.unfed_tokens
        seen["tree"] = cache_arg.pending_draft_tokens
        return torch.tensor([[1]]), None, None, None, None

    monkeypatch.setattr(decoding_module, "initialize_tree", _initialize_tree)

    model.eagle3_prefill(torch.tensor([[20, 21]]), cache)

    assert torch.equal(seen["unfed"], torch.tensor([[5, 6, 7, 9]]))
    assert seen["tree"] is None


def test_eagle3_prefill_rejects_a_batch_or_an_empty_budget_without_touching_the_cache(monkeypatch) -> None:
    model = _eagle3_model()
    cache = _RoundCache()
    monkeypatch.setattr(decoding_module, "initialize_tree", lambda *args, **kwargs: pytest.fail("must not run"))

    with pytest.raises(NotImplementedError, match="batch size 1"):
        model.eagle3_prefill(torch.tensor([[1], [2]]), cache)
    with pytest.raises(NotImplementedError, match="batch size 1"):
        model.eagle3_prefill(torch.tensor([1, 2]), cache)
    with pytest.raises(ValueError, match="max_new_tokens must be >= 1"):
        model.eagle3_prefill(torch.tensor([[1]]), cache, max_new_tokens=0)
    assert torch.equal(cache.accept_tokens, torch.tensor([[5, 6, 7]]))
    assert cache.unfed_tokens is None
