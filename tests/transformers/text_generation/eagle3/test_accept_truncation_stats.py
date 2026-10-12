"""Unit tests for EAGLE-3 acceptance accounting under EOS and budget truncation.

The decode loop must record acceptance from the tokens each round emits, so
tokens ``decode_step`` drops (EOS in-candidate, remaining budget) never
inflate ``acceptance_tokens_sum`` or downstream ratios.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.generation.stopping_criteria import StoppingCriteriaList

from transformers_mblt.utils.generation_utils import MobilintEagle3GenerationMixin


class _FakeCache:
    """Minimal ``MobilintEagle3Cache`` stand-in for the decode loop."""

    def __init__(self) -> None:
        self.accept_tokens: torch.Tensor | None = None


def _make_fake_self() -> SimpleNamespace:
    """Build a ``self`` stub sufficient for ``_run_eagle3_decode_loop``."""
    fake = SimpleNamespace()
    fake._eagle3_stopping_scores_adapter = MobilintEagle3GenerationMixin._eagle3_stopping_scores_adapter
    return fake


def _run_loop(monkeypatch: pytest.MonkeyPatch, decode_rounds, *, emit_lengths, should_stops, max_tokens: int):
    """Run the loop on rounds that emit ``emit_lengths[i]`` tokens: what ``decode_step`` returns after EOS and budget
    truncation. Every round's tree offers 4 draft tokens."""
    from transformers_mblt.utils.eagle3 import decoding as decoding_module

    monkeypatch.setattr(decoding_module, "initialize_tree", lambda *_args, **_kwargs: None)
    decode_rounds(
        *[([99] * emit, stop) for emit, stop in zip(emit_lengths, should_stops)],
        candidate_draft_tokens=4,
    )
    fake_self = _make_fake_self()
    generated = torch.tensor([[1, 2]], dtype=torch.long)
    result = MobilintEagle3GenerationMixin._run_eagle3_decode_loop(
        fake_self,
        generated=generated,
        cache=_FakeCache(),
        logits_processor=None,
        max_tokens=max_tokens,
        eos_token_id=None,
        stopping_criteria_list=StoppingCriteriaList(),
        streamer=None,
        count_npu_time=False,
    )
    return result, fake_self._eagle3_acceptance_stats


def test_eagle3_decode_loop_records_full_iteration_without_truncation(monkeypatch, decode_rounds):
    """Normal iter: emit 4 tokens (3 drafts + 1 root); tally the 3 accepted drafts."""
    _, stats = _run_loop(
        monkeypatch,
        decode_rounds,
        emit_lengths=[4],
        should_stops=[True],
        max_tokens=64,
    )

    assert stats["steps"] == 1
    assert stats["accepted_tokens_sum"] == 3
    assert stats["accepted_tokens_avg"] == pytest.approx(3.0)
    # Each round's tree offers 4 draft tokens.
    assert stats["acceptance_ratio"] == pytest.approx(3.0 / 4.0)


def test_eagle3_decode_loop_records_eos_truncated_iteration(monkeypatch, decode_rounds):
    """EOS shrinks emit to 2 even though posterior accepted 3 drafts."""
    _, stats = _run_loop(
        monkeypatch,
        decode_rounds,
        emit_lengths=[2],
        should_stops=[True],
        max_tokens=64,
    )

    assert stats["steps"] == 1
    assert stats["accepted_tokens_sum"] == 1  # emit=2 → drafts = emit-1 = 1
    assert stats["accepted_tokens_avg"] == pytest.approx(1.0)
    assert stats["acceptance_ratio"] == pytest.approx(1.0 / 4.0)


def test_eagle3_decode_loop_records_budget_truncated_iteration(monkeypatch, decode_rounds):
    """Remaining-token budget shrinks emit to 2 even though posterior accepted 4 drafts."""
    _, stats = _run_loop(
        monkeypatch,
        decode_rounds,
        emit_lengths=[2],
        should_stops=[True],
        max_tokens=2,
    )

    assert stats["steps"] == 1
    assert stats["accepted_tokens_sum"] == 1
    assert stats["accepted_tokens_avg"] == pytest.approx(1.0)
    assert stats["acceptance_ratio"] == pytest.approx(1.0 / 4.0)


def test_eagle3_decode_loop_handles_zero_emission_iteration(monkeypatch, decode_rounds):
    """Empty emission (no accepted candidate survived truncation): step counts, tokens = 0."""
    _, stats = _run_loop(
        monkeypatch,
        decode_rounds,
        emit_lengths=[0],
        should_stops=[True],
        max_tokens=64,
    )

    assert stats["steps"] == 1
    assert stats["accepted_tokens_sum"] == 0
    assert stats["accepted_tokens_avg"] == pytest.approx(0.0)
    assert stats["acceptance_ratio"] == pytest.approx(0.0)


def test_eagle3_decode_loop_averages_across_normal_then_truncated(monkeypatch, decode_rounds):
    """Mixed run: full iter then a budget-truncated final iter averages correctly."""
    _, stats = _run_loop(
        monkeypatch,
        decode_rounds,
        emit_lengths=[4, 2],  # iter 0 emits 4 (3 drafts), iter 1 emits 2 (1 draft) — budget hit
        should_stops=[False, True],
        max_tokens=6,
    )

    assert stats["steps"] == 2
    assert stats["accepted_tokens_sum"] == 4  # 3 + 1
    assert stats["accepted_tokens_avg"] == pytest.approx(2.0)
    assert stats["acceptance_ratio"] == pytest.approx((3.0 / 4.0 + 1.0 / 4.0) / 2.0)
