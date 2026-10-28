"""Tests for the fused top-k / top-p logits warper."""

from __future__ import annotations

import pytest
import torch
from transformers import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    MinPLogitsWarper,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from transformers_mblt.utils.generation_utils import MobilintGenerationMixin
from transformers_mblt.utils.sampling_utils import (
    MobilintTopKTopPLogitsWarper,
    MobilintTopKTopPSampler,
    fuse_top_k_top_p,
    sample_in_top_k_slice,
)

VOCAB = 151_936  # Qwen3


def _upstream(top_k: TopKLogitsWarper, top_p: TopPLogitsWarper, scores: torch.Tensor) -> torch.Tensor:
    return top_p(None, top_k(None, scores.clone()))


def _fused(top_k: TopKLogitsWarper, top_p: TopPLogitsWarper, scores: torch.Tensor) -> torch.Tensor:
    return MobilintTopKTopPLogitsWarper(top_k, top_p)(None, scores.clone())


@pytest.mark.parametrize(
    ("k", "p", "min_keep", "batch"),
    [(20, 0.95, 1, 1), (20, 0.8, 1, 3), (50, 0.95, 1, 2), (5, 0.3, 1, 1), (20, 0.5, 3, 2), (1, 0.9, 1, 1)],
)
def test_fused_scores_equal_upstream(k: int, p: float, min_keep: int, batch: int) -> None:
    generator = torch.Generator().manual_seed(k * 1000 + batch)
    for _ in range(5):
        scores = TemperatureLogitsWarper(0.6)(None, torch.randn(batch, VOCAB, generator=generator) * 4)
        top_k = TopKLogitsWarper(k, min_tokens_to_keep=min_keep)
        top_p = TopPLogitsWarper(p, min_tokens_to_keep=min_keep)
        assert torch.equal(_fused(top_k, top_p, scores), _upstream(top_k, top_p, scores))


def test_custom_filter_values_match_upstream() -> None:
    scores = torch.randn(2, 1000, generator=torch.Generator().manual_seed(0)) * 3
    top_k = TopKLogitsWarper(10, filter_value=-1e4)
    top_p = TopPLogitsWarper(0.7, filter_value=-2e4)
    assert torch.equal(_fused(top_k, top_p, scores), _upstream(top_k, top_p, scores))


def test_tie_at_kth_value_keeps_every_tied_entry() -> None:
    scores = torch.randn(1, 1000, generator=torch.Generator().manual_seed(1))
    threshold = torch.topk(scores, 20).values[0, -1]
    scores[0, :5] = threshold  # five extra entries equal to the k-th value: upstream keeps all of them
    top_k, top_p = TopKLogitsWarper(20), TopPLogitsWarper(0.999)
    out = _fused(top_k, top_p, scores)
    assert torch.equal(out, _upstream(top_k, top_p, scores))
    assert int(torch.isfinite(out).sum()) == 25


def test_batch_rows_with_different_tie_counts_match_upstream() -> None:
    scores = torch.randn(3, 1000, generator=torch.Generator().manual_seed(3))
    for row, extra in ((0, 0), (1, 4), (2, 9)):
        threshold = torch.topk(scores[row], 20).values[-1]
        scores[row, torch.topk(scores[row], extra, largest=False).indices] = threshold
    top_k, top_p = TopKLogitsWarper(20), TopPLogitsWarper(0.999)
    out = _fused(top_k, top_p, scores)
    assert torch.equal(out, _upstream(top_k, top_p, scores))
    assert torch.isfinite(out).sum(dim=-1).tolist() == [20, 24, 29]


def _kept_values(scores: torch.Tensor) -> list[list[float]]:
    return [sorted(row[torch.isfinite(row)].tolist()) for row in scores]


@pytest.mark.parametrize("p", [0.95, 0.8])
def test_npu_like_quantized_logits_keep_the_same_values(p: float) -> None:
    """Coarse logits tie inside the top 20 on most NPU decode steps; which of several equal entries survives at the
    top-p boundary is sort-order dependent upstream too, so the kept values are compared as a multiset."""
    generator = torch.Generator().manual_seed(4)
    top_k, top_p = TopKLogitsWarper(20), TopPLogitsWarper(p)
    for _ in range(20):
        scores = TemperatureLogitsWarper(0.6)(None, (torch.randn(2, VOCAB, generator=generator) * 4 * 4).round() / 4)
        assert _kept_values(_fused(top_k, top_p, scores)) == _kept_values(_upstream(top_k, top_p, scores))


def test_fewer_finite_entries_than_k_match_upstream() -> None:
    scores = torch.full((1, 1000), -float("inf"))
    scores[0, [3, 70, 500]] = torch.tensor([1.0, 2.0, 0.5])
    top_k, top_p = TopKLogitsWarper(20), TopPLogitsWarper(0.95)
    assert torch.equal(_fused(top_k, top_p, scores), _upstream(top_k, top_p, scores))


def test_seeded_sampling_draws_the_same_tokens() -> None:
    scores = torch.randn(1, VOCAB, generator=torch.Generator().manual_seed(2)) * 3
    top_k, top_p = TopKLogitsWarper(20), TopPLogitsWarper(0.95)
    draws = []
    for warp in (_fused, _upstream):
        torch.manual_seed(7)
        probs = warp(top_k, top_p, scores).softmax(dim=-1)
        draws.append(torch.multinomial(probs, 50, replacement=True))
    assert torch.equal(draws[0], draws[1])


def test_fuse_replaces_only_an_adjacent_exact_pair() -> None:
    temperature, top_k, top_p, min_p = (
        TemperatureLogitsWarper(0.7),
        TopKLogitsWarper(20),
        TopPLogitsWarper(0.9),
        MinPLogitsWarper(0.05),
    )
    processors = fuse_top_k_top_p(LogitsProcessorList([temperature, top_k, top_p, min_p]))
    assert [type(x) for x in processors] == [TemperatureLogitsWarper, MobilintTopKTopPLogitsWarper, MinPLogitsWarper]
    assert processors[1].top_k_warper is top_k and processors[1].top_p_warper is top_p

    assert list(fuse_top_k_top_p(LogitsProcessorList([top_p, top_k]))) == [top_p, top_k]
    assert list(fuse_top_k_top_p(LogitsProcessorList([top_k, min_p, top_p]))) == [top_k, min_p, top_p]

    class CustomTopP(TopPLogitsWarper):
        pass

    custom = CustomTopP(0.9)
    assert list(fuse_top_k_top_p(LogitsProcessorList([top_k, custom]))) == [top_k, custom]


class _Generator(MobilintGenerationMixin):
    """Bare mixin instance; only ``_get_logits_processor`` is exercised."""


def test_mixin_fuses_the_upstream_list(monkeypatch: pytest.MonkeyPatch) -> None:
    def upstream(self, *args, **kwargs):
        return LogitsProcessorList([TemperatureLogitsWarper(0.6), TopKLogitsWarper(20), TopPLogitsWarper(0.95)])

    monkeypatch.setattr(GenerationMixin, "_get_logits_processor", upstream)
    monkeypatch.delenv("MBLT_FUSED_TOPK_TOPP", raising=False)
    fused = _Generator()._get_logits_processor(None)
    assert [type(x) for x in fused] == [TemperatureLogitsWarper, MobilintTopKTopPLogitsWarper]

    monkeypatch.setenv("MBLT_FUSED_TOPK_TOPP", "0")
    kept = _Generator()._get_logits_processor(None)
    assert [type(x) for x in kept] == [TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper]


@pytest.mark.parametrize("quantized", [False, True])
def test_sampler_draws_from_the_upstream_distribution(quantized: bool) -> None:
    torch.manual_seed(5)
    scores = torch.randn(1, VOCAB) * 3
    if quantized:
        scores = (scores * 4).round() / 4
    scores = TemperatureLogitsWarper(0.6)(None, scores)
    top_k, top_p = TopKLogitsWarper(20), TopPLogitsWarper(0.95)
    expected = _upstream(top_k, top_p, scores).softmax(dim=-1)[0]
    sampler = MobilintTopKTopPSampler(top_k, top_p)
    counts = torch.zeros(VOCAB)
    draws = 6000
    for _ in range(draws):
        out = sampler(None, scores)
        assert int(torch.isfinite(out).sum()) == 1
        counts[out.argmax()] += 1
    assert (counts[expected == 0] == 0).all()
    assert 0.5 * (counts / draws - expected).abs().sum().item() < 0.03


def test_sampler_falls_back_when_the_slice_would_be_too_wide() -> None:
    scores = torch.full((2, 1000), -float("inf"))
    scores[:, [3, 70, 500]] = torch.tensor([1.0, 2.0, 0.5])
    out = MobilintTopKTopPSampler(TopKLogitsWarper(20), TopPLogitsWarper(0.95))(None, scores)
    assert torch.isfinite(out).sum(dim=-1).tolist() == [1, 1]
    assert set(out.argmax(dim=-1).tolist()) <= {3, 70, 500}


def _fused_list() -> LogitsProcessorList:
    return LogitsProcessorList(
        [TemperatureLogitsWarper(0.6), MobilintTopKTopPLogitsWarper(TopKLogitsWarper(20), TopPLogitsWarper(0.95))]
    )


def test_sample_in_top_k_slice_swaps_only_when_safe() -> None:
    config = GenerationConfig(do_sample=True, temperature=0.6, top_k=20, top_p=0.95)
    processors = _fused_list()
    swapped, greedy = sample_in_top_k_slice(processors, config)
    assert type(swapped[-1]) is MobilintTopKTopPSampler and swapped[0] is processors[0]
    assert swapped[-1].top_k_warper is processors[-1].top_k_warper
    assert greedy.do_sample is False and config.do_sample is True  # the caller's config is untouched
    assert type(processors[-1]) is MobilintTopKTopPLogitsWarper

    assert sample_in_top_k_slice(processors, GenerationConfig(do_sample=False)) is None
    assert sample_in_top_k_slice(LogitsProcessorList([*processors, MinPLogitsWarper(0.05)]), config) is None
    assert sample_in_top_k_slice(LogitsProcessorList(), config) is None
    returns_scores = GenerationConfig(do_sample=True, top_k=20, return_dict_in_generate=True, output_scores=True)
    assert sample_in_top_k_slice(processors, returns_scores) is None


def test_mixin_sample_passes_the_swapped_processors(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def upstream(self, input_ids, logits_processor, stopping_criteria, generation_config, *args, **kwargs):
        seen.update(processors=logits_processor, config=generation_config, args=args, kwargs=kwargs)
        return "result"

    monkeypatch.setattr(GenerationMixin, "_sample", upstream)
    monkeypatch.delenv("MBLT_FUSED_TOPK_TOPP", raising=False)
    config = GenerationConfig(do_sample=True, temperature=0.6, top_k=20, top_p=0.95)
    result = _Generator()._sample(
        None, logits_processor=_fused_list(), stopping_criteria=None, generation_config=config, synced_gpus=False
    )
    assert result == "result" and seen["kwargs"] == {"synced_gpus": False}
    assert type(seen["processors"][-1]) is MobilintTopKTopPSampler and seen["config"].do_sample is False

    monkeypatch.setenv("MBLT_FUSED_TOPK_TOPP", "0")
    processors = _fused_list()
    _Generator()._sample(None, processors, None, config, False, None)
    assert seen["processors"] is processors and seen["config"] is config and seen["args"] == (False, None)
