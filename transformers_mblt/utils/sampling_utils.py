"""Top-k / top-p sampling over the top-k slice instead of the whole vocabulary.

Hugging Face builds ``TopKLogitsWarper`` followed by ``TopPLogitsWarper`` when a generation config sets both
(Qwen3 ships ``top_k=20, top_p=0.95``), then draws with ``torch.multinomial`` over the softmax of the whole
vocabulary. Both full-vocabulary steps cost milliseconds per token on the host CPU for Qwen3's 151,936 entries
(``TopPLogitsWarper`` sorts the vocabulary; ``multinomial`` draws one exponential per entry), while a 0.6B decode
step on the NPU takes about 28 ms. Only the top-k entries can be sampled, so both steps are done on that slice:

- ``MobilintTopKTopPLogitsWarper`` replaces the warper pair. It takes the top-k slice, applies the
  ``TopPLogitsWarper`` rule to it (same formula, see below for ties) and scatters the result back.
- ``MobilintTopKTopPSampler`` also draws the token on the slice and returns scores that are finite only at the
  drawn token. ``MobilintGenerationMixin._sample`` uses it, with ``do_sample=False`` on a copy of the generation
  config, when the fused warper is the last processor and the processed scores are not returned
  (``output_scores``). The random stream differs from upstream, so a seeded run draws different tokens.

On the slice the softmax, the cumulative sum and the ``min_tokens_to_keep`` guard see the same finite values the
full-vocabulary call would see. ``TopKLogitsWarper`` keeps every logit equal to the k-th value, and NPU logits tie
there on most decode steps, so the slice is widened to hold those ties.

Ties at the top-p cutoff. When equal logits straddle the cutoff, top-p keeps only some of them. Upstream, which
ones depends on ``torch.sort``'s unstable order over the whole vocabulary (on CPU it does not keep index order
among equal values), so it cannot be reproduced without that sort. Here the rule is explicit: among equal logits,
higher token ids survive. The number of survivors and their scores are the upstream ones; only which of the
equal-logit token ids receive the probability can differ. NPU logits are coarse, so this happens often (about a
third of decode steps in a quantized-logit test at ``top_p=0.8``). Separately, the softmax sums over the slice
rather than the vocabulary, so a cumulative probability within float rounding of ``1 - top_p`` can land on the
other side; that changes which entries survive at the cutoff, possibly several when equal small probabilities sit
there. Tie sets wider than four times k (for example when an
earlier processor left fewer than k finite entries) are processed over the whole vocabulary with the same token-id
rule.

Set ``MBLT_FUSED_TOPK_TOPP=0`` to keep the upstream processors and sampling (for A/B comparison).
"""

import copy
import os
from typing import Optional

import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper

_FUSED_TOPK_TOPP_ENV_VAR = "MBLT_FUSED_TOPK_TOPP"
# A tie set wider than this many times k is processed over the whole vocabulary instead of the slice.
_MAX_SLICE_FACTOR = 4


def fused_top_k_top_p_enabled() -> bool:
    """Return whether ``MBLT_FUSED_TOPK_TOPP`` leaves the fusion on (the default)."""
    return os.environ.get(_FUSED_TOPK_TOPP_ENV_VAR, "1").strip().lower() not in ("0", "false", "off", "no")


class MobilintTopKTopPLogitsWarper(LogitsProcessor):
    """Apply a ``TopKLogitsWarper`` and the ``TopPLogitsWarper`` after it on the top-k slice only.

    Args:
        top_k_warper: The upstream top-k warper; its ``top_k`` already includes ``min_tokens_to_keep``.
        top_p_warper: The upstream top-p warper that follows it.
    """

    def __init__(self, top_k_warper: TopKLogitsWarper, top_p_warper: TopPLogitsWarper):
        self.top_k_warper = top_k_warper
        self.top_p_warper = top_p_warper

    def _processed_slice(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> Optional[tuple[torch.FloatTensor, torch.LongTensor]]:
        """Return the processed scores of the top-k slice and their vocabulary indices, or ``None`` to fall back."""
        top_k = min(self.top_k_warper.top_k, scores.size(-1))
        threshold = torch.topk(scores, top_k, dim=-1).values[..., -1:]
        # Upstream top-k keeps every entry equal to the k-th value, so the slice is widened to hold the ties.
        slice_size = int((scores >= threshold).sum(dim=-1).max())
        if slice_size > _MAX_SLICE_FACTOR * top_k:
            # e.g. an earlier processor left fewer than k finite entries, so the k-th value is -inf
            return None
        values, indices = torch.topk(scores, slice_size, dim=-1)
        # In a batch, rows with fewer ties got entries below their own threshold; top-k removes those upstream.
        values = values.masked_fill(values < threshold, self.top_k_warper.filter_value)
        return self._top_p(values, indices), indices

    def _top_p(self, values: torch.FloatTensor, indices: Optional[torch.LongTensor]) -> torch.FloatTensor:
        """``TopPLogitsWarper.__call__`` on the slice, breaking ties among equal logits by token id.

        Index-ordering the slice and sorting it stably puts lower token ids first among equal values, so those are
        the ones removed when the cutoff splits them (see the module docstring). ``indices=None`` means ``values``
        is the whole vocabulary, already in token-id order.
        """
        if indices is None:
            order, by_index = None, values
        else:
            order = indices.argsort(dim=-1)
            by_index = values.gather(-1, order)
        sorted_logits, sorted_positions = torch.sort(by_index, descending=False, stable=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_to_remove = cumulative_probs <= (1 - self.top_p_warper.top_p)
        sorted_to_remove[..., -self.top_p_warper.min_tokens_to_keep :] = False
        to_remove = sorted_to_remove.scatter(-1, sorted_positions, sorted_to_remove)
        kept_by_index = by_index.masked_fill(to_remove, self.top_p_warper.filter_value)
        if order is None:
            return kept_by_index
        return torch.empty_like(values).scatter_(-1, order, kept_by_index)

    def _full_vocabulary(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Both rules over the whole vocabulary, for tie sets too wide to slice; same token-id tie break."""
        return self._top_p(self.top_k_warper(input_ids, scores), None)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        processed_slice = self._processed_slice(input_ids, scores)
        if processed_slice is None:
            return self._full_vocabulary(input_ids, scores)
        kept, indices = processed_slice
        # Upstream, the entries removed by top-k also fail the top-p test, so they end at top-p's filter value.
        processed = torch.full_like(scores, self.top_p_warper.filter_value)
        return processed.scatter_(-1, indices, kept)


class MobilintTopKTopPSampler(MobilintTopKTopPLogitsWarper):
    """Draw the next token on the top-k slice and return scores that are finite only at that token.

    Meant to run last, with greedy selection after it (see ``sample_in_top_k_slice``).
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        processed_slice = self._processed_slice(input_ids, scores)
        if processed_slice is None:
            kept, indices = self._full_vocabulary(input_ids, scores), None
        else:
            kept, indices = processed_slice
        choice = torch.multinomial(kept.softmax(dim=-1), num_samples=1)
        tokens = choice if indices is None else indices.gather(-1, choice)
        return torch.full_like(scores, -float("inf")).scatter_(-1, tokens, 0.0)


def fuse_top_k_top_p(processors: LogitsProcessorList) -> LogitsProcessorList:
    """Replace each adjacent ``TopKLogitsWarper`` → ``TopPLogitsWarper`` pair with the fused warper.

    Only the exact upstream classes are fused, and only in the shape ``generate`` builds: top-k filters to ``-inf``
    and top-p keeps at most k entries by its ``min_tokens_to_keep``. A finite top-k filter value leaves probability
    on the entries outside the top-k slice, and a larger top-p ``min_tokens_to_keep`` makes top-p keep entries that
    top-k removed; the slice cannot represent either. Subclasses, other orderings and such pairs are left alone.

    Args:
        processors: The list built by ``GenerationMixin._get_logits_processor``; modified in place.

    Returns:
        The same list.
    """
    i = 0
    while i + 1 < len(processors):
        first, second = processors[i], processors[i + 1]
        if (
            type(first) is TopKLogitsWarper
            and type(second) is TopPLogitsWarper
            and first.filter_value == -float("inf")
            and second.min_tokens_to_keep <= first.top_k
        ):
            processors[i : i + 2] = [MobilintTopKTopPLogitsWarper(first, second)]
        i += 1
    return processors


def sample_in_top_k_slice(
    processors: LogitsProcessorList, generation_config: GenerationConfig
) -> Optional[tuple[LogitsProcessorList, GenerationConfig]]:
    """Move the draw into the fused warper when nothing after it needs the processed scores.

    Applies when sampling, when ``MobilintTopKTopPLogitsWarper`` is the last processor, when its top-p filters to
    ``-inf`` (a finite filter value leaves probability on the filtered entries, which the slice draw would drop),
    and when the processed scores are not returned (``return_dict_in_generate`` with ``output_scores``).

    Args:
        processors: The processor list ``_sample`` received.
        generation_config: The generation config ``_sample`` received; not modified.

    Returns:
        A new processor list ending in ``MobilintTopKTopPSampler`` and a copy of the config with
        ``do_sample=False``, or ``None`` to keep the upstream draw.
    """
    if not generation_config.do_sample or not processors or type(processors[-1]) is not MobilintTopKTopPLogitsWarper:
        return None
    fused = processors[-1]
    if fused.top_p_warper.filter_value != -float("inf"):
        return None
    if generation_config.return_dict_in_generate and generation_config.output_scores:
        return None
    swapped = LogitsProcessorList([*processors[:-1], MobilintTopKTopPSampler(fused.top_k_warper, fused.top_p_warper)])
    greedy = copy.copy(generation_config)
    greedy.do_sample = False
    return swapped, greedy
