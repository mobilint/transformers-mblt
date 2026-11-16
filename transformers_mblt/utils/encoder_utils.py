"""Shared host side of the encoder-only MXQs (the embedder and the cross-encoder reranker).

Those MXQs hold the encoder body only: they take the embedding-layer output of one sequence, ``(1, length, hidden)``,
with a dynamic length axis and no attention mask. The host looks up the embeddings and runs each sequence of a batch
on its own, without its padding, so padding never reaches the attention and position ids start fresh per sequence.
With several launched slots (``max_batch_size`` or a list ``dev_no``), the rows are spread over them round robin.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Sequence

import numpy as np
import torch


def unpadded_positions(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> list[torch.Tensor]:
    """The positions each row of ``(batch, length)`` reads: where ``attention_mask`` is 1, or all of them."""
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be (batch, length), got shape {tuple(input_ids.shape)}")
    if attention_mask is None:
        return [torch.arange(input_ids.shape[1], device=input_ids.device) for _ in range(input_ids.shape[0])]
    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"attention_mask {tuple(attention_mask.shape)} does not match input_ids {tuple(input_ids.shape)}"
        )
    rows = [torch.nonzero(row, as_tuple=True)[0] for row in attention_mask]
    for i, positions in enumerate(rows):
        if not len(positions):
            raise ValueError(f"row {i} of attention_mask has no token")
    return rows


def infer_sequence(mxq_model: object, hidden: torch.Tensor) -> np.ndarray:
    """One MXQ call on the embedding-layer output ``(1, n, hidden)`` of one unpadded sequence."""
    x = np.ascontiguousarray(hidden.detach().float().cpu().numpy())
    result = mxq_model.infer([x])  # type: ignore[attr-defined]
    if result is None:
        raise RuntimeError("the MXQ returned no output")
    out = np.asarray(result[0], dtype=np.float32)
    if not np.all(np.isfinite(out)):
        raise RuntimeError("the MXQ returned non-finite values")
    return out


def mxq_max_length(mxq_model: object, limit: int) -> int:
    """``limit``, capped by the longest input the MXQ reports (``max_width``), when it reports one."""
    try:
        width = int(getattr(mxq_model.get_input_buffer_info()[0], "max_width", 0))  # type: ignore[attr-defined]
    except (AttributeError, IndexError, TypeError):
        width = 0
    return min(limit, width) if width > 0 else limit


def encode_rows(
    mxq_models: Sequence[object],
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    token_type_ids: Optional[torch.Tensor],
    embed: Callable[..., torch.Tensor],
    max_length: int,
) -> list[tuple[torch.Tensor, np.ndarray]]:
    """Each row's read positions and the MXQ output for its unpadded tokens, in row order.

    ``embed(input_ids=..., token_type_ids=...)`` is the host embedding layer. Every row is checked against
    ``max_length`` before any MXQ call. Rows run round robin over ``mxq_models``, one thread per slot.
    """
    rows = unpadded_positions(input_ids, attention_mask)
    for positions in rows:
        if len(positions) > max_length:
            raise ValueError(f"a sequence has {len(positions)} tokens; this model takes at most {max_length}")

    def run(row: int, slot: int) -> np.ndarray:
        positions = rows[row]
        types = None if token_type_ids is None else token_type_ids[row, positions][None]
        with torch.inference_mode():
            hidden = embed(input_ids=input_ids[row, positions][None], token_type_ids=types)
        return infer_sequence(mxq_models[slot], hidden)

    slots = max(1, len(mxq_models))
    if slots == 1 or len(rows) == 1:
        outs = [run(row, 0) for row in range(len(rows))]
    else:
        def run_slot(slot: int) -> list[tuple[int, np.ndarray]]:
            return [(row, run(row, slot)) for row in range(slot, len(rows), slots)]

        with ThreadPoolExecutor(max_workers=min(slots, len(rows))) as pool:
            done = [item for chunk in pool.map(run_slot, range(min(slots, len(rows)))) for item in chunk]
        outs = [out for _, out in sorted(done, key=lambda item: item[0])]
    return list(zip(rows, outs))
