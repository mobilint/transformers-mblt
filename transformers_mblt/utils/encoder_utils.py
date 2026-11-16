"""Shared host side of the encoder-only MXQs (the embedder and the cross-encoder reranker).

Those MXQs hold the encoder body only: they take the embedding-layer output of one sequence, ``(1, length, hidden)``,
with a dynamic length axis and no attention mask. The host looks up the embeddings and runs each sequence of a batch
on its own, without its padding, so padding never reaches the attention and position ids start fresh per sequence.
"""

from __future__ import annotations

from typing import Optional

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
