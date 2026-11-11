"""Mobilint D1: LiquidAI's d1 decision model with its LFM2 text graph on the NPU.

A decision reads a state and a set of questions and returns calibrated, typed answers in one forward pass with no
generated tokens. The NPU graph maps ``inputs_embeds`` to the final hidden state; the host looks the tokens up in the
embedding table and scores each option with the tied LM head rows of its tokens, so no full-vocabulary head runs.

Several questions on one state are a tree: the state prefix (the trunk) is prefilled once, and each question (a
branch) continues from it. The LFM2 short-convolution layers keep their state in qbruntime ``FixedTailCache``
buffers, which are snapshotted after the trunk and restored before every branch. A branch only writes KV at positions
past the trunk, so the trunk's attention KV is reused as it is. The last trunk stays on the device, so a later request
on the same state skips its prefill.

LFM2 hybrid graphs hang the NPU in the global core modes for infer calls whose length is ``0`` or ``63`` modulo 64.
Calls are planned so no call has such a length: a prefill of real tokens is split into safe calls, and only a
branch's final call may get trailing zeros instead. Zeros land in KV slots
past the branch's end, which the causal graph never reads and the next branch overwrites.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np
import qbruntime
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModel
from transformers.utils.generic import logging

from ...utils.modeling_utils import MobilintModelMixin
from .configuration_d1 import MobilintD1Config
from .prompt_d1 import D1Prompter, Question, answer, as_question

logger = logging.get_logger(__name__)

# Infer-call lengths that hang LFM2 hybrid graphs in the global core modes (length % 64).
_UNSAFE_RESIDUES = (0, 63)
_SAFE_CORE_MODES = ("single",)


def is_unsafe_length(length: int, core_mode: str) -> bool:
    """Whether an infer call of ``length`` tokens may hang an LFM2 graph in ``core_mode``."""
    return core_mode not in _SAFE_CORE_MODES and length % 64 in _UNSAFE_RESIDUES


@dataclass(frozen=True)
class Chunk:
    """One infer call: ``real`` tokens from ``offset``, followed by ``pad`` zero rows."""

    offset: int
    real: int
    pad: int = 0


def plan_chunks(
    length: int, max_chunk: int, core_mode: str, pad_last: bool = False, pad_budget: int = 2
) -> list[Chunk]:
    """Split ``length`` tokens into infer calls of at most ``max_chunk`` tokens, none of an unsafe length.

    Real tokens are never separated by padding. Only a final branch (``pad_last``) may be padded, within
    ``pad_budget`` KV slots; otherwise an unsafe remainder is split into two safe calls of real tokens. A trunk is
    never padded, since the next branch starts right after it.
    """
    chunks: list[Chunk] = []
    offset = 0
    while offset < length:
        remaining = length - offset
        if pad_last:
            pad = 0
            while is_unsafe_length(remaining + pad, core_mode):
                pad += 1
            if pad <= pad_budget and remaining + pad <= max_chunk:
                chunks.append(Chunk(offset, remaining, pad))
                break
        take = min(remaining, max_chunk)
        while take and is_unsafe_length(take, core_mode):
            take -= 1
        if not take:
            raise ValueError(f"an infer call of at most {max_chunk} tokens cannot have a safe nonempty length")
        chunks.append(Chunk(offset, take))
        offset += take
    return chunks


def _softmax(scores: Sequence[float]) -> list[float]:
    if not scores or not all(math.isfinite(s) for s in scores):
        raise RuntimeError("option scores must be nonempty and finite")
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


@dataclass
class _DeviceContext:
    """The trunk whose KV the device holds: its token ids and the short-convolution state after it."""

    ids: tuple[int, ...]
    snapshot: Optional[dict[str, Any]]


class MobilintD1PreTrainedModel(PreTrainedModel):
    config: MobilintD1Config
    config_class = MobilintD1Config
    base_model_prefix = "model"
    main_input_name = "input_ids"
    input_modalities = ("text",)
    supports_gradient_checkpointing = False
    _no_split_modules = []
    _supports_flash_attn = False
    _supports_sdpa = False
    _can_compile_fullgraph = False
    _supports_attention_backend = False
    _can_record_outputs = {}

    def _init_weights(self, module: nn.Module) -> None:
        """The embedding is read from the checkpoint; nothing is initialized randomly."""


class MobilintD1Model(MobilintModelMixin, MobilintD1PreTrainedModel):
    """D1 decisions on the NPU, with the System One API of ``LiquidAI/d1-3B``.

    - :meth:`system_one` and :meth:`system_one_batch` return typed answers per named question.
    - :meth:`probabilities` and :meth:`probabilities_batch` return each question's distribution over its options
      (``yes``, ``no`` for a noul), in the question order.
    - :meth:`run` is the primitive under them: each ``(state, questions, images)`` request's probabilities and the
      number of tokens it read.

    Requests run one after another on one MXQ, so ``max_batch_size`` stays 1. The checkpoint's tokenizer is loaded
    on first use; assign :attr:`tokenizer` to supply one.
    """

    config: MobilintD1Config
    config_class = MobilintD1Config

    def __init__(self, config: MobilintD1Config, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self._tokenizer: Any = None
        self._prompter: Optional[D1Prompter] = None
        self._context: Optional[_DeviceContext] = None
        self._tail_cache_names: Optional[list[str]] = None

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    # -- tokenizer and prompt ---------------------------------------------------------------------------

    @property
    def tokenizer(self) -> Any:
        """The checkpoint's tokenizer, loaded from the model folder or repository on first use."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            revision = getattr(self.config, "revision", None) or getattr(self.config, "_commit_hash", None)
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.name_or_path, revision=revision)
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: Any) -> None:
        self._tokenizer = value
        self._prompter = None

    @property
    def prompter(self) -> D1Prompter:
        """Renders prompts and readout ids with :attr:`tokenizer` and the config's prompt settings."""
        if self._prompter is None:
            bos = getattr(self.tokenizer, "bos_token", None)
            self._prompter = D1Prompter(
                self.tokenizer,
                bos=bos if isinstance(bos, str) else "",
                state_style=self.config.state_style,
                system=self.config.system,
                option_style=self.config.option_style,
                lead=self.config.lead,
            )
        return self._prompter

    # -- System One API ---------------------------------------------------------------------------------

    def probabilities(
        self, state: Any, questions: Sequence[Any], images: Optional[Sequence] = None
    ) -> list[list[float]]:
        """Each question's distribution over its options (``yes``, ``no`` for a noul), in one pass."""
        return self.run([(state, [as_question(q) for q in questions], images or ())])[0][0]

    def probabilities_batch(self, requests: Sequence[tuple]) -> list[list[list[float]]]:
        """:meth:`probabilities` for many ``(state, questions[, images])`` requests."""
        reqs = [(r[0], [as_question(q) for q in r[1]], (r[2] if len(r) > 2 else None) or ()) for r in requests]
        return [probs for probs, _ in self.run(reqs)]

    def system_one(self, state: Any, questions: Mapping[str, Any], images: Optional[Sequence] = None) -> dict:
        """Named questions over one state: ``{"answers": {name: answer}, "usage": {...}}``.

        A state is a string, any JSON value, or ``None``. Text-only: Mobilint D1 builds have no vision graph.
        """
        return self.system_one_batch([(state, questions, images)])[0]

    def system_one_batch(self, requests: Sequence[tuple]) -> list[dict]:
        """Many ``(state, questions)`` or ``(state, questions, images)`` requests, one response each."""
        named = [
            (r[0], {name: as_question(q) for name, q in r[1].items()}, (r[2] if len(r) > 2 else None) or ())
            for r in requests
        ]
        done = self.run([(state, list(qs.values()), images) for state, qs, images in named])
        return [
            {
                "answers": {name: answer(q, p) for (name, q), p in zip(qs.items(), probs)},
                "usage": {"input_tokens": read, "output_tokens": 0},
            }
            for (_, qs, _), (probs, read) in zip(named, done)
        ]

    @torch.inference_mode()
    def run(self, requests: Sequence[tuple[Any, Sequence[Question], Sequence]]) -> list[tuple[list[list[float]], int]]:
        """Each ``(state, questions, images)`` request's probabilities, in its questions' order, and the tokens it read.

        Requests run in order; consecutive requests on the same state reuse its prefill.
        """
        for _, questions, images in requests:
            if images:
                raise NotImplementedError(
                    "Mobilint D1 is text-only: this build ships no vision graph, so images cannot be part of the "
                    "state. Describe the picture in the text state instead."
                )
            if not questions:
                raise ValueError("a request needs at least one question")
        out = []
        for state, questions, _ in requests:
            qs = [as_question(q) for q in questions]
            rows, shared, read = self.prompter.request_rows(state, qs)
            hidden = self._decide(rows, shared)
            out.append(([self._readout(q, h) for q, h in zip(qs, hidden)], read))
        return out

    def _readout(self, q: Question, hidden: np.ndarray) -> list[float]:
        groups = self.prompter.readout_ids(q)
        selected = sorted({t for g in groups for t in g})
        rows = self.embed_tokens.weight.detach()[torch.tensor(selected)].float().cpu().numpy()
        values = rows @ hidden
        if not np.isfinite(values).all():
            raise RuntimeError("the option readout is not finite")
        logits = dict(zip(selected, values.tolist()))
        return _softmax([max(logits[t] for t in g) for g in groups])

    # -- NPU ----------------------------------------------------------------------------------------------

    @property
    def kv_capacity(self) -> int:
        """KV positions the MXQ holds (the longest prompt a decision can read)."""
        capacity = getattr(self.get_mxq_model().get_input_buffer_info()[0], "max_cache_size", 0)
        return int(capacity) or int(self.config.max_position_embeddings)

    def _max_chunk(self) -> int:
        width = int(getattr(self.get_mxq_model().get_input_buffer_info()[0], "max_width", 0))
        configured = self._coerce_positive_int(self._get_config_npu_prefill_chunk_size())
        if configured is None:
            return width
        return min(configured, width) if width else configured

    def _tail_names(self) -> list[str]:
        if self._tail_cache_names is None:
            names = [
                info.name
                for info in self.get_mxq_model().get_cache_infos()
                if info.cache_type == qbruntime.CacheType.FixedTailCache
            ]
            if not names:
                raise RuntimeError("the D1 MXQ exposes no FixedTailCache; it was not compiled with a KV cache graph")
            self._tail_cache_names = names
        return self._tail_cache_names

    def _embed(self, ids: Sequence[int]) -> np.ndarray:
        weight = self.embed_tokens.weight.detach()
        return np.ascontiguousarray(weight[torch.tensor(list(ids), dtype=torch.long)].float().cpu().numpy())

    def _infer(self, ids: Sequence[int], position: int, final: bool) -> np.ndarray:
        """Prefill ``ids`` from KV ``position`` in safe chunks; a final call returns its last real hidden state."""
        if final and not ids:
            raise ValueError("a branch needs at least one token")
        capacity = self.kv_capacity
        if position + len(ids) > capacity:
            raise ValueError(
                f"the prompt is {position + len(ids)} tokens; this MXQ holds {capacity}. Shorten the state."
            )
        x = self._embed(ids)
        core_mode = str(self.npu_backend.core_mode)
        chunks = plan_chunks(
            len(ids), self._max_chunk(), core_mode, pad_last=final, pad_budget=capacity - position - len(ids)
        )
        mxq_model = self.get_mxq_model()
        output = None
        for chunk in chunks:
            segment = x[chunk.offset : chunk.offset + chunk.real]
            if chunk.pad:
                segment = np.pad(segment, ((0, chunk.pad), (0, 0)))
            output = mxq_model.infer([np.ascontiguousarray(segment)[None]], cache_size=position + chunk.offset)
        if not final:
            return np.empty((0,), dtype=np.float32)
        raw = np.asarray(cast(list, output)[0])
        if not np.issubdtype(raw.dtype, np.floating):
            raise RuntimeError("the D1 MXQ must return dequantized floating-point hidden states")
        hidden = raw.astype(np.float32, copy=False).reshape(-1, self.config.hidden_size)[chunks[-1].real - 1]
        if not np.all(np.isfinite(hidden)):
            raise RuntimeError("the D1 MXQ returned non-finite hidden states")
        return hidden.copy()

    def _restore(self, context: _DeviceContext) -> None:
        mxq_model = self.get_mxq_model()
        if context.snapshot is None:
            mxq_model.reset_fixed_tail_cache_memory()
            return
        for name, buffer in context.snapshot.items():
            mxq_model.load_cache_memory_by_name(buffer, name)

    def _snapshot(self) -> dict[str, Any]:
        mxq_model = self.get_mxq_model()
        return {name: deepcopy(mxq_model.dump_cache_memory_by_name(name)) for name in self._tail_names()}

    def _set_trunk(self, ids: Sequence[int]) -> _DeviceContext:
        """Make ``ids`` the device trunk; the trunk already there is kept when it is the same."""
        ids = tuple(ids)
        if self._context is not None and self._context.ids == ids:
            return self._context
        # A prefill that fails half way has already overwritten the device KV.
        self._context = None
        self.get_mxq_model().reset_fixed_tail_cache_memory()
        snapshot = None
        if ids:
            self._infer(ids, 0, final=False)
            snapshot = self._snapshot()
        self._context = _DeviceContext(ids, snapshot)
        return self._context

    def _decide(self, rows: Sequence[Sequence[int]], shared: int) -> list[np.ndarray]:
        """Final hidden state of every row, which share their first ``shared`` tokens."""
        longest = max(map(len, rows))
        if longest > self.kv_capacity:
            raise ValueError(f"the prompt is {longest} tokens; this MXQ holds {self.kv_capacity}. Shorten the state.")
        # The trunk is the shared prefix; every branch keeps at least its last token, which is read out.
        trunk = max(0, min(shared, min(map(len, rows)) - 1))
        context = self._set_trunk(rows[0][:trunk])
        hidden = []
        for row in rows:
            self._restore(context)
            hidden.append(self._infer(row[trunk:], trunk, final=True))
        return hidden

    def reset_context(self) -> None:
        """Forget the trunk held on the device; the next request prefills its state from scratch."""
        self._context = None

    def dispose(self) -> None:
        self._context = None
        super().dispose()

    # -- plain forward ------------------------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Next-token logits ``(1, 1, vocab_size)`` at the last position of one sequence ``(1, length)``.

        The head is the tied embedding. Decisions should use :meth:`system_one`, which scores only the option tokens.
        """
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Mobilint D1 forward takes input_ids of shape (1, length)")
        ids = input_ids[0].tolist()
        self._context = None
        self.get_mxq_model().reset_fixed_tail_cache_memory()
        hidden = torch.from_numpy(self._infer(ids, 0, final=True))
        logits = self.embed_tokens.weight.detach().float() @ hidden
        return CausalLMOutputWithPast(logits=cast(torch.FloatTensor, logits.view(1, 1, -1)))


AutoModel.register(MobilintD1Config, MobilintD1Model)
