"""Shared fixtures for EAGLE-3 text-generation tests."""

from __future__ import annotations

from typing import Optional

import pytest
from transformers import AutoTokenizer, TextStreamer, pipeline


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize the shared pipeline fixture from module-level model paths."""
    if "pipe" not in metafunc.fixturenames:
        return

    model_paths = getattr(metafunc.module, "MODEL_PATHS", None)
    if not model_paths:
        return

    metafunc.parametrize("pipe", model_paths, indirect=True, ids=list(model_paths), scope="module")


@pytest.fixture(scope="module")
def pipe(request: pytest.FixtureRequest, revision: Optional[str], eagle3_npu_params):
    """Create an EAGLE-3 text-generation pipeline for the parametrized model."""
    model_path = request.param
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        revision=revision,
        trust_remote_code=True,
    )
    model_kwargs = eagle3_npu_params.model
    if model_kwargs:
        created_pipe = pipeline(
            "text-generation",
            model=model_path,
            streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
            trust_remote_code=True,
            revision=revision,
            model_kwargs=model_kwargs,
        )
    else:
        created_pipe = pipeline(
            "text-generation",
            model=model_path,
            streamer=TextStreamer(tokenizer=tokenizer, skip_prompt=False),
            trust_remote_code=True,
            revision=revision,
        )
    yield created_pipe
    del created_pipe


@pytest.fixture
def decode_rounds(monkeypatch: pytest.MonkeyPatch):
    """Replace ``decode_step`` with rounds that return the given tokens, for tests of the generate loop.

    Call the fixture with one entry per round: ``(tokens, stopped)``, or a callable taking the cache and returning
    that pair; the last entry repeats. ``tokens`` lists the ids the round settles. Rounds report zero logits over a
    16-token vocabulary and ``candidate_draft_tokens`` (default 1). ``initialize_tree`` is left to the test.
    """
    import torch

    from transformers_mblt.utils.eagle3 import decoding as decoding_module
    from transformers_mblt.utils.eagle3.tree_decoding import Eagle3DecodeStep

    def install(*rounds, candidate_draft_tokens: int = 1) -> list[dict]:
        # The keyword arguments of each decode_step call.
        calls: list[dict] = []
        pending = list(rounds)

        def _decode_step(_model, cache, _logits_processor, **kwargs):
            calls.append(kwargs)
            round_ = pending.pop(0) if len(pending) > 1 else pending[0]
            tokens, stopped = round_(cache) if callable(round_) else round_
            return Eagle3DecodeStep(
                tokens=torch.tensor([list(tokens)], dtype=torch.long),
                next_token=None if stopped else torch.tensor([[3]], dtype=torch.long),
                logits=torch.zeros((1, 1, 16), dtype=torch.float32),
                accepted_draft_tokens=max(0, len(tokens) - 1),
                candidate_draft_tokens=candidate_draft_tokens,
            )

        monkeypatch.setattr(decoding_module, "decode_step", _decode_step)
        return calls

    return install
