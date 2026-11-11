"""Hardware-free checks of Mobilint D1: the prompt format, the call planning and the trunk/branch runtime.

The runtime tests drive :class:`MobilintD1Model` with a fake MXQ that behaves like a causal hybrid graph: each
position attends to the KV of every earlier position and also reads the previous token's embedding, which it keeps
in a ``FixedTailCache``. A wrong KV position, a missed tail-cache restore or a mis-planned chunk changes the hidden
state, so the trunk/branch results must equal one plain pass over each prompt.

The upstream parity tests load the checkpoint's own ``prompt.py`` / ``runner.py`` and tokenizer from a local
``LiquidAI/d1-3B`` snapshot (``MBLT_D1_UPSTREAM`` or the Hugging Face cache) and are skipped without one.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
import torch
import torch.nn as nn

import transformers_mblt
from transformers_mblt.models.d1 import prompt_d1
from transformers_mblt.models.d1.configuration_d1 import MobilintD1Config
from transformers_mblt.models.d1.modeling_d1 import Chunk, MobilintD1Model, is_unsafe_length, plan_chunks
from transformers_mblt.models.d1.prompt_d1 import Choice, D1Prompter, Noul, Score, as_question

UPSTREAM_REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"

QUESTIONS: dict[str, dict[str, Any]] = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "refund_defined": {
        "type": "noul",
        "instructions": "Does the message ask for money back?",
        "criteria": {"true": "A refund or chargeback", "false": "Anything else"},
    },
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults", "fraud": None},
    },
    "letters": {"type": "choice", "instructions": "Pick one.", "criteria": {"x": "Ex", "y": "Why", "z": "Zed"}},
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["Can wait", "Today", "Blocking the customer now"],
    },
    "korean": {
        "type": "choice",
        "instructions": "이 요청은 어느 부서로 가야 합니까?",
        "criteria": {"결제": "요금, 환불", "기술": "앱 오류", "보안": "도용 의심"},
    },
}
STATES: list[Any] = [
    "I was charged twice this month, please refund one of them.",
    {"ticket": 4512, "body": "The app crashes when I open settings.", "tags": ["android", "v2.3"]},
    ["first line", {"nested": "value\nwith a newline"}],
    "이번 달에 결제가 두 번 됐어요. 하나 환불해 주세요.",
    None,
]


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #


class FakeTokenizer:
    """Greedy longest-match tokenizer: known words and special tokens are one token, anything else one per char."""

    WORDS = ("<|startoftext|>", "<|im_start|>", "<|im_end|>", "yes", "Yes", "YES", "no", "No", "NO")

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.bos_token = "<|startoftext|>"
        for word in self.WORDS:
            self._id(word)

    def _id(self, piece: str) -> int:
        return self.vocab.setdefault(piece, len(self.vocab))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids, i = [], 0
        while i < len(text):
            word = next((w for w in self.WORDS if text.startswith(w, i)), None)
            piece = word or text[i]
            ids.append(self._id(piece))
            i += len(piece)
        return ids


class FakeCacheInfo:
    def __init__(self, name: str, cache_type: Any) -> None:
        self.name = name
        self.cache_type = cache_type


class FakeHybridMxq:
    """A causal graph: ``h_t = mean(e_0..e_t) + 0.5 * e_{t-1}``, with ``e_{t-1}`` carried in a FixedTailCache."""

    def __init__(self, hidden: int, capacity: int = 4096, max_width: int = 64, core_mode: str = "global8") -> None:
        self.kv = np.zeros((capacity, hidden), dtype=np.float64)
        self.tail = np.zeros((hidden,), dtype=np.float64)
        self.capacity, self.max_width, self.core_mode = capacity, max_width, core_mode
        self.calls: list[tuple[int, int]] = []

    def get_input_buffer_info(self):
        return [SimpleNamespace(max_width=self.max_width, max_cache_size=self.capacity)]

    def get_cache_infos(self):
        import qbruntime

        return [
            FakeCacheInfo("attn", qbruntime.CacheType.Default),
            FakeCacheInfo("conv", qbruntime.CacheType.FixedTailCache),
        ]

    def reset_fixed_tail_cache_memory(self) -> None:
        self.tail = np.zeros_like(self.tail)

    def dump_cache_memory_by_name(self, name: str):
        assert name == "conv"
        return [self.tail.tobytes()]

    def load_cache_memory_by_name(self, buffers, name: str) -> None:
        assert name == "conv"
        self.tail = np.frombuffer(buffers[0], dtype=np.float64).copy()

    def infer(self, inputs, cache_size: int = 0):
        x = np.asarray(inputs[0])[0].astype(np.float64)
        length = x.shape[0]
        assert 0 < length <= self.max_width
        assert not is_unsafe_length(length, self.core_mode), f"unsafe call length {length}"
        assert cache_size + length <= self.capacity
        self.calls.append((cache_size, length))
        out = np.zeros_like(x)
        for i, e in enumerate(x):
            position = cache_size + i
            self.kv[position] = e
            out[i] = self.kv[: position + 1].mean(0) + 0.5 * self.tail
            self.tail = e
        return [out.astype(np.float32)[None]]


def reference_hidden(table: np.ndarray, row: list[int]) -> np.ndarray:
    e = table[row].astype(np.float64)
    previous = e[-2] if len(row) > 1 else np.zeros_like(e[-1])
    return e.mean(0) + 0.5 * previous


def make_model(
    tokenizer: Any, vocab: int, hidden: int = 8, core_mode: str = "global8", max_width: int = 64, capacity: int = 4096
) -> tuple[MobilintD1Model, FakeHybridMxq, np.ndarray]:
    model = object.__new__(MobilintD1Model)
    nn.Module.__init__(model)
    model.config = MobilintD1Config(vocab_size=vocab, hidden_size=hidden)
    generator = torch.Generator().manual_seed(0)
    model.embed_tokens = nn.Embedding(vocab, hidden)
    with torch.no_grad():
        model.embed_tokens.weight.copy_(torch.randn(vocab, hidden, generator=generator))
    mxq = FakeHybridMxq(hidden, capacity=capacity, max_width=max_width, core_mode=core_mode)
    model.npu_backend = SimpleNamespace(core_mode=core_mode, mxq_model=mxq)
    model._tokenizer, model._prompter, model._context, model._tail_cache_names = tokenizer, None, None, None
    return model, mxq, model.embed_tokens.weight.detach().numpy().copy()


def expected_probs(model: MobilintD1Model, table: np.ndarray, row: list[int], q: Any) -> list[float]:
    hidden = reference_hidden(table, row)
    scores = [max(float(table[t] @ hidden) for t in g) for g in model.prompter.readout_ids(as_question(q))]
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    return [e / sum(exps) for e in exps]


# --------------------------------------------------------------------------- #
# prompt format
# --------------------------------------------------------------------------- #


def test_prompt_text_matches_the_d1_format() -> None:
    prompter = D1Prompter(FakeTokenizer(), bos="<|startoftext|>")
    team = as_question(QUESTIONS["team"])
    assert prompter.render("Hello", team) == (
        "<|startoftext|><|im_start|>user\nHello\n\n\nQUESTION:\n"
        "Which team should handle this?\n\nOptions:\nA Charges, refunds, invoices\nB App or site faults\nC fraud\n\n"
        "Reply with the option code only.<|im_end|>\n<|im_start|>assistant\n"
    )
    assert prompter.suffix_text(as_question(QUESTIONS["refund_defined"])) == (
        "Does the message ask for money back?\nYes: A refund or chargeback\nNo: Anything else\n\n"
        "Reply with yes or no only.<|im_end|>\n<|im_start|>assistant\n"
    )
    assert prompter.suffix_text(as_question(QUESTIONS["urgency"])).startswith(
        "How urgent is this?\n\n0 Can wait\n1 Today\n2 Blocking the customer now\n\nReply with a single digit 0-2 only."
    )
    assert prompter.prefix_text(None) == "<|startoftext|><|im_start|>user\n"
    assert prompter.prefix_text({"a": 1}) == '<|startoftext|><|im_start|>user\n{\n  "a": 1\n}\n\n\nQUESTION:\n'
    assert prompter.question_block(as_question(QUESTIONS["letters"])).count("\nx Ex\ny Why\nz Zed\n") == 1


def test_option_codes_and_fallback_aliases() -> None:
    assert prompt_d1.option_codes(["a", "b"]) == ["a", "b"]
    assert prompt_d1.option_codes(["one", "two"]) == ["A", "B"]
    assert prompt_d1.option_codes([str(i) for i in range(27)])[:3] == ["00", "01", "02"]
    # "00".."26" are two tokens with a per-character tokenizer, so the aliases fall back to single tokens.
    prompter = D1Prompter(FakeTokenizer())
    aliases = prompter.aliases([f"opt{i}" for i in range(27)])
    assert len({tid for _, tid in aliases}) == 27
    assert [code for code, _ in aliases[:2]] == ["A", "B"]


def test_as_question_accepts_dicts_dataclasses_and_lookalikes() -> None:
    assert as_question({"instructions": "Pick", "criteria": {"a": None}}) == Choice("Pick", {"a": None})
    assert as_question({"type": "score", "instructions": "s", "criteria": ("lo", "hi")}) == Score("s", ["lo", "hi"])
    lookalike = SimpleNamespace(type="noul", instructions="Is it?", criteria=None)
    assert as_question(lookalike) == Noul("Is it?")
    with pytest.raises(ValueError, match="question type"):
        as_question({"type": "rank", "instructions": "x"})


def test_answer_shapes_match_the_system_one_schema() -> None:
    assert prompt_d1.answer(Noul("?"), [0.7, 0.3]) == {"type": "noul", "noul": 0.7}
    choice = prompt_d1.answer(Choice("?", {"a": None, "b": None}), [0.25, 0.75])
    assert choice == {"type": "choice", "choice": "b", "confidence": 0.75, "probabilities": {"a": 0.25, "b": 0.75}}
    score = prompt_d1.answer(Score("?", ["lo", "hi"]), [0.4, 0.6])
    assert score["score"] == pytest.approx(0.6) and score["legend"] == {"0": "lo", "1": "hi"}


# --------------------------------------------------------------------------- #
# call planning
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("length", [1, 62, 63, 64, 65, 127, 128, 2047, 2048, 2049, 4096, 5000])
@pytest.mark.parametrize("pad_last", [False, True])
def test_plan_chunks_never_makes_an_unsafe_call(length: int, pad_last: bool) -> None:
    chunks = plan_chunks(length, 2048, "global8", pad_last=pad_last, pad_budget=8)
    assert sum(c.real for c in chunks) == length
    assert all(not is_unsafe_length(c.real + c.pad, "global8") and c.real + c.pad <= 2048 for c in chunks)
    assert all(c.pad == 0 for c in chunks[:-1])
    if not pad_last:
        assert chunks[-1].pad == 0


def test_plan_chunks_splits_instead_of_padding_without_budget() -> None:
    assert plan_chunks(64, 2048, "global8", pad_last=True, pad_budget=0) == [Chunk(0, 62), Chunk(62, 2)]
    assert plan_chunks(64, 2048, "global8", pad_last=True, pad_budget=2) == [Chunk(0, 64, 1)]
    assert plan_chunks(64, 2048, "single") == [Chunk(0, 64)]


# --------------------------------------------------------------------------- #
# runtime with the fake graph
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("core_mode", ["global8", "single"])
@pytest.mark.parametrize("state", STATES, ids=range(len(STATES)))
def test_trunk_and_branches_equal_one_pass_per_prompt(state: Any, core_mode: str) -> None:
    tokenizer = FakeTokenizer()
    model, _, table = make_model(tokenizer, vocab=4096, core_mode=core_mode, max_width=48)
    questions = list(QUESTIONS.values())
    got = model.probabilities(state, questions)
    prompter = model.prompter
    rows = [
        prompter.encode(prompter.prefix_text(state)) + prompter.encode(prompter.suffix_text(as_question(q)))
        for q in questions
    ]
    for probs, row, q in zip(got, rows, questions):
        assert probs == pytest.approx(expected_probs(model, table, row, q), rel=1e-4, abs=1e-6)


def test_single_question_reads_the_whole_prompt_and_reuses_the_state() -> None:
    tokenizer = FakeTokenizer()
    model, mxq, table = make_model(tokenizer, vocab=4096, max_width=48)
    state = STATES[1]
    first = model.system_one(state, {"team": QUESTIONS["team"]})
    row = model.prompter.encode(model.prompter.render(state, as_question(QUESTIONS["team"])))
    assert first["usage"] == {"input_tokens": len(row), "output_tokens": 0}
    team = first["answers"]["team"]
    assert list(team["probabilities"].values()) == pytest.approx(expected_probs(model, table, row, QUESTIONS["team"]))

    prefix_tokens = len(model.prompter.encode(model.prompter.prefix_text(state)))
    mxq.calls.clear()
    second = model.system_one(state, {"urgency": QUESTIONS["urgency"]})
    # The state's KV and tail state are still on the device: only the question is prefilled.
    assert min(position for position, _ in mxq.calls) == prefix_tokens
    row = model.prompter.encode(model.prompter.render(state, as_question(QUESTIONS["urgency"])))
    probs = list(second["answers"]["urgency"]["probabilities"].values())
    assert probs == pytest.approx(expected_probs(model, table, row, QUESTIONS["urgency"]), rel=1e-4)


def test_a_new_state_replaces_the_held_trunk() -> None:
    tokenizer = FakeTokenizer()
    model, mxq, table = make_model(tokenizer, vocab=4096, max_width=48)
    questions = [QUESTIONS["refund"], QUESTIONS["urgency"]]
    model.probabilities("log line one", questions)
    mxq.calls.clear()
    probs = model.probabilities("log line one and two", questions)
    assert mxq.calls[0][0] == 0
    prompter = model.prompter
    for p, q in zip(probs, questions):
        row = prompter.encode(prompter.render("log line one and two", as_question(q)))
        assert p == pytest.approx(expected_probs(model, table, row, q), rel=1e-4)
    model.reset_context()
    mxq.calls.clear()
    model.probabilities("log line one and two", questions)
    assert mxq.calls[0][0] == 0


def test_batch_returns_one_response_per_request_in_order() -> None:
    model, _, _ = make_model(FakeTokenizer(), vocab=4096)
    out = model.system_one_batch([(s, {"refund": QUESTIONS["refund"]}) for s in STATES[:3]])
    assert [set(r["answers"]) for r in out] == [{"refund"}] * 3
    singles = [model.system_one(s, {"refund": QUESTIONS["refund"]}) for s in STATES[:3]]
    assert [r["answers"]["refund"]["noul"] for r in out] == pytest.approx(
        [r["answers"]["refund"]["noul"] for r in singles], rel=1e-6
    )
    batch = model.probabilities_batch([(STATES[0], [QUESTIONS["refund"]]), (STATES[1], [QUESTIONS["team"]], None)])
    assert [len(p[0]) for p in batch] == [2, 3]


def test_images_and_overlong_prompts_are_rejected() -> None:
    model, _, _ = make_model(FakeTokenizer(), vocab=4096, capacity=256)
    with pytest.raises(NotImplementedError, match="text-only"):
        model.system_one(None, {"cats": QUESTIONS["letters"]}, images=[object()])
    with pytest.raises(ValueError, match="this MXQ holds 256"):
        model.system_one("x" * 400, {"refund": QUESTIONS["refund"]})


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #


def test_registration_and_proxy_exports() -> None:
    from transformers import AutoConfig, AutoModel

    assert "d1" in transformers_mblt.register()
    assert AutoModel._model_mapping[MobilintD1Config] is MobilintD1Model
    config = AutoConfig.for_model("mobilint-d1", mxq_path="d1-3B.mxq", core_mode="global8")
    assert isinstance(config, MobilintD1Config) and config.to_dict()["core_mode"] == "global8"

    proxy = Path(transformers_mblt.__file__).parent / "models" / "d1" / "proxy_d1.py"
    spec = importlib.util.spec_from_file_location("proxy_d1_under_test", proxy)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.MobilintD1Model is MobilintD1Model and module.MobilintD1Config is MobilintD1Config


# --------------------------------------------------------------------------- #
# upstream parity (local LiquidAI/d1-3B snapshot only)
# --------------------------------------------------------------------------- #


def _upstream_snapshot() -> Optional[Path]:
    explicit = os.environ.get("MBLT_D1_UPSTREAM")
    if explicit:
        return Path(explicit)
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    found = try_to_load_from_cache("LiquidAI/d1-3B", "runner.py", revision=UPSTREAM_REVISION)
    return Path(found).parent if isinstance(found, str) else None


@pytest.fixture(scope="module")
def upstream() -> SimpleNamespace:
    snapshot = _upstream_snapshot()
    if snapshot is None or not (snapshot / "runner.py").is_file():
        pytest.skip("no local LiquidAI/d1-3B snapshot (set MBLT_D1_UPSTREAM or cache the revision)")
    package = "_d1_upstream_under_test"
    spec = importlib.util.spec_from_file_location(
        package, snapshot / "__init__.py", submodule_search_locations=[str(snapshot)]
    )
    module = ModuleType(package)
    module.__path__ = [str(snapshot)]
    module.__spec__ = spec
    sys.modules[package] = module
    try:
        prompt = importlib.import_module(f"{package}.prompt")
        runner = importlib.import_module(f"{package}.runner")
    except ImportError as exc:
        pytest.skip(f"the upstream runner does not import here: {exc}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot))
    return SimpleNamespace(prompt=prompt, runner=runner, tokenizer=tokenizer)


class _RecordingBackbone(nn.Module):
    """Stands in for the upstream LFM2-VL backbone: records the token rows it is asked to read."""

    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(_name_or_path="LiquidAI/d1-3B", model_type="lfm2_vl")
        self.vocab = vocab
        self.dummy = nn.Parameter(torch.zeros(1))
        self.rows: list[list[int]] = []

    def forward(self, input_ids: torch.Tensor, logits_to_keep: int = 1):
        self.rows.append(input_ids[0].tolist())
        return SimpleNamespace(logits=torch.zeros(1, 1, self.vocab))

    def answer(self, trunk: torch.Tensor, packed: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        trunk_ids, flat, start = trunk[0].tolist(), packed.tolist(), 0
        for n in lengths.tolist():
            self.rows.append(trunk_ids + flat[start : start + n])
            start += n
        return torch.zeros(len(lengths), self.vocab)


@pytest.mark.parametrize("state", STATES, ids=range(len(STATES)))
@pytest.mark.parametrize("names", [["refund"], ["korean"], ["refund", "refund_defined", "team", "letters", "urgency"]])
def test_token_rows_equal_the_upstream_runner(upstream: SimpleNamespace, state: Any, names: list[str]) -> None:
    tokenizer = upstream.tokenizer
    backbone = _RecordingBackbone(len(tokenizer))
    system_one = upstream.runner.SystemOne(model=backbone, tokenizer=tokenizer, device="cpu")
    up_questions = [upstream.prompt.as_question(QUESTIONS[n]) for n in names]
    [(_, read)] = system_one.run([(state, up_questions, ())])

    prompter = D1Prompter(tokenizer, bos=tokenizer.bos_token)
    rows, shared, our_read = prompter.request_rows(state, [as_question(QUESTIONS[n]) for n in names])
    assert rows == backbone.rows
    assert our_read == read
    assert all(row[:shared] == rows[0][:shared] for row in rows)
    for name, up_q in zip(names, up_questions):
        assert prompter.readout_ids(as_question(QUESTIONS[name])) == upstream.prompt.readout_ids(tokenizer, up_q)


def test_readout_and_many_options_match_upstream(upstream: SimpleNamespace) -> None:
    tokenizer = upstream.tokenizer
    prompter = D1Prompter(tokenizer, bos=tokenizer.bos_token)
    many = {"type": "choice", "instructions": "Which?", "criteria": {f"item_{i}": f"Item {i}" for i in range(40)}}
    for q in [
        *QUESTIONS.values(),
        many,
        {"type": "score", "instructions": "s", "criteria": [str(i) for i in range(10)]},
    ]:
        up_q = upstream.prompt.as_question(q)
        ours = as_question(q)
        assert prompter.render(STATES[1], ours) == upstream.prompt.render(
            tokenizer, STATES[1], up_q, tokenizer.bos_token
        )
        logz = torch.randn(len(tokenizer), generator=torch.Generator().manual_seed(len(json.dumps(q))))
        groups = prompter.readout_ids(ours)
        scores = [max(float(logz[i]) for i in g) for g in groups]
        top = max(scores)
        probs = [math.exp(s - top) / sum(math.exp(t - top) for t in scores) for s in scores]
        assert probs == pytest.approx(upstream.prompt.readout(tokenizer, up_q, logz), rel=1e-9)


def test_forward_returns_tied_head_logits_at_the_last_position() -> None:
    model, _, table = make_model(FakeTokenizer(), vocab=64, max_width=16)
    ids = list(range(1, 40))
    logits = model(input_ids=torch.tensor([ids])).logits
    assert logits.shape == (1, 1, 64)
    expected = table.astype(np.float64) @ reference_hidden(table, ids)
    assert logits[0, 0].numpy() == pytest.approx(expected, rel=1e-4, abs=1e-5)
    assert model._context is None
