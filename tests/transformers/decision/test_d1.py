"""Hardware-free checks of Mobilint D1: the prompt glue, the call planning and the trunk/branch runtime.

The runtime tests drive :class:`MobilintD1Model` with a fake MXQ that behaves like a causal hybrid graph: each
position attends to the KV of every earlier position and also reads the previous token's embedding, which it keeps
in a ``FixedTailCache``. A wrong KV position, a missed tail-cache restore or a mis-planned chunk changes the hidden
state, so the trunk/branch results must equal one plain pass over each prompt.

The runtime tests read prompts through ``fixtures/prompt.py``, a stand-in with the checkpoint file's call signatures
and a simple format of its own; the package ships no prompt code. The upstream parity tests load the checkpoint's own
``prompt.py`` / ``runner.py`` and tokenizer from a local ``LiquidAI/d1-3B`` snapshot (``MBLT_D1_UPSTREAM`` or the
Hugging Face cache) and are skipped without one.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
import torch
import torch.nn as nn

import transformers_mblt
from transformers_mblt.models.d1 import D1_TASK
from transformers_mblt.models.d1.configuration_d1 import MobilintD1Config
from transformers_mblt.models.d1.modeling_d1 import Chunk, MobilintD1Model, is_unsafe_length, plan_chunks
from transformers_mblt.models.d1.prompt_d1 import D1Prompter, answer, load_prompt_module, resolve_prompt_file

UPSTREAM_REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"
REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PROMPT = Path(__file__).parent / "fixtures" / "prompt.py"

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

    WORDS = ("<|startoftext|>", "<|im_start|>", "<|im_end|>", "yes", "Yes", "no", "No")

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

    def __init__(
        self, hidden: int, capacity: int = 4096, max_width: int = 64, core_mode: str = "global8", delay: float = 0.0
    ) -> None:
        self.kv = np.zeros((capacity, hidden), dtype=np.float64)
        self.tail = np.zeros((hidden,), dtype=np.float64)
        self.capacity, self.max_width, self.core_mode, self.delay = capacity, max_width, core_mode, delay
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
            if self.delay:
                time.sleep(self.delay)  # lets another thread interleave, as a device call would
            out[i] = self.kv[: position + 1].mean(0) + 0.5 * self.tail
            self.tail = e
        return [out.astype(np.float32)[None]]


def reference_hidden(table: np.ndarray, row: list[int]) -> np.ndarray:
    e = table[row].astype(np.float64)
    previous = e[-2] if len(row) > 1 else np.zeros_like(e[-1])
    return e.mean(0) + 0.5 * previous


def make_model(
    tokenizer: Any,
    vocab: int,
    hidden: int = 8,
    core_mode: str = "global8",
    max_width: int = 64,
    capacity: int = 4096,
    delay: float = 0.0,
) -> tuple[MobilintD1Model, FakeHybridMxq, np.ndarray]:
    model = object.__new__(MobilintD1Model)
    nn.Module.__init__(model)
    model.config = MobilintD1Config(vocab_size=vocab, hidden_size=hidden)
    generator = torch.Generator().manual_seed(0)
    model.embed_tokens = nn.Embedding(vocab, hidden)
    with torch.no_grad():
        model.embed_tokens.weight.copy_(torch.randn(vocab, hidden, generator=generator))
    mxq = FakeHybridMxq(hidden, capacity=capacity, max_width=max_width, core_mode=core_mode, delay=delay)
    launches: list[str] = []
    model.npu_backend = SimpleNamespace(
        core_mode=core_mode,
        mxq_model=mxq,
        launch=lambda: launches.append("launch"),
        dispose=lambda: launches.append("dispose"),
        launches=launches,
    )
    model._tokenizer, model._prompter, model._context, model._tail_cache_names = tokenizer, None, None, None
    model._prompt_module, model.calibration = load_prompt_module(FIXTURE_PROMPT), None
    return model, mxq, model.embed_tokens.weight.detach().numpy().copy()


def row_of(model: MobilintD1Model, state: Any, q: Any) -> list[int]:
    return model.prompter.encode(model.render(state, q))


def expected_probs(model: MobilintD1Model, table: np.ndarray, row: list[int], q: Any) -> list[float]:
    hidden = reference_hidden(table, row)
    scores = [max(float(table[t] @ hidden) for t in g) for g in model.prompter.readout_ids(model.prompter.question(q))]
    top = max(scores)
    exps = [math.exp(s - top) for s in scores]
    return [e / sum(exps) for e in exps]


# --------------------------------------------------------------------------- #
# prompt glue
# --------------------------------------------------------------------------- #


def test_prompt_py_is_loaded_from_the_model_folder(tmp_path: Path) -> None:
    (tmp_path / "prompt.py").write_bytes(FIXTURE_PROMPT.read_bytes())
    assert resolve_prompt_file(tmp_path) == tmp_path / "prompt.py"
    assert load_prompt_module(tmp_path / "prompt.py") is load_prompt_module(FIXTURE_PROMPT)
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no prompt.py"):
        resolve_prompt_file(tmp_path / "empty")
    (tmp_path / "other.py").write_text("X = 1\n", encoding="utf-8")
    with pytest.raises(ImportError, match="not a D1 prompt.py"):
        load_prompt_module(tmp_path / "other.py")


def test_the_package_ships_no_prompt_code() -> None:
    package = REPO_ROOT / "transformers_mblt" / "models" / "d1"
    text = "\n".join(p.read_text(encoding="utf-8") for p in package.glob("*.py"))
    for marker in ("_FALLBACK_POOL", "Reply with the option code only", "def _sections", "QUESTION:"):
        assert marker not in text


def test_questions_accept_dicts_module_classes_and_lookalikes() -> None:
    module = load_prompt_module(FIXTURE_PROMPT)
    prompter = D1Prompter(module, FakeTokenizer(), "json_only", "none", "desc", "")
    assert prompter.question({"instructions": "Pick", "criteria": {"a": None}}) == module.Choice("Pick", {"a": None})
    noul = module.Noul("Is it?")
    assert prompter.question(noul) is noul
    lookalike = SimpleNamespace(type="score", instructions="s", criteria=("lo", "hi"))
    assert prompter.question(lookalike) == module.Score("s", ["lo", "hi"])
    with pytest.raises(ValueError, match="question type"):
        prompter.question({"type": "rank", "instructions": "x"})
    with pytest.raises(ValueError, match="needs criteria"):
        prompter.question({"type": "choice", "instructions": "x"})
    with pytest.raises(TypeError):
        prompter.question(3)


def test_answer_shapes_match_the_system_one_schema() -> None:
    module = load_prompt_module(FIXTURE_PROMPT)
    assert answer(module.Noul("?"), [0.7, 0.3]) == {"type": "noul", "noul": 0.7}
    choice = answer(module.Choice("?", {"a": None, "b": None}), [0.25, 0.75])
    assert choice == {"type": "choice", "choice": "b", "confidence": 0.75, "probabilities": {"a": 0.25, "b": 0.75}}
    score = answer(module.Score("?", ["lo", "hi"]), [0.4, 0.6])
    assert score == {
        "type": "score",
        "score": pytest.approx(0.6),
        "confidence": 0.6,
        "probabilities": {"0": 0.4, "1": 0.6},
        "legend": {"0": "lo", "1": "hi"},
    }


def test_task_tag_is_one_constant() -> None:
    from transformers_mblt.utils.api import TASKS

    assert D1_TASK in TASKS
    assert f"| `{D1_TASK}` | D1" in (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert f"`{D1_TASK}`" in (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    script = (REPO_ROOT / "scripts" / "build_d1_hf_package.py").read_text(encoding="utf-8")
    assert "pipeline_tag: {D1_TASK}" in script


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
        prompter.encode(prompter.prefix(state)) + prompter.encode(prompter.suffix(prompter.question(q)))
        for q in questions
    ]
    assert model.tokens(state, questions) == max(map(len, rows))
    for probs, row, q in zip(got, rows, questions):
        assert probs == pytest.approx(expected_probs(model, table, row, q), rel=1e-4, abs=1e-6)


def test_single_question_reads_the_whole_prompt_and_reuses_the_state() -> None:
    tokenizer = FakeTokenizer()
    model, mxq, table = make_model(tokenizer, vocab=4096, max_width=48)
    state = STATES[1]
    first = model.system_one(state, {"team": QUESTIONS["team"]})
    row = row_of(model, state, QUESTIONS["team"])
    assert first["usage"] == {"input_tokens": len(row), "output_tokens": 0}
    team = first["answers"]["team"]
    assert list(team["probabilities"].values()) == pytest.approx(expected_probs(model, table, row, QUESTIONS["team"]))

    prefix_tokens = len(model.prompter.encode(model.prompter.prefix(state)))
    mxq.calls.clear()
    second = model.system_one(state, {"urgency": QUESTIONS["urgency"]})
    # The state's KV and tail state are still on the device: only the question is prefilled.
    assert min(position for position, _ in mxq.calls) == prefix_tokens
    row = row_of(model, state, QUESTIONS["urgency"])
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
    for p, q in zip(probs, questions):
        row = row_of(model, "log line one and two", q)
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


def test_launch_dispose_and_forward_drop_the_held_trunk() -> None:
    model, mxq, _ = make_model(FakeTokenizer(), vocab=4096)
    for step in ("launch", "dispose"):
        model.probabilities(STATES[0], [QUESTIONS["refund"], QUESTIONS["urgency"]])
        assert model._context is not None
        getattr(model, step)()
        assert model._context is None
    assert model.npu_backend.launches == ["launch", "dispose"]


def test_concurrent_requests_are_serialized() -> None:
    model, _, table = make_model(FakeTokenizer(), vocab=4096, max_width=48, delay=1e-4)
    questions = [QUESTIONS["refund"], QUESTIONS["team"]]
    states = STATES[:4]
    results: dict[int, Any] = {}
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            for _ in range(3):
                results[i] = model.probabilities(states[i], questions)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(states))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    prompter = model.prompter
    for i, state in enumerate(states):
        for probs, q in zip(results[i], questions):
            row = prompter.encode(prompter.prefix(state)) + prompter.encode(prompter.suffix(prompter.question(q)))
            assert probs == pytest.approx(expected_probs(model, table, row, q), rel=1e-4, abs=1e-6)


def test_calibration_sees_log_probabilities() -> None:
    model, _, table = make_model(FakeTokenizer(), vocab=4096)
    seen: list[list[float]] = []

    class Shift:
        def apply(self, q: Any, scores: list[float]) -> list[float]:
            seen.append(list(scores))
            return [s + (2.0 if i == 0 else 0.0) for i, s in enumerate(scores)]

    q = QUESTIONS["refund"]
    plain = model.probabilities(STATES[0], [q])[0]
    model.calibration = Shift()
    shifted = model.probabilities(STATES[0], [q])[0]
    hidden = reference_hidden(table, row_of(model, STATES[0], q))
    logits = table.astype(np.float64) @ hidden
    logz = logits - np.log(np.exp(logits - logits.max()).sum()) - logits.max()
    groups = model.prompter.readout_ids(model.prompter.question(q))
    assert seen[0] == pytest.approx([max(logz[t] for t in g) for g in groups], rel=1e-4, abs=1e-4)
    assert shifted[0] / shifted[1] == pytest.approx(math.exp(2.0) * plain[0] / plain[1], rel=1e-4)


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
    # What the model loads at runtime: the checkpoint's prompt.py, imported on its own.
    loaded = load_prompt_module(snapshot / "prompt.py")
    prompter = D1Prompter(loaded, tokenizer, "json_only", "none", "desc", "")
    return SimpleNamespace(prompt=prompt, runner=runner, tokenizer=tokenizer, loaded=loaded, prompter=prompter)


def test_prompt_text_matches_the_d1_format(upstream: SimpleNamespace) -> None:
    prompter = D1Prompter(upstream.loaded, FakeTokenizer(), "json_only", "none", "desc", "")
    team = prompter.question(QUESTIONS["team"])
    assert prompter.render("Hello", team) == (
        "<|startoftext|><|im_start|>user\nHello\n\n\nQUESTION:\n"
        "Which team should handle this?\n\nOptions:\nA Charges, refunds, invoices\nB App or site faults\nC fraud\n\n"
        "Reply with the option code only.<|im_end|>\n<|im_start|>assistant\n"
    )
    assert prompter.prefix(None) == "<|startoftext|><|im_start|>user\n"
    assert prompter.prefix({"a": 1}) == '<|startoftext|><|im_start|>user\n{\n  "a": 1\n}\n\n\nQUESTION:\n'


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

    prompter = upstream.prompter
    ours = [prompter.question(QUESTIONS[n]) for n in names]
    rows, shared, our_read = prompter.request_rows(state, ours)
    assert rows == backbone.rows
    assert our_read == read
    assert all(row[:shared] == rows[0][:shared] for row in rows)
    assert prompter.tokens(state, ours) == system_one.tokens(state, up_questions)
    for q, up_q in zip(ours, up_questions):
        assert prompter.readout_ids(q) == upstream.prompt.readout_ids(tokenizer, up_q)


def test_readout_and_many_options_match_upstream(upstream: SimpleNamespace) -> None:
    tokenizer = upstream.tokenizer
    prompter = upstream.prompter
    many = {"type": "choice", "instructions": "Which?", "criteria": {f"item_{i}": f"Item {i}" for i in range(40)}}
    for q in [
        *QUESTIONS.values(),
        many,
        {"type": "score", "instructions": "s", "criteria": [str(i) for i in range(10)]},
    ]:
        up_q = upstream.prompt.as_question(q)
        ours = prompter.question(q)
        assert prompter.render(STATES[1], ours) == upstream.prompt.render(
            tokenizer, STATES[1], up_q, tokenizer.bos_token
        )
        logz = torch.randn(len(tokenizer), generator=torch.Generator().manual_seed(len(json.dumps(q))))
        selected = {t for g in prompter.readout_ids(ours) for t in g}
        probs = prompter.probabilities(ours, {t: float(logz[t]) for t in selected})
        assert probs == pytest.approx(upstream.prompt.readout(tokenizer, up_q, logz), rel=1e-9)


def test_forward_returns_tied_head_logits_at_the_last_position() -> None:
    model, _, table = make_model(FakeTokenizer(), vocab=64, max_width=16)
    ids = list(range(1, 40))
    logits = model(input_ids=torch.tensor([ids])).logits
    assert logits.shape == (1, 1, 64)
    expected = table.astype(np.float64) @ reference_hidden(table, ids)
    assert logits[0, 0].numpy() == pytest.approx(expected, rel=1e-4, abs=1e-5)
    assert model._context is None
