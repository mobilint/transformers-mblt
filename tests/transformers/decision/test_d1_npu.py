"""End-to-end D1 decisions on the NPU.

The model folder comes from ``MBLT_D1_MODEL`` (a local directory or Hub id, such as ``mobilint/d1-3B``); the suite
is skipped when the variable is unset. The model runs in its config's core mode.
"""

from __future__ import annotations

import os

import pytest

import transformers_mblt

MODEL_PATHS = tuple(filter(None, [os.environ.get("MBLT_D1_MODEL")]))

pytestmark = [pytest.mark.requires_npu, pytest.mark.requires_network]

QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Charges, refunds, invoices",
            "technical": "App or site faults",
            "fraud": "Suspected unauthorised use",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["Can wait", "Today", "Blocking the customer now"],
    },
}


@pytest.fixture(scope="module", params=MODEL_PATHS or ("<unset>",))
def model(request):
    if not MODEL_PATHS:
        pytest.skip("set MBLT_D1_MODEL to a prepared D1 model folder or Hub id")
    from transformers import AutoModel

    transformers_mblt.register()
    # The shipped config's core mode: D1 builds are compiled for one mode, so the core-mode sweep does not apply.
    loaded = AutoModel.from_pretrained(request.param)
    yield loaded
    loaded.dispose()


def test_system_one_answers_named_questions(model) -> None:
    out = model.system_one("I was charged twice this month, please refund one of them.", QUESTIONS)
    answers = out["answers"]
    assert answers["refund"]["noul"] > 0.5
    assert answers["team"]["choice"] == "billing"
    assert sum(answers["team"]["probabilities"].values()) == pytest.approx(1.0)
    assert 0.0 <= answers["urgency"]["score"] <= 2.0
    assert out["usage"]["output_tokens"] == 0 and out["usage"]["input_tokens"] > 0


def test_shared_state_matches_one_question_at_a_time(model) -> None:
    state = {"ticket": 4512, "body": "The app crashes when I open settings.", "tags": ["android"]}
    together = model.probabilities(state, list(QUESTIONS.values()))
    model.reset_context()
    alone = [model.probabilities(state, [q])[0] for q in QUESTIONS.values()]
    for a, b in zip(together, alone):
        assert a == pytest.approx(b, abs=1e-3)


def test_batch_and_images(model) -> None:
    tickets = ["Where is my parcel? It was due Monday.", "The app crashes when I open settings."]
    out = model.system_one_batch([(t, {"team": QUESTIONS["team"]}) for t in tickets])
    assert [r["answers"]["team"]["choice"] for r in out][1] == "technical"
    with pytest.raises(NotImplementedError, match="text-only"):
        model.system_one(None, {"team": QUESTIONS["team"]}, images=[object()])
