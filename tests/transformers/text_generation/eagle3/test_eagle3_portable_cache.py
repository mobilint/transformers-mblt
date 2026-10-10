"""EAGLE-3 caches: base/draft binding, unfed tokens after a stop, and cache rebuilds."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from transformers_mblt.utils.cache_residency import MobilintCacheStateError
from transformers_mblt.utils.cache_utils import MobilintEagle3Cache
from transformers_mblt.utils.eagle3.tree_decoding import initialize_tree
from transformers_mblt.utils.generation_utils import MobilintEagle3GenerationMixin, llm_eagle3_forward


class _SlotMxq:
    """MXQ double whose single KV slot holds a string."""

    def __init__(self) -> None:
        self.kv = ""
        self.calls: list[str] = []

    def dump_cache_memory(self, cache_id: int = 0) -> list[bytes]:
        self.calls.append("dump")
        return [self.kv.encode()]

    def load_cache_memory(self, buffer: list[bytes], cache_id: int = 0) -> None:
        self.calls.append("load")
        self.kv = buffer[0].decode()


def _write(layer, mxq: _SlotMxq, text: str) -> None:
    mxq.kv = mxq.kv[: layer.get_seq_length()] + text
    layer.update_seen_tokens(len(text))


def test_two_eagle3_caches_alternate_on_the_base_and_draft_mxqs() -> None:
    base, draft = _SlotMxq(), _SlotMxq()
    a, b = MobilintEagle3Cache(base, draft), MobilintEagle3Cache(base, draft)

    for cache, text in ((a, "aaa"), (b, "bb")):
        cache.bind_base(base)
        _write(cache.base_layer, base, text)
        cache.bind_draft(draft)
        _write(cache.draft_layer, draft, text)

    a.bind_base(base)
    a.bind_draft(draft)
    assert (base.kv, draft.kv) == ("aaa", "aaa")
    assert a.is_resident and not b.is_resident

    b.bind_base(base)
    b.bind_draft(draft)
    assert (base.kv, draft.kv) == ("bb", "bb")


def test_eagle3_cache_moves_to_another_instance() -> None:
    src_base, src_draft, dst_base, dst_draft = _SlotMxq(), _SlotMxq(), _SlotMxq(), _SlotMxq()
    cache = MobilintEagle3Cache(src_base, src_draft)
    cache.bind_base(src_base)
    _write(cache.base_layer, src_base, "hello")
    cache.bind_draft(src_draft)
    _write(cache.draft_layer, src_draft, "hello")

    cache.bind_base(dst_base)
    cache.bind_draft(dst_draft)

    assert (dst_base.kv, dst_draft.kv) == ("hello", "hello")
    assert cache.base_mxq_model is dst_base and cache.draft_mxq_model is dst_draft


def test_lost_draft_kv_is_tolerated_but_lost_base_kv_is_refused() -> None:
    """The draft only proposes tokens the base verifies, so missing draft KV is acceptable there and only there."""
    base, draft = _SlotMxq(), _SlotMxq()
    cache = MobilintEagle3Cache(base, draft)
    cache.bind_base(base)
    _write(cache.base_layer, base, "abc")
    cache.bind_draft(draft)
    _write(cache.draft_layer, draft, "abc")
    base._acc = None  # dispose: the KV on both MXQs is gone, nothing was saved
    draft._acc = None
    new_base, new_draft = _SlotMxq(), _SlotMxq()

    cache.bind_draft(new_draft)  # draft: continues on whatever the slot holds
    assert cache.draft_mxq_model is new_draft
    with pytest.raises(MobilintCacheStateError):
        cache.bind_base(new_base)


def test_copy_keeps_unfed_tokens_and_reset_clears_them() -> None:
    cache = MobilintEagle3Cache(_SlotMxq(), _SlotMxq())
    cache.accept_tokens = torch.tensor([[4, 5]])
    cache.defer_accept_tokens()

    copied = cache.copy()
    assert torch.equal(copied.unfed_tokens, torch.tensor([[4, 5]]))
    cache.reset()
    assert cache.unfed_tokens is None and torch.equal(copied.unfed_tokens, torch.tensor([[4, 5]]))


class _RecordingBase:
    def __init__(self) -> None:
        self.fed: list[list[int]] = []

    def __call__(self, input_ids, *_args, **_kwargs):
        self.fed.append(input_ids[0].tolist())
        n = input_ids.shape[1]
        return {"hidden_states": [torch.zeros((1, n, 4))]}, torch.zeros((1, 8))


class _RecordingDraft:
    def __init__(self) -> None:
        self.fed: list[list[int]] = []

    def topk_generate(self, hidden_states, *, input_ids_delta=None, **_kwargs):
        self.fed.append(input_ids_delta[0].tolist())
        assert input_ids_delta.shape[1] == hidden_states.shape[1]
        return (
            torch.tensor([[3]]),
            torch.tensor([[0]]),
            torch.ones((1, 1, 1, 1)),
            torch.tensor([0]),
        )


def _cache_after_stop(committed: int, unfed: list[int]) -> SimpleNamespace:
    seq = {"base": committed}
    return SimpleNamespace(
        unfed_tokens=torch.tensor([unfed]),
        pending_draft_tokens=None,
        retrieve_indices=None,
        tree_mask=None,
        tree_position_ids=None,
        get_base_seq_length=lambda: seq["base"],
        update_base_seen_tokens=lambda n: seq.__setitem__("base", seq["base"] + int(n)),
        get_draft_seq_length=lambda: committed,
    )


@pytest.mark.parametrize(
    "input_ids, expected_base",
    [
        ([1, 2, 3, 7, 8, 9, 10], [7, 8, 9, 10]),  # full sequence: already repeats the unfed tokens
        ([9, 10], [7, 8, 9, 10]),  # delta only: the unfed tokens are prepended
    ],
)
def test_next_generate_feeds_the_unfed_tokens_first(input_ids, expected_base) -> None:
    base, draft = _RecordingBase(), _RecordingDraft()
    model = SimpleNamespace(eagle3_base_model=base, eagle3_draft_model=draft)
    cache = _cache_after_stop(committed=3, unfed=[7, 8])

    initialize_tree(torch.tensor([input_ids]), model, cache, logits_processor=None)

    assert base.fed == [expected_base]
    assert draft.fed == [expected_base[1:] + [0]]  # shifted by one, then the sampled token (argmax of zeros)
    assert cache.unfed_tokens is None
    assert cache.get_base_seq_length() == 7


def test_forward_after_a_stop_feeds_the_unfed_tokens_first() -> None:
    base = _RecordingBase()
    model = SimpleNamespace(eagle3_base_model=base, config=SimpleNamespace(vocab_size=8))
    seen = {"n": 0}

    class _Cache(MobilintEagle3Cache):
        pass

    cache = object.__new__(_Cache)
    cache.unfed_tokens = torch.tensor([[7, 8]])
    cache.base_layer = SimpleNamespace(update_seen_tokens=lambda n: seen.__setitem__("n", seen["n"] + n))

    llm_eagle3_forward(model, input_ids=torch.tensor([[9]]), past_key_values=cache)

    assert base.fed == [[7, 8, 9]]
    assert seen["n"] == 3 and cache.unfed_tokens is None


def test_get_cache_rebuilds_after_a_relaunch() -> None:
    class _Model:
        _get_cache = MobilintEagle3GenerationMixin._get_cache

        def __init__(self) -> None:
            self.models = (_SlotMxq(), _SlotMxq())

        def get_cache_mxq_models(self):
            return self.models

    model = _Model()
    first = model._get_cache("", 1, 0)
    assert model._get_cache("", 1, 0) is first  # reused (and reset) while the MXQs are the same

    model.models = (_SlotMxq(), _SlotMxq())  # dispose + launch
    rebuilt = model._get_cache("", 1, 0)
    assert rebuilt is not first and rebuilt.base_mxq_model is model.models[0]


def test_base_forward_binds_the_cache_before_inferring() -> None:
    from transformers_mblt.utils.eagle3.eagle3_utils import MobilintEagle3BaseModelMixin

    calls: list[str] = []

    class _Cache(MobilintEagle3Cache):
        def bind_base(self, model):
            calls.append("bind")

    class _Mxq:
        def infer(self, inputs, _outputs, cache_size):
            calls.append("infer")
            n = inputs[0].shape[2]
            h = np.zeros((1, 1, n, 2), dtype=np.float32)
            return [h, h, h, np.zeros((1, 1, n, 8), dtype=np.float32)]

    base = MobilintEagle3BaseModelMixin.__new__(MobilintEagle3BaseModelMixin)
    base.get_mxq_model = lambda: _Mxq()
    base.embed_tokens = torch.nn.Embedding(16, 6)
    base.rotary_emb = lambda embeds, positions: np.zeros((1, embeds.shape[1], 4), dtype=np.float32)
    base.resolve_npu_prefill_chunk_size = lambda _override: 64
    base._prepare_decoder_attention_mask = lambda mask, shape, embeds, past, cache: torch.zeros(
        (1, 1, shape[1], past + shape[1])
    )
    cache = MobilintEagle3Cache(_SlotMxq(), _SlotMxq())
    cache.__class__ = _Cache

    with torch.no_grad():
        base.forward(torch.tensor([[1, 2]]), cache=cache, requires_all_features_logits=False)

    assert calls[:2] == ["bind", "infer"]


def test_tree_positions_follow_the_cache_not_the_input_length() -> None:
    """A delta-only input is shorter than the conversation; the tree must still sit right after the cached prefix."""
    from transformers_mblt.utils.eagle3.tree_decoding import tree_decoding

    seen: dict[str, torch.Tensor] = {}
    seq = {"base": 10}

    def base_model(tree_candidates, *, position_ids, **_kwargs):
        seen["positions"] = position_ids
        n = tree_candidates.shape[1]
        return {"hidden_states": [torch.zeros((1, n, 4))]}, torch.zeros((1, n, 8))

    cache = SimpleNamespace(
        accept_tokens=torch.tensor([[5, 6]]),
        get_base_seq_length=lambda: seq["base"],
        update_base_seen_tokens=lambda n: seq.__setitem__("base", seq["base"] + int(n)),
    )
    model = SimpleNamespace(eagle3_base_model=base_model)

    tree_decoding(
        model,
        cache,
        torch.tensor([[7, 8, 9]]),
        torch.tensor([[1, 2, 3]]),  # delta-only: far shorter than the 10 cached tokens
        torch.tensor([[0, 1]]),
        torch.tensor([0, 1, 1]),
    )

    assert seen["positions"].tolist() == [[10, 11, 12, 13, 13]]
    assert seq["base"] == 12  # the pending accepted tokens are now fed


def test_decode_step_feeds_the_draft_the_accepted_tokens_shifted_by_one() -> None:
    from transformers_mblt.utils.eagle3.tree_decoding import update_inference_inputs

    draft = _RecordingDraft()
    model = SimpleNamespace(eagle3_draft_model=draft)
    cache = SimpleNamespace(
        accept_tokens=None, pending_draft_tokens=None, retrieve_indices=None, tree_mask=None, tree_position_ids=None
    )
    candidates = torch.tensor([[4, 5, 6]])

    update_inference_inputs(
        torch.tensor([[1, 2]]),  # delta-only history
        candidates,
        torch.tensor(0),
        torch.tensor(2),
        torch.tensor([[0, 1, 2]]),
        None,
        0,
        model,
        cache,
        torch.zeros((1, 3, 4)),
        torch.zeros(8),
        None,
    )

    assert draft.fed == [[5, 6, 0]]  # accepted [4, 5, 6] shifted by one, then the sampled token
    assert torch.equal(cache.accept_tokens, torch.tensor([[4, 5, 6]]))


def test_forward_with_explicit_positions_refuses_to_skip_unfed_tokens() -> None:
    base = _RecordingBase()
    model = SimpleNamespace(eagle3_base_model=base, config=SimpleNamespace(vocab_size=8))

    class _Cache(MobilintEagle3Cache):
        pass

    cache = object.__new__(_Cache)
    cache.unfed_tokens = torch.tensor([[7, 8]])
    cache.base_layer = SimpleNamespace(update_seen_tokens=lambda n: None)

    with pytest.raises(ValueError, match="2 generated token"):
        llm_eagle3_forward(
            model, input_ids=torch.tensor([[9]]), attention_mask=torch.ones((1, 4)), past_key_values=cache
        )

    assert base.fed == []  # nothing advanced
    assert torch.equal(cache.unfed_tokens, torch.tensor([[7, 8]]))


def test_snapshots_keep_the_unfed_tokens(tmp_path) -> None:
    class _FileMxq(_SlotMxq):
        def dump_cache_memory_to(self, cache_dir, cache_id=0):
            (tmp_path / "kv").write_text(self.kv)

        def load_cache_memory_from(self, cache_dir, cache_id=0):
            self.kv = (tmp_path / "kv").read_text()

    cache = MobilintEagle3Cache(_FileMxq(), _FileMxq())
    cache.accept_tokens = torch.tensor([[4, 5]])
    cache.defer_accept_tokens()

    cache.dump_cache_memory()
    cache.unfed_tokens = None
    cache.load_cache_memory()
    assert torch.equal(cache.unfed_tokens, torch.tensor([[4, 5]]))

    cache.dump_cache_memory_to(str(tmp_path))
    restored = MobilintEagle3Cache(_FileMxq(), _FileMxq())
    restored.load_cache_memory_from(str(tmp_path))
    assert torch.equal(restored.unfed_tokens, torch.tensor([[4, 5]]))
