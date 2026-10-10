"""Beam caches (Whisper, Qwen3-ASR) on the owner table: losing the slot forgets the active history, so the next
forward replays from the start instead of reading another cache's KV."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import torch

from transformers_mblt.models.whisper.modeling_whisper import MobilintWhisperDecoder
from transformers_mblt.utils import cache_residency as residency
from transformers_mblt.utils.cache_utils import MobilintBeamCache, MobilintCache, MobilintWhisperCache


class _DecoderMxq:
    """Decoder MXQ double: records the cache position of every infer and can dump its single KV slot."""

    def __init__(self) -> None:
        self.cache_sizes: list[int] = []
        self.kv = ""

    def infer(self, inputs, output_buffer, cache_size):
        del output_buffer
        self.cache_sizes.append(int(cache_size))
        suffix_length = int(inputs[0].shape[2])
        self.kv = self.kv[: int(cache_size)] + "w" * suffix_length
        return [np.zeros((suffix_length, 4), dtype=np.float32)]

    def dump_cache_memory(self, cache_id: int = 0) -> list[bytes]:
        return [self.kv.encode()]

    def load_cache_memory(self, buffer: list[bytes], cache_id: int = 0) -> None:
        self.kv = buffer[0].decode()


def _decoder(mxq: _DecoderMxq) -> MobilintWhisperDecoder:
    decoder = object.__new__(MobilintWhisperDecoder)
    decoder.npu_backend = SimpleNamespace(mxq_model=mxq, mxq_models=[mxq])

    def embed_token_suffix(self, token_history, start_index, *, device):
        del self
        return torch.ones((1, 1, len(token_history) - start_index, 2), dtype=torch.float32, device=device)

    decoder._embed_token_suffix = MethodType(embed_token_suffix, decoder)
    return decoder


def _step(decoder: MobilintWhisperDecoder, cache: MobilintWhisperCache, token_ids: list[int]) -> None:
    decoder.decoder_forward(
        torch.ones((1, 1, len(token_ids), 2)),
        torch.ones((1, 1, 1, 2)),
        cache,
        torch.arange(len(token_ids)),
        input_ids=torch.tensor([token_ids]),
    )


def test_two_whisper_caches_replay_after_the_other_used_the_slot() -> None:
    mxq = _DecoderMxq()
    decoder = _decoder(mxq)
    a, b = MobilintWhisperCache(mxq), MobilintWhisperCache(mxq)

    _step(decoder, a, [1, 2, 3])
    _step(decoder, a, [4])
    assert mxq.cache_sizes == [0, 3]  # a keeps decoding on its own slot: only the new token

    _step(decoder, b, [7, 8])
    assert not a.is_resident and a.get_active_tokens() == []

    mxq.cache_sizes.clear()
    _step(decoder, a, [5])
    assert mxq.cache_sizes == [0]  # a replays its whole history [1, 2, 3, 4, 5], not on top of b's KV
    assert a.get_active_tokens() == [1, 2, 3, 4, 5] and b.get_active_tokens() == []


def test_beam_cache_and_dense_cache_share_a_slot() -> None:
    mxq = _DecoderMxq()
    beam = MobilintBeamCache(mxq)
    dense = MobilintCache(mxq)
    layer = dense.layers[0]

    residency.acquire(layer, mxq, 0)
    mxq.kv = "dense"
    layer.update_seen_tokens(5)

    beam.bind_active(mxq)  # the dense cache is saved before the beam replays over the slot
    beam.commit_active_tokens([1, 2])
    mxq.kv = "beam"
    assert layer._host_kv == [b"dense"]

    residency.acquire(layer, mxq, 0)  # the dense cache comes back: its KV is loaded, the beam forgets
    assert mxq.kv == "dense" and beam.get_active_tokens() == []


def test_forward_without_a_cache_makes_the_beam_cache_replay() -> None:
    mxq = _DecoderMxq()
    decoder = _decoder(mxq)
    cache = MobilintWhisperCache(mxq)
    _step(decoder, cache, [1, 2])

    residency.bind_cacheless([(mxq, 0)])

    assert cache.get_active_tokens() == []


def test_copy_replays_instead_of_trusting_the_slot() -> None:
    mxq = _DecoderMxq()
    decoder = _decoder(mxq)
    cache = MobilintWhisperCache(mxq)
    _step(decoder, cache, [1, 2])
    copied = cache.copy()
    _step(decoder, cache, [3])  # the original moves on: the slot no longer holds [1, 2] only

    mxq.cache_sizes.clear()
    _step(decoder, copied, [9])
    assert mxq.cache_sizes == [0]  # the copy replays [1, 2, 9]
    assert copied.get_active_tokens() == [1, 2, 9] and cache.get_active_tokens() == []


def test_beam_cache_moves_to_another_decoder_instance() -> None:
    src, dst = _DecoderMxq(), _DecoderMxq()
    cache = MobilintWhisperCache(src)
    _step(_decoder(src), cache, [1, 2])

    _step(_decoder(dst), cache, [3])

    assert dst.cache_sizes == [0]  # the history is replayed on the new instance
    assert cache.is_resident and cache._replay_slot.mxq_model is dst
    assert residency.owner(src, 0) is None


def test_offload_frees_the_slot_and_keeps_no_host_kv() -> None:
    mxq = _DecoderMxq()
    cache = MobilintWhisperCache(mxq)
    cache.bind_active(mxq)
    cache.commit_active_tokens([1, 2])

    cache.offload()

    assert not cache.is_resident and cache.get_active_tokens() == [] and cache.host_nbytes == 0


def test_a_failed_replay_keeps_only_the_shared_prefix_as_active() -> None:
    class _FailingMxq(_DecoderMxq):
        def infer(self, inputs, output_buffer, cache_size):
            raise RuntimeError("infer failed")

    mxq = _FailingMxq()
    decoder = _decoder(mxq)
    cache = MobilintWhisperCache(mxq)
    cache.bind_active(mxq)
    cache.commit_active_tokens([1, 2, 3], source_index=0)
    cache.commit_beam_tokens(0, [1, 2])

    try:
        _step(decoder, cache, [9])  # replays from position 2: overwrites what held token 3
    except RuntimeError:
        pass

    assert cache.get_active_tokens() == [1, 2]
