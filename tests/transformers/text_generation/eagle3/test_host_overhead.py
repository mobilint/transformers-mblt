"""Hardware-free tests for the EAGLE-3 host-overhead paths in ``eagle3_utils``.

- ``infer_keep_cpu_awake`` must return the same result as a direct ``infer``, re-raise its errors, honour the
  ``MBLT_EAGLE3_KEEP_CPU_AWAKE`` modes, and keep NPU calls from different threads concurrent.
- ``CachedRotaryEmbedding`` built lazily (after meta init) must not rebuild its table as decode positions grow.
"""

from __future__ import annotations

import gc
import threading
import time
import weakref

import pytest
import torch

from transformers_mblt.utils.eagle3 import eagle3_utils


class _FakeMxq:
    def __init__(self, delay: float = 0.0, error: Exception | None = None) -> None:
        self.delay = delay
        self.error = error
        self.threads: list[str] = []

    def infer(self, inputs, outputs=None, cache_size=0):
        self.threads.append(threading.current_thread().name)
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return [inputs[0] * 2, cache_size]


@pytest.fixture
def keep_awake_on(monkeypatch):
    monkeypatch.setattr(eagle3_utils, "_KEEP_CPU_AWAKE_MODE", "1")


def test_keep_awake_returns_infer_result_from_worker(keep_awake_on):
    mxq = _FakeMxq()
    out = eagle3_utils.infer_keep_cpu_awake(mxq, [3], None, 7)
    assert out == [6, 7]
    assert mxq.threads == ["mblt-eagle3-infer"]


def test_keep_awake_reraises_infer_error(keep_awake_on):
    mxq = _FakeMxq(error=RuntimeError("npu failed"))
    with pytest.raises(RuntimeError, match="npu failed"):
        eagle3_utils.infer_keep_cpu_awake(mxq, [1])
    # The worker survives the error and serves the next call.
    assert eagle3_utils.infer_keep_cpu_awake(_FakeMxq(), [2]) == [4, 0]


def test_keep_awake_worker_drops_finished_call(keep_awake_on):
    mxq = _FakeMxq()
    eagle3_utils.infer_keep_cpu_awake(mxq, [1])
    ref = weakref.ref(mxq)
    del mxq
    gc.collect()
    assert ref() is None


def test_keep_awake_off_calls_infer_directly(monkeypatch):
    monkeypatch.setattr(eagle3_utils, "_KEEP_CPU_AWAKE_MODE", "0")
    mxq = _FakeMxq()
    assert eagle3_utils.infer_keep_cpu_awake(mxq, [5]) == [10, 0]
    assert mxq.threads == [threading.current_thread().name]


def test_keep_awake_keeps_calls_from_different_threads_concurrent(keep_awake_on):
    delay = 0.3
    results: list = []

    def run() -> None:
        results.append(eagle3_utils.infer_keep_cpu_awake(_FakeMxq(delay=delay), [1]))

    threads = [threading.Thread(target=run) for _ in range(2)]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - start
    assert results == [[2, 0], [2, 0]]
    # One shared worker would serialize the two calls (>= 2 * delay).
    assert elapsed < 1.6 * delay


def test_keep_awake_auto_stays_off_without_load_average(monkeypatch):
    # Windows has no os.getloadavg; auto mode must call infer directly instead of raising.
    monkeypatch.setattr(eagle3_utils, "_KEEP_CPU_AWAKE_MODE", "auto")
    monkeypatch.setattr(eagle3_utils, "_host_idle_cache", (float("-inf"), True))
    monkeypatch.delattr(eagle3_utils.os, "getloadavg", raising=False)
    mxq = _FakeMxq()
    assert eagle3_utils.infer_keep_cpu_awake(mxq, [5]) == [10, 0]
    assert mxq.threads == [threading.current_thread().name]


@pytest.mark.parametrize("mode", ["0", "1"])
def test_keep_awake_records_infer_time_only(monkeypatch, mode):
    monkeypatch.setattr(eagle3_utils, "_KEEP_CPU_AWAKE_MODE", mode)
    recorded: list = []
    eagle3_utils.infer_keep_cpu_awake(_FakeMxq(delay=0.05), [1], record_npu_time=recorded.append)
    assert len(recorded) == 1
    assert 0.05 <= recorded[0] < 0.5


def test_keep_awake_interrupt_waits_for_infer_then_propagates(keep_awake_on, monkeypatch):
    delay = 0.2
    mxq = _FakeMxq(delay=delay)
    real_sleep = time.sleep
    interrupted = []

    def interrupting_sleep(seconds):
        if not interrupted:  # interrupt the first poll, as a Ctrl+C would
            interrupted.append(True)
            raise KeyboardInterrupt
        real_sleep(seconds)

    monkeypatch.setattr(eagle3_utils.time, "sleep", interrupting_sleep)
    start = time.perf_counter()
    with pytest.raises(KeyboardInterrupt):
        eagle3_utils.infer_keep_cpu_awake(mxq, [1])
    # The caller left only after the NPU call it started had finished.
    assert time.perf_counter() - start >= delay
    assert mxq.threads == ["mblt-eagle3-infer"]


def test_lazy_rotary_table_is_built_once_while_decoding():
    rope = eagle3_utils.CachedRotaryEmbedding(64, 4096, base=10000)
    assert rope.position_table is not None
    reference = rope.position_table.copy()
    # Simulate the meta-init path: the table is built lazily on the first forward.
    rope.position_table = None
    builds = []
    original = rope._build_position_table

    def counting_build(*args, **kwargs):
        builds.append(rope.max_seq_len)
        return original(*args, **kwargs)

    rope._build_position_table = counting_build
    x = torch.zeros(1, 1, 64)
    for length in (19, 20, 24, 40, 300):
        position_ids = torch.arange(length).view(1, -1)
        out = rope(x, position_ids)
        assert out.shape[2] == length
        assert (out[0, 0] == reference[:length]).all()
    assert len(builds) == 1
    assert rope.max_seq_len == 4096


def test_sampler_slices_only_plain_hf_warpers():
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    from transformers_mblt.utils.eagle3.tree_decoding import _sliceable_top_k

    temp, topk, topp = TemperatureLogitsWarper(0.5), TopKLogitsWarper(100), TopPLogitsWarper(0.9)
    assert _sliceable_top_k(LogitsProcessorList([temp, topk, topp])) == (100, 1)
    assert _sliceable_top_k(LogitsProcessorList([temp, topp, topk])) is None
    assert _sliceable_top_k(LogitsProcessorList([RepetitionPenaltyLogitsProcessor(1.1), topk])) is None
    assert _sliceable_top_k(LogitsProcessorList([TopKLogitsWarper(100, filter_value=-5.0)])) is None


def test_sampler_keeps_ties_created_by_temperature():
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
    )

    from transformers_mblt.utils.eagle3.tree_decoding import _sample_next_token_from_processor

    # Adjacent float32 values that differ raw but merge once divided by 3: HF's TopK(1) then keeps both.
    a = torch.tensor(0.9999998807907104)
    b = torch.tensor(0.9999998211860657)
    assert b < a and b / 3 == a / 3
    logits = torch.full((8,), -10.0)
    logits[2], logits[5] = a, b
    processors = LogitsProcessorList([TemperatureLogitsWarper(3.0), TopKLogitsWarper(1)])
    torch.manual_seed(0)
    drawn = {int(_sample_next_token_from_processor(logits, processors)) for _ in range(200)}
    assert drawn == {2, 5}


def test_sampler_top_p_ties_keep_higher_token_ids():
    from transformers.generation.logits_process import LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper

    from transformers_mblt.utils.eagle3.tree_decoding import _sample_next_token_from_processor

    # TopK(3) keeps ids 1, 3, 5 with probs 0.576, 0.212, 0.212. TopP(0.7) drops one of the equal pair; as in the
    # fused top-k/top-p sampler, the lower token id goes.
    logits = torch.full((8,), -10.0)
    logits[1], logits[3], logits[5] = 2.0, 1.0, 1.0
    processors = LogitsProcessorList([TopKLogitsWarper(3), TopPLogitsWarper(0.7)])
    torch.manual_seed(0)
    drawn = {int(_sample_next_token_from_processor(logits, processors)) for _ in range(200)}
    assert drawn == {1, 5}
