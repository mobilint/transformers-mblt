# Portable Mobilint Cache — 1단계 상세 설계

상태: 1번 PR 구현 (2026-10-10) · 대상: transformers-mblt · 후속: vllm-mblt 전환(2단계), 실행 API(3단계)

## 1. 목표

Mobilint Cache 객체 하나가 시퀀스 하나의 KV 상태를 온전히 들고 다니게 한다. 모델은 지금 NPU의 각 KV 슬롯을 어느 Cache가 점유하고 있는지 기억하고, `forward`나 `generate`에 다른 Cache가 들어오면 필요한 dump와 load를 스스로 한다.

- 사용자는 `dump_cache_memory()` / `load_cache_memory()`를 직접 부르지 않아도 여러 Cache를 번갈아 쓸 수 있다.
- vllm-mblt는 요청마다 Cache 객체를 하나씩 들고, qbruntime의 dump/load를 직접 부르지 않는다(2단계).
- 모든 Mobilint Cache는 지금처럼 HF `transformers.cache_utils.Cache`의 하위 클래스로 남고, `generate()` / `forward()`의 `past_key_values` 계약은 바뀌지 않는다.

비목표: 여러 Cache를 한 번의 batched infer로 묶는 실행 API(3단계), Whisper/Qwen3-ASR의 HF beam search `reorder_cache`를 자동 교체로 바꾸는 일.

## 2. 현재 상태 (조사 결과 요약)

### 2.1 KV가 qbruntime에 닿는 곳

| # | 위치 | 슬롯 결정 | 갱신 |
| --- | --- | --- | --- |
| a | `MobilintModelMixin.llm_forward._do_infer` (`utils/modeling_utils.py:805-833`) | `mxq_models[0]`, `cache_id` 생략(0) | `update_cache_position` → `layers[0]` |
| b | `MultiSlotDispatcher.dispatch` (`utils/multi_slot_dispatch.py:156-399`) | `pkv.slot_of(row)` → `BatchParam(cache_id=local)` | 호출자가 `update_seen_tokens(dict)` |
| c | Qwen3-VL `TextModel.llm_forward._do_infer` (`models/qwen3_vl/modeling_qwen3_vl.py:1414-1475`) | `get_mxq_model()`, 0 | `update_cache_position` |
| d | Qwen3-ASR `_beam_llm_forward` (`models/qwen3_asr/modeling_qwen3_asr.py:352-367`) | `mxq_model`, 0 | `commit_beam_tokens` / `commit_active_tokens` |
| e | `decoder_forward` (`utils/modeling_utils.py:1741`), Whisper `decoder_forward` (`models/whisper/modeling_whisper.py:262-394`) | `mxq_model`, 0 | `update_cache_position` / beam commit |
| f | EAGLE-3 base `forward` (`utils/eagle3/eagle3_utils.py:539-649`), draft `forward` (`:865-990`) | 각 MXQ, 0 | `update_base_seen_tokens` / `update_draft_seen_tokens` |
| g | MoE `moe_forward` (`utils/modeling_utils.py:1882-1954`) | shared 48개, 0 | `update_cache_position` + `kv_owner` |

한 곳으로 모이는 지점이 없다. 행 → `(Model, cache_id)` 매핑을 아는 곳은 (b)뿐이고 나머지는 슬롯 0을 가정한다.

### 2.2 지금도 조용히 틀리는 경우

Cache A로 프리필한 뒤 Cache B로 `generate()`(또는 Cache 없는 forward)를 하고 A를 다시 넘기면, A는 `_seen_tokens > 0`인 채로 B의 KV 위에서 `cache_size = A.get_seq_length()`로 추론한다. 출력이 조용히 틀린다. `_prepare_cache_for_generation`(`utils/generation_utils.py:456-482`)은 넘겨받은 `MobilintCache`를 검사 없이 그대로 쓴다. 이를 감지하는 곳은 MoE의 `kv_owner`(`generation_utils.py:532-557`)뿐이고, 그마저 이어 쓰지 않고 리셋 후 다시 프리필한다.

### 2.3 수명과 스레드

- backend(`MobilintNPUBackend`, mblt_npu)는 config에 붙어 있고 `dispose()` → `create()`를 거쳐도 객체는 살아남는다. Model 핸들은 그때 새로 만들어지므로 `is` 비교가 달라진다.
- 멀티 그룹 dispatch만 Model마다 스레드를 쓴다(`multi_slot_dispatch.py:331-361`). Cache 갱신은 모든 future가 끝난 뒤 호출 스레드에서 한다. 그 밖의 경로는 단일 스레드다.
- 한 모델을 여러 스레드에서 동시에 `generate()`하는 경우는 지금도 보호되지 않는다.

## 3. 핵심 개념

### 3.1 단위: 물리 슬롯과 `MobilintLayer`

- **물리 슬롯**: `(qbruntime.Model, cache_id)`. KV 메모리 한 칸.
- **`MobilintLayer`**: 한 시퀀스가 한 물리 슬롯에 대해 가진 KV 상태. 지금도 `mxq_model`, `cache_id`, `_seen_tokens`, `buffer`, `buffer_seq_length`를 가진다.
- Cache 클래스마다 layer 구성만 다르다.

| Cache | 시퀀스당 layer |
| --- | --- |
| `MobilintCache`, `MobilintDeepStackCache` | 행마다 1개 (텍스트 MXQ) |
| `MobilintEagle3Cache` | base 1개 + draft 1개 |
| `MobilintMixtureOfExpertsCache` | shared MXQ마다 1개 (Qwen3-30B-A3B: 48개) |
| `MobilintWhisperCache` / `MobilintBeamCache` | decoder 1개 (beam 정보는 Cache가 따로 보관) |

소유권은 **layer 단위**로 추적한다. 그러면 EAGLE-3의 base/draft, MoE의 48층, 멀티카드 배치의 여러 행이 같은 규칙으로 처리된다.

### 3.2 layer 상태

layer는 다음 셋 중 하나다. 별도 enum을 두지 않고 owner 표와 기존 필드로 판정한다.

| 상태 | 판정 | 의미 |
| --- | --- | --- |
| **resident** | owner 표에서 자기 슬롯의 owner가 자기 자신 | NPU가 최신. host `buffer`는 낡았을 수 있음 |
| **offloaded** | resident가 아니고 `buffer_seq_length >= _seen_tokens > 0` | host `buffer`에 필요한 KV가 다 있음 |
| **empty** | `_seen_tokens == 0` | 복원할 KV 없음 |

추가로 `kv_valid: bool`을 둔다. `fake_prefill()`처럼 길이만 맞추고 KV 내용은 의미가 없는 layer는 `False`이며, 복원할 때 load하지 않는다. TPS 벤치마크의 기존 동작을 유지하기 위한 것이다.

### 3.3 owner 표 (`MobilintKVResidency`)

backend 객체 하나당 하나 둔다. mblt_npu를 고치지 않도록 `transformers_mblt/_npu.py`가 dispatcher를 붙이는 것과 같은 방식으로 property를 설치한다(`backend._mblt_kv_residency`).

```python
class MobilintKVResidency:
    # key: (id(model), cache_id) -> (weakref(model), weakref(MobilintLayer))
    def owner(self, model, cache_id) -> Optional[MobilintLayer]
    def claim(self, layer) -> None
    def release(self, layer) -> None          # layer가 owner일 때만
    def clear(self) -> None                   # dispose/create 시
```

- 키에 Model 핸들의 약한 참조도 같이 저장해 두고, 조회할 때 핸들이 같은 객체인지 확인한다. dispose 후 다시 만들어진 Model은 `id()`가 재사용되어도 다른 객체로 취급한다. `qbruntime.Model`은 weakref를 지원한다(확인함).
- owner는 약한 참조로 든다. Cache가 GC되면 그 슬롯은 dump 없이 빈 것으로 본다.
- `dispose()` 경로에서 `clear()`한다. backend가 Model 목록을 비울 때 같이 비우면 되고, MoE backend의 `_release_buffers`가 `kv_owner`를 지우는 자리와 같다.

## 4. 바인딩 규칙

### 4.1 `acquire(layers)` — infer 직전에 호출

```text
for layer in layers:                       # 이번 infer가 쓰는 layer만
    slot = (layer.mxq_model, layer.cache_id)
    cur = residency.owner(*slot)
    if cur is layer: continue              # 이미 올라와 있음: 비용 0
    if cur is not None:                     # 다른 시퀀스가 점유 중 → evict
        cur.on_evict()                      # 기본: _seen_tokens>0 && kv_valid면 dump → cur.buffer, buffer_seq_length=_seen_tokens
    if layer._seen_tokens > 0 and layer.kv_valid:
        require layer.buffer_seq_length >= layer._seen_tokens   # 없으면 MobilintCacheStateError
        layer.mxq_model.load_cache_memory(layer.buffer, layer.cache_id)
    residency.claim(layer)
```

- **필요할 때만 저장한다.** 같은 Cache로 계속 디코드하면 dump/load가 일어나지 않는다. 교체가 일어날 때만 비용이 든다. MoE는 교체 한 번에 48 × 4 MiB를 dump하고 load한다.
- **길이를 되감는 경우(`crop`, `_seen_tokens` 축소)에도 안전하다.** qbruntime은 `cache_size`로 읽을 접두사를 정하므로, buffer가 더 길어도 load해도 된다.
- **load 조건은 `_seen_tokens > 0`이다.** 지금 `MobilintLayer.load_cache_memory()`가 길이 0이면 건너뛰는 것과 같다. 길이 0인 시퀀스는 앞부분부터 덮어쓰므로 load할 필요가 없다.

### 4.2 Cache 없는 forward

`past_key_values=None`인 forward도 슬롯의 KV를 덮어쓴다. 그 슬롯의 owner를 evict(dump)한 뒤 owner를 `None`으로 둔다. 지금 dense 모델의 Cache 없는 forward는 공유 `self._cache`를 reset해서 쓰므로, 실제로는 `self._cache`의 layer가 그 슬롯을 차지한다(4.1과 같은 경로).

### 4.3 실패한 forward

KV 쓰기는 항상 `cache_size` 이후 위치에 일어난다. 그래서 forward가 중간에 실패해도 `[0, _seen_tokens)` 구간은 그대로 남는다. 커서는 성공한 뒤에만 전진하므로, **실패해도 owner를 유지한다.** MoE의 "실패하면 `kv_owner = None`" 규칙보다 단순하고, 다음 호출이 그대로 이어서 진행할 수 있다.

예외가 하나 있다. MoE에서 일부 층만 실행된 채 실패하면 층별 KV 끝부분의 상태가 서로 다르다. 이 역시 커서 이후 영역이라 다음 forward가 같은 위치부터 덮어쓰므로 문제없다. 기존 테스트 `test_mixture_of_experts_runtime.py:419`(실패 후 `kv_owner is None`)는 새 규칙에 맞게 고친다.

### 4.4 evict 방식: dump 대신 무효화

Cache 종류마다 `on_evict()`를 고를 수 있다.

| Cache | `on_evict()` | 이유 |
| --- | --- | --- |
| 기본 (`MobilintCache` 등) | dump | 일반 시퀀스 |
| `MobilintBeamCache`, `MobilintWhisperCache` | `_active_token_history`와 `_active_source_index` 무효화, dump 없음 | 이 Cache는 NPU에 무엇이 있는지 토큰 이력으로 추적하고, 다음 `decoder_forward`가 다른 부분만 다시 프리필한다. 무효화하면 공통 접두사가 0이 되어 전체를 다시 프리필하므로 정확하다. 1단계 범위(reorder 자동 교체 제외)와도 맞는다 |

`MobilintBeamCache`는 layer를 여러 개(cache_id `0..B-1`) 만들지만 실제 infer는 layer 0만 쓴다. 그래서 4.1의 `acquire`도 layer 0만 대상으로 한다.

## 5. 바인딩 지점 (choke point)

2.1의 각 경로에 다음 공통 함수 하나를 넣는다.

```python
# transformers_mblt/utils/cache_residency.py
def bind_cache_layers(backend, layers: Iterable[MobilintLayer]) -> None
def bind_cacheless(backend, slots: Iterable[tuple[Model, int]]) -> None
```

| 경로 | 넘기는 layer |
| --- | --- |
| (a) 단일 행 `llm_forward` | `past_key_values.layers[0]` |
| (b) `MultiSlotDispatcher.dispatch` | 이번 호출의 행에 해당하는 layer들. 스레드를 나누기 **전에** 호출 스레드에서 실행 |
| (c) Qwen3-VL 단일 행 | `layers[0]` |
| (d) Qwen3-ASR beam, (e) Whisper | `layers[0]` (beam 규칙 4.4) |
| (e) 일반 `decoder_forward` | `layers[0]` |
| (f) EAGLE-3 | base forward는 `base_layer`, draft forward는 `draft_layer` |
| (g) MoE | 48개 shared layer 전부, forward 시작 시 한 번 |

행 → layer 매핑은 Cache가 정한다. `MultiSlotDispatcher.slot_of`는 이미 `pkv.slot_of(row)`에 위임하므로, `slot_of`가 `divmod` 대신 `layers[row]`의 `(mxq_model, cache_id)`를 돌려주게 바꾼다. 그러면 6.2의 슬롯 지정 Cache도 dispatcher를 고치지 않고 동작한다.

## 6. Cache 쪽 API 변경

### 6.1 유지하는 것 (HF 호환)

- 모든 클래스는 HF `Cache`를 상속한다. `get_seq_length`, `update`, `reset`, `crop`, `reorder_cache`, `layers` 등 HF가 부르는 표면은 그대로 둔다.
- `dump_cache_memory()` / `load_cache_memory()` / `*_to` / `*_from`의 의미도 유지한다.
  - `dump_cache_memory()`: resident면 dump해서 buffer를 갱신한다. owner는 그대로다. offloaded면 아무것도 하지 않는다.
  - `load_cache_memory()`: 4.1의 `acquire`를 호출한다. 즉 지금처럼 즉시 NPU에 올린다.
  - `load_cache_memory_from()`: 파일에서 buffer를 읽고 owner를 무효화한다. 다음 forward 때 load된다.

### 6.2 추가하는 것

```python
class MobilintCache(Cache):
    def __init__(self, mxq_models, per_model_batch=1, *, batch_size=None, slots=None): ...
        # slots=[(model_idx, cache_id), ...]: 행 i를 지정한 물리 슬롯에 둠. 기본은 지금과 같은 divmod 배치
    @property
    def is_resident(self) -> bool         # 모든 layer가 resident
    def offload(self) -> None             # resident layer를 dump하고 owner를 놓음. 다른 Cache가 슬롯을 쓸 때 덤프 비용을 미리 치르고 싶을 때
    @property
    def host_nbytes(self) -> int          # host buffer 총 크기 (vLLM 2단계의 메모리 계산용)
    def copy(self) -> Self                # 아래 6.3
```

EAGLE-3, MoE, Whisper도 같은 이름으로 구현한다. `slots`는 슬롯이 여러 개인 `MobilintCache` 계열에만 의미가 있다.

### 6.3 `copy()` = 스냅샷

- resident layer라면 먼저 dump해서 buffer를 최신으로 만든다. 원본은 owner를 유지한다.
- 사본은 새 `MobilintLayer`(새 identity)를 가지고, buffer는 복사한다. qbruntime이 dump할 때마다 새 bytes 리스트를 만들어 주므로, 리스트를 얕게 복사해도 안전한지 구현 때 확인한다.
- 사본은 offloaded 상태로 시작하므로, 처음 forward할 때 load된다.
- EAGLE-3의 tree 상태와 Beam의 이력 같은 비KV 상태는 지금 `copy()` 규칙을 따른다.

## 7. 모델 쪽 상태: `rope_deltas`

Qwen2-VL과 Qwen3-VL은 M-RoPE 위치 보정값 `rope_deltas`를 **Cache가 아니라 모델**(`self.model.rope_deltas`)에 둔다. upstream transformers도 똑같다.

- 프리필 때(`past_key_values_length == 0` 또는 값이 없을 때) `get_rope_index()`로 계산해 `self.rope_deltas`에 저장한다. 이후 디코드는 `past_len + rope_deltas`로 위치를 만든다.
  - 5.17: `models/qwen3_vl/modeling_qwen3_vl.py:1125-1147`, `_prepare_position_ids_for_generation` `:1395-1424`
  - 4.57: `:1200-1212`
- Cache를 번갈아 쓰면 마지막 프리필의 값이 남는다. upstream에도 같은 잠재 버그가 있다. 5.17은 출력 필드 `rope_deltas`를 deprecated로 바꾸고 `model.base_model.rope_deltas`를 쓰라고 안내한다.
- vLLM은 이 값을 요청별로 관리한다. vllm-mblt worker도 `RequestState.mrope_position_delta`로 따로 계산하므로 2단계에는 영향이 없다.

방침: **계산은 하나로 통일하고, 보관 위치는 실행 경로마다 둔다.**

- 계산: `get_rope_index()` / `get_mrope_input_positions()`가 유일한 계산 함수다. HF 경로와 vLLM 경로가 같은 함수를 쓴다. vllm-mblt worker가 지금 이미 그렇게 하고 있다.
- 보관: 시퀀스마다 하나씩 있어야 하는 값이다. HF 경로에서 시퀀스를 대표하는 객체는 Mobilint Cache이고, vLLM 경로에서는 요청 상태다. 그래서 HF 경로는 Cache에, vLLM 경로는 지금처럼 `RequestState`에 둔다.
- 3단계에서 vllm-mblt가 요청마다 Cache 객체를 들게 되면, 같은 칸(`model_state`)을 쓰도록 합쳐 두 경로가 한 곳을 보게 한다.

1단계 구현: Cache에 모델 상태를 보관하는 범용 칸(`model_state: dict`)을 두고, 모델 mixin에 선택적 hook 두 개를 둔다.

```python
class MobilintModelMixin:
    def _export_cache_bound_state(self) -> dict      # 기본: {}
    def _import_cache_bound_state(self, state) -> None
```

- 바인딩 때 다른 Cache로 바뀌면, 이전 Cache의 `model_state`에 현재 값을 저장하고 새 Cache의 값을 복원한다.
- Qwen2-VL과 Qwen3-VL만 `rope_deltas`로 구현한다. upstream보다 한 걸음 더 나간 동작이다. HF 계약을 깨지는 않는다.

## 8. 클래스별 정리

| Cache | layer | `on_evict` | 바뀌는 점 |
| --- | --- | --- | --- |
| `MobilintCache` | 행마다 | dump | `slots=` 추가, `slot_of`가 layer 기준 |
| `MobilintDeepStackCache` | 상속 | dump | DeepStack 텐서는 프리필 동안만 쓰므로 바인딩과 무관. `copy`는 지금처럼 복제 |
| `MobilintEagle3Cache` | base + draft | dump | 아래 8.1 |
| `MobilintMixtureOfExpertsCache` | shared 48 | dump | `kv_owner`를 owner 표로 대체 (8.2) |
| `MobilintWhisperCache` / `MobilintBeamCache` | layer 0 | 무효화 | `encoder_source_count` 등 기존 상태는 Cache와 함께 이동 |

### 8.1 EAGLE-3

- tree 검증은 확정된 길이 너머에 추측 KV를 쓴다(base: accept와 tree 전체, draft: 깊이 확장). dump는 슬롯 전체를 담고 다음 스텝이 확정 위치부터 덮어쓰므로, 교체해도 정확하다.
- `accept_tokens`는 한 `generate()` 루프 안에서만 non-None이고, 정상 종료 경로는 모두 반영한 뒤 비운다. 예외로 중단되면 남을 수 있다.
- 다른 Cache로 바뀔 때 `accept_tokens`가 남아 있어도 그 Cache의 상태이므로 함께 이동한다. 별도 처리는 필요 없다.
- **종료 경로의 커서를 명확하게 한다.** 종료 경로(`tree_decoding.py:631`, `generation_utils.py:1124`)는 생성이 끝났으므로 추가 forward 없이 승인 토큰 L개를 길이에 반영한다. 이때 base KV `[committed, committed+L)`에는 tree 노드가 tree 순서대로 들어 있고, 그중 위치 `committed`(tree root)만 승인 경로와 일치한다.
  - 지금 다음 `generate()`의 `initialize_tree`는 입력을 `input_ids[:, base_seq_length:]`부터 넣는다(`tree_decoding.py:377-381`). 그래서 이 L개는 다시 통과되지 않고, 일치하지 않는 KV가 그대로 쓰인다.
  - 변경: 종료 경로에서는 **실제로 맞는 KV만 길이에 반영**한다(root 1개). 나머지 L−1개는 커서 뒤에 둔다. 다음 `generate()`가 같은 `input_ids[:, base_seq_length:]` 규칙으로 이들을 자연스럽게 다시 통과시키므로, 생성이 끝날 때 추가 forward는 없다.
  - draft도 같은 기준으로 맞춘다. `sync_draft_seq_length_to_base`가 draft가 실제로 본 길이를 넘지 않게 한다.
  - portable Cache와 vLLM 프리픽스 스냅샷은 `get_seq_length()`를 "이만큼의 KV가 유효하다"는 뜻으로 믿고 복사·재사용한다. 그래서 커서가 실제 유효 KV와 일치해야 한다.
- EAGLE-3 `_get_cache`(`generation_utils.py:706-713`)에 topology 확인을 추가한다. 지금 다른 계열에는 있고 여기만 없다.

### 8.2 MoE

- `npu_backend.kv_owner`와 `MobilintMixtureOfExpertsCache.owner`, `_claim_npu_kv`, `MobilintMixtureOfExpertsGenerationMixin`의 "덮어써졌으면 리셋" 분기를 owner 표로 대체한다.
- `kv_owner`는 읽기 전용 호환 property로 남긴다. 48층의 owner가 모두 같은 Cache면 그 Cache를, 아니면 `None`을 돌려준다.
- 동작 변화: 다른 Cache가 끼어들었던 Cache를 다시 쓰면 지금은 리셋 후 다시 프리필하지만, 이제는 load해서 이어 간다. 관련 테스트(`test_mixture_of_experts_runtime.py:306-441`)를 새 의미로 고친다.

## 8.3 다른 모델 인스턴스로 옮기기

같은 체크포인트를 두 번 로드한 인스턴스 A와 B가 있을 때, A에서 만든 Cache를 B의 `forward`나 `generate`에 넘길 수 있게 한다. 예를 들어 다른 NPU나 다른 코어에 띄운 두 인스턴스가 있는 경우다.

**필요한 것**

- **소유 기록을 layer에도 둔다.** layer가 지금 어느 backend의 어느 슬롯에 올라가 있는지(`_resident_at = (weakref(residency), model, cache_id)`)를 기억한다. B가 바인딩할 때 A의 owner 표를 몰라도, layer를 통해 A에게 dump를 요청할 수 있다. 소유권은 owner 표와 layer 양쪽에 같이 기록하고 함께 갱신한다.
- **슬롯을 다시 배정한다.** layer의 `mxq_model`/`cache_id`를 B의 슬롯으로 바꾼다. 1행 Cache는 B의 기본 배치(또는 `slots=`)를 따르고, 여러 행 Cache는 B의 용량이 행 수 이상이어야 한다.
- **KV 형식이 호환되는지 확인한다.** dump 결과는 MXQ 컴파일 결과에 묶여 있다. layer에 원래 MXQ의 서명(MXQ 경로와 파일 크기 또는 해시, KV 버퍼 정보 `get_cache_infos()`, `max_cache_size`, core mode)을 저장해 두고, B의 MXQ와 다르면 `MobilintCacheStateError`로 거부한다. W8 Cache를 W4 인스턴스에 넣는 실수를 막는다.

**동작**

```text
B.acquire(layer):
    if layer가 다른 backend(A)에 resident: A 쪽에서 evict (dump → layer.buffer), A의 owner 표에서 해제
    if layer의 MXQ 서명 != B 슬롯의 MXQ 서명: 오류
    layer를 B의 슬롯으로 재배정
    이후는 4.1과 같음 (B 슬롯의 기존 owner evict → load → claim)
```

- 비용: 옮길 때 A에서 dump 한 번, B에서 load 한 번.
- A가 이미 dispose됐으면 A에 올라가 있던 KV는 사라졌으므로, layer는 마지막 host buffer가 있을 때만 옮길 수 있다. 그렇지 않으면 오류를 낸다. `test_cache.py:60-155`가 지금 손으로 하는 "dispose → create → 핸들 재바인딩 → load"를 이 경로가 자동으로 해 주게 된다.
- 지금 다른 인스턴스의 Cache를 받으면 `cache_matches_backend_topology`가 Cache를 새로 만들어 내용을 버리고, MoE는 경고 후 교체한다. 이것도 옮기기로 바뀐다. 서명이 다를 때만 지금처럼 거부한다.
- 범위: 같은 프로세스 안에서만 옮긴다. 다른 프로세스로 옮기는 것은 기존 `dump_cache_memory_to` / `load_cache_memory_from`(파일)로 한다. 필요하면 나중에 pickle을 지원한다.

## 9. 호환성과 동작 변화

| 상황 | 지금 | 1단계 후 |
| --- | --- | --- |
| A → B → A로 `generate` | A가 B의 KV 위에서 조용히 틀림 | B 시작 때 A를 dump, A 재개 때 load. 정확 |
| 같은 Cache로 연속 `generate`·디코드 | 비용 없음 | 비용 없음 |
| 사용자가 직접 dump/load | 동작 | 그대로 동작. 자동 교체와 함께 써도 안전 |
| `fake_prefill` 벤치마크 | KV load 없음 | `kv_valid=False`라 load 없음 |
| MoE 다른 Cache 재사용 | 리셋 후 재프리필 | load 후 이어 감 |
| 실패한 forward | MoE만 `kv_owner=None` | owner 유지 (4.3) |
| beam/Whisper | 이력 기반 재프리필 | 같음. 다른 Cache가 끼면 전체 재프리필 |
| 인스턴스 A의 Cache를 B에 넘김 | 새 Cache로 바꿔 내용을 버림 (MoE는 경고) | A에서 dump, B에서 load로 옮김. MXQ 서명이 다르면 오류 |
| 동시 다중 스레드 `generate` | 보호 안 됨 | 보호 안 됨 (문서에 명시) |

## 10. 테스트 계획

**단위 테스트** (`tests/transformers/_fake_mxq.py` 확장: fake `dump_cache_memory` / `load_cache_memory`가 `(cache_id, 길이)`를 기록)

- A → B → A 교체 순서: B 시작 때 A dump, A 재개 때 B dump와 A load. 같은 Cache로 연속하면 호출 0회.
- 길이 0 Cache는 load하지 않음, `fake_prefill`은 load하지 않음.
- Cache GC 후 그 슬롯을 쓰면 dump 없음.
- dispose → create 후 owner 표가 비어 있음. 이전 launch의 Cache를 쓰면 기존 topology 오류.
- 실패한 forward 후 owner 유지, 재시도하면 같은 위치부터.
- `slots=`로 지정한 행이 dispatcher에서 올바른 `BatchParam.cache_id`로 감.
- EAGLE-3: base와 draft가 독립적으로 교체되고, tree 상태는 Cache를 따라감.
- MoE: 48층이 한 번에 교체되고, `kv_owner` 호환 property가 동작함.
- beam: 교체되면 active 이력이 무효화되어 전체 재프리필.
- `rope_deltas` hook: Cache를 번갈아 쓰면 각자의 값이 복원됨.
- 인스턴스 옮기기: A에 올라가 있던 Cache를 B에 넘기면 A dump → B load, A의 owner 해제. 서명이 다르면 오류. A가 dispose된 뒤에는 host buffer로만 옮길 수 있음.
- EAGLE-3 종료 경로: 길이가 실제 유효 KV와 일치하고, 다음 `generate()`가 나머지 승인 토큰을 다시 통과시킴.
- 기존 계약 테스트(2.1의 각 경로, `test_cache_utils.py`, `test_generation_cache_initialization.py`, `test_multi_slot_dispatcher.py`, `test_cache.py`의 수동 dump/load)가 계속 통과해야 함.

**NPU 확인** (노드 `beomsu`)

각 모델에서 Cache A로 프리필하고 디코드 몇 스텝을 한 뒤 Cache B로 다른 프롬프트를 생성한다. 그다음 A로 돌아가 디코드한 결과가, A만 연속으로 돌린 결과와 logits(cos ≥ 0.999) 및 greedy 토큰이 같은지 확인한다.

- `mobilint/Llama-3.2-1B-Instruct` (단일 행, 그리고 같은 체크포인트를 다른 코어에 띄운 두 인스턴스 사이 옮기기)
- `mobilint/Llama-3.1-8B-Instruct-Batch16` (슬롯 여러 개, `slots=`)
- `mobilint/Qwen3-30B-A3B-aries-rb` (MoE 48층)
- `mobilint/EAGLE3-JPharmatron-7B` (base와 draft)
- Qwen3-VL과 Whisper는 로컬 캐시에 있는 릴리스로 확인한다. 없으면 받아 온다.

## 11. 작업 순서 (PR 단위)

1. **owner 표와 바인딩 핵심**: `cache_residency.py`, `MobilintLayer`(`kv_valid`, `on_evict`, `_resident_at`, MXQ 서명), backend property 설치, dispose 연동, dense (a)(b) 경로, `copy`/`offload`/`host_nbytes`/`slots=`, 인스턴스 옮기기, 단위 테스트.
2. **VL·ASR·Whisper·decoder**: (c)(d)(e) 경로, beam 무효화 규칙, `rope_deltas` hook.
3. **EAGLE-3**: (f) 경로, `_get_cache` topology 확인, 종료 경로 커서 정리(8.1).
4. **MoE 이전**: (g) 경로, `kv_owner`를 owner 표로 대체, 테스트 갱신.
5. **문서**: `AGENTS.md`, skill 문서, README, CHANGELOG. NPU 검증 결과를 PR에 첨부한다.

각 PR은 독립적으로 병합할 수 있게 한다. 1번만 들어가도 dense 경로의 2.2 버그가 고쳐진다.

## 12. 결정 사항

1. **실패한 forward**: owner를 유지한다(4.3).
2. **`rope_deltas`**: 계산은 하나의 함수로 통일하고, 보관은 HF 경로는 Cache, vLLM 경로는 요청 상태에 둔다. 3단계에서 vLLM도 Cache를 들게 되면 같은 칸으로 합친다(7).
3. **EAGLE-3 종료 경로**: 추가 forward 없이 커서만 실제 유효 KV에 맞춘다(8.1).
4. **이름**: `offload()`, `is_resident`, `slots=`, `MobilintKVResidency`.
