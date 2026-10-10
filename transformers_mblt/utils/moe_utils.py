"""NPU backend for Mobilint Mixture-of-Experts releases.

A MoE release is not one MXQ but ``num_layers * (1 + num_experts) + 1`` of them: a ``shared`` MXQ per decoder layer
(attention, router gate and residual), an MXQ per ``(layer, expert)`` pair, and an ``lm_head`` MXQ.
:class:`MobilintNPUBackend` hosts ``N`` slots of a single MXQ, so it cannot host this layout directly. Instead the
three role backends on :class:`~transformers_mblt.utils.configuration_utils.MobilintMixtureOfExpertsConfigMixin`
describe *where* each role runs (canonical ``"d:c:k"`` target cores, board, revision) and
:class:`MobilintMixtureOfExpertsNPUBackend` loads every MXQ against those specs, sharing one ``Accelerator`` per
device across roles.

The backend owns NPU I/O only: model loading, placement, buffer management, quantization-scale bookkeeping, and
teardown. Routing (softmax, top-k, weighting) and the layer loop live in
:class:`~transformers_mblt.utils.modeling_utils.MobilintMixtureOfExpertsModelMixin`.

Releases ship the shared/expert MXQs packed: one uncompressed ``.mxqpack`` per layer holding the shared MXQ and then
every expert MXQ, each starting on an ``alignment`` boundary, described by a separate index JSON
(:class:`MobilintMixtureOfExpertsPackIndex`). Thousands of small Hub files would otherwise exhaust the Hub API rate
limit on download. A packed member never exists as a regular file next to its pack: qbruntime only opens MXQs by
path and re-reads them during ``launch()``, so each member is materialized just before ``Model()`` and released
right after ``launch()`` (see :class:`_MxqPackReader`).
"""

from __future__ import annotations

import errno
import gc
import json
import logging
import mmap
import os
import re
import sys
import tempfile
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Sequence

import numpy as np
import qbruntime
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from qbruntime import Accelerator, Model, ModelConfig, QbRuntimeError

from .._npu import MobilintBackendAllocError, MobilintNPUBackend

logger = logging.getLogger(__name__)

# Activation slots per role. The thousands of shared/expert models run strictly one inference at a time, so they
# launch without activation buffers to keep the per-model NPU memory overhead down.
_ROLE_ACTIVATION_SLOTS: dict[str, int] = {"shared": 0, "expert": 0, "lm_head": 1}
_TEMPLATE_FIELD_RE = re.compile(r"\{[^{}]*\}")
# A boundary tensor between shared and expert MXQs carries exactly reciprocal quantization scales.
_EXPERT_IN_SCALE_TOLERANCE = 1e-3
# Prefill chunk sizes whose shared-MXQ buffers stay allocated; older sizes are released (least recently used).
_MAX_CACHED_CHUNK_SIZES = 2
# Token axis of a shared MXQ output, matching ``MobilintModelMixin``'s ``_MXQ_TOKEN_AXIS_INDEX``.
_MXQ_TOKEN_AXIS_INDEX = -2


MXQ_PACK_FORMAT = "mobilint-moe-mxqpack"
MXQ_PACK_VERSION = 1


@dataclass(frozen=True)
class MobilintMixtureOfExpertsPackIndex:
    """Index of a packed MoE release: where each shared/expert MXQ lives inside its layer pack.

    ``sizes[layer]`` lists the exact byte sizes of the layer's members in pack order: the shared MXQ, then experts
    ``0 .. num_experts - 1``. Member ``i`` starts at the previous member's end rounded up to ``alignment``; the
    padding is zero bytes and is not part of the member. ``pack`` is a ``str.format`` template (``layer``) relative
    to the index file's directory.
    """

    pack: str
    alignment: int
    sizes: tuple[tuple[int, ...], ...]

    @classmethod
    def load(cls, path: str) -> "MobilintMixtureOfExpertsPackIndex":
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        if data.get("format") != MXQ_PACK_FORMAT or data.get("version") != MXQ_PACK_VERSION:
            raise ValueError(
                f"{path} is not a {MXQ_PACK_FORMAT} v{MXQ_PACK_VERSION} index "
                f"(format={data.get('format')!r}, version={data.get('version')!r})."
            )
        alignment = int(data["alignment"])
        if alignment < 1 or alignment & (alignment - 1):
            raise ValueError(f"{path}: alignment must be a power of two, got {alignment}.")
        return cls(
            pack=str(data["pack"]),
            alignment=alignment,
            sizes=tuple(tuple(int(size) for size in row) for row in data["sizes"]),
        )

    def to_json(self) -> str:
        """Serialize with one line per layer so the index stays readable (``num_layers + 6`` lines)."""
        rows = ",\n".join("    " + json.dumps(list(row), separators=(",", ":")) for row in self.sizes)
        return (
            "{\n"
            f'  "format": "{MXQ_PACK_FORMAT}",\n'
            f'  "version": {MXQ_PACK_VERSION},\n'
            f'  "alignment": {self.alignment},\n'
            f'  "pack": {json.dumps(self.pack)},\n'
            '  "sizes": [\n'
            f"{rows}\n"
            "  ]\n"
            "}\n"
        )

    def aligned(self, size: int) -> int:
        return -(-size // self.alignment) * self.alignment

    def offsets(self, layer_idx: int) -> list[int]:
        """Return the start offset of every member of ``layer_idx`` within its pack."""
        offsets, cursor = [], 0
        for size in self.sizes[layer_idx]:
            offsets.append(cursor)
            cursor += self.aligned(size)
        return offsets

    def pack_size(self, layer_idx: int) -> int:
        """Return the byte size of ``layer_idx``'s pack (the last member is not padded)."""
        sizes = self.sizes[layer_idx]
        return sum(self.aligned(size) for size in sizes[:-1]) + sizes[-1]

    def validate(self, num_layers: int, num_experts: int) -> None:
        if len(self.sizes) != num_layers:
            raise ValueError(f"MoE pack index lists {len(self.sizes)} layers, expected {num_layers}.")
        for layer_idx, row in enumerate(self.sizes):
            if len(row) != 1 + num_experts:
                raise ValueError(
                    f"MoE pack index layer {layer_idx} lists {len(row)} members, expected 1 shared + "
                    f"{num_experts} experts."
                )


MXQ_MATERIALIZE_ENV = "MBLT_MOE_MXQ_MATERIALIZE"
MXQ_TMPDIR_ENV = "MBLT_MOE_MXQ_TMPDIR"
_MATERIALIZE_MODES = ("auto", "memfd", "tempfile")
# sendfile() refuses some source/destination pairs (e.g. older kernels, FUSE or network mounts).
_SENDFILE_UNSUPPORTED = {getattr(errno, name) for name in ("EINVAL", "ENOSYS", "EOPNOTSUPP", "ENOTSUP", "EXDEV") if hasattr(errno, name)}
_COPY_CHUNK = 8 * 1024 * 1024


def _memfd_path_usable() -> bool:
    """Return True when a ``memfd`` can be reopened by path through ``/proc/self/fd``.

    ``qbruntime.Model`` opens MXQs by path, so a ``memfd`` is only usable when ``/proc`` is mounted and readable. Some
    sandboxes and minimal containers hide ``/proc``; probe it once instead of assuming.
    """
    if not (sys.platform.startswith("linux") and hasattr(os, "memfd_create")):
        return False
    try:
        fd = os.memfd_create("mblt-moe-probe", os.MFD_CLOEXEC)
    except OSError:
        return False
    try:
        os.write(fd, b"\x01")
        with open(f"/proc/self/fd/{fd}", "rb") as file:
            return file.read(1) == b"\x01"
    except OSError:
        return False
    finally:
        os.close(fd)


def _default_tempdir() -> str:
    """Pick the directory for materialized members when ``memfd`` is not used.

    ``$MBLT_MOE_MXQ_TMPDIR`` wins. On Linux ``/dev/shm`` (RAM-backed tmpfs) is preferred over ``/tmp``, which is often a
    disk. Otherwise ``tempfile.gettempdir()`` applies, which honors ``TMPDIR`` / ``TEMP``.
    """
    configured = os.environ.get(MXQ_TMPDIR_ENV)
    if configured:
        return configured
    if sys.platform.startswith("linux") and os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK | os.X_OK):
        return "/dev/shm"
    return tempfile.gettempdir()


class _MxqPackReader:
    """Materialize packed MXQ members as short-lived files for ``qbruntime.Model``.

    ``qbruntime.Model`` accepts only a path and re-reads the file during ``launch()``, so a member must exist as a
    file from ``Model()`` until ``launch()`` returns, and no longer. Two materializations exist:

    * ``memfd`` (Linux, when ``/proc/self/fd`` is usable): an anonymous in-memory file exposed as
      ``/proc/self/fd/N``. Nothing touches the disk.
    * ``tempfile`` (everywhere else, and the Linux fallback): a file in :func:`_default_tempdir` created with
      ``O_SHORT_LIVED`` (``FILE_ATTRIBUTE_TEMPORARY`` on Windows) and deleted right after ``launch()``.

    The copy is ``os.sendfile`` (in-kernel, no user-space buffer) where available, falling back to positional reads,
    and a read-only ``mmap`` view of the pack on Windows. On Linux the consumed pack range is then dropped from the
    page cache with ``posix_fadvise(DONTNEED)`` so loading a 15-30 GB release does not crowd out application memory.
    ``$MBLT_MOE_MXQ_MATERIALIZE`` (``auto`` / ``memfd`` / ``tempfile``) overrides the choice.

    Releasing never raises: a temp file that cannot be deleted yet (Windows keeps files open by another handle) is
    retried in :meth:`close`, and anything still left is logged.

    Packs are opened once each and shared by the per-device loader threads; every read is positional, so no seek
    state is shared.
    """

    def __init__(
        self, index_path: str, index: MobilintMixtureOfExpertsPackIndex, *, mode: Optional[str] = None
    ) -> None:
        self.index = index
        self._pack_dir = os.path.dirname(index_path)
        self._offsets = [index.offsets(layer_idx) for layer_idx in range(len(index.sizes))]
        self._lock = threading.Lock()
        self._fds: dict[int, int] = {}
        self._maps: dict[int, mmap.mmap] = {}
        self._pending_removal: list[str] = []
        requested = (mode or os.environ.get(MXQ_MATERIALIZE_ENV) or "auto").strip().lower()
        if requested not in _MATERIALIZE_MODES:
            raise ValueError(f"{MXQ_MATERIALIZE_ENV} must be one of {_MATERIALIZE_MODES}, got {requested!r}.")
        if requested == "memfd" and not _memfd_path_usable():
            raise RuntimeError("memfd materialization requested, but /proc/self/fd paths are not usable here.")
        self.mode = "memfd" if requested == "memfd" or (requested == "auto" and _memfd_path_usable()) else "tempfile"
        self.tempdir = _default_tempdir() if self.mode == "tempfile" else None
        self._sendfile_ok = hasattr(os, "sendfile")
        logger.debug("MoE pack reader: mode=%s tempdir=%s", self.mode, self.tempdir)

    def pack_path(self, layer_idx: int) -> str:
        return os.path.join(self._pack_dir, self.index.pack.format(layer=layer_idx))

    def _pack_fd(self, layer_idx: int) -> int:
        with self._lock:
            fd = self._fds.get(layer_idx)
            if fd is None:
                fd = os.open(self.pack_path(layer_idx), os.O_RDONLY | getattr(os, "O_BINARY", 0))
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
                self._fds[layer_idx] = fd
            return fd

    def _pack_map(self, layer_idx: int) -> mmap.mmap:
        fd = self._pack_fd(layer_idx)
        with self._lock:
            pack_map = self._maps.get(layer_idx)
            if pack_map is None:
                pack_map = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                self._maps[layer_idx] = pack_map
            return pack_map

    def materialize(self, layer_idx: int, member_idx: int, name: str) -> tuple[str, Callable[[], None]]:
        """Return ``(path, release)`` for member ``member_idx`` (0 = shared, 1 + e = expert e) of a layer.

        ``release()`` never raises.
        """
        offset = self._offsets[layer_idx][member_idx]
        size = self.index.sizes[layer_idx][member_idx]
        if self.mode == "memfd":
            return self._materialize_memfd(layer_idx, offset, size, name)
        return self._materialize_tempfile(layer_idx, offset, size, name)

    def _copy_member(self, layer_idx: int, dst: int, offset: int, size: int, name: str) -> None:
        """Copy ``size`` bytes at ``offset`` of the layer pack into the writable descriptor ``dst``."""
        src = self._pack_fd(layer_idx)
        copied = 0
        if self._sendfile_ok:
            try:
                while copied < size:
                    sent = os.sendfile(dst, src, offset + copied, size - copied)
                    if sent == 0:
                        break
                    copied += sent
            except OSError as exc:
                if exc.errno not in _SENDFILE_UNSUPPORTED:
                    raise
                self._sendfile_ok = False
                logger.debug("sendfile unsupported (%s); falling back to positional reads.", exc)
        if copied < size and hasattr(os, "pread"):
            while copied < size:
                chunk = os.pread(src, min(_COPY_CHUNK, size - copied), offset + copied)
                if not chunk:
                    break
                view = memoryview(chunk)
                while view:
                    view = view[os.write(dst, view) :]
                copied += len(chunk)
        elif copied < size:
            pack_map = self._pack_map(layer_idx)
            end = min(offset + size, len(pack_map))
            with memoryview(pack_map) as whole, whole[offset + copied : end] as member:
                view = member
                while view:
                    written = os.write(dst, view)
                    copied += written
                    view = view[written:]
        if copied != size:
            raise EOFError(f"{self.pack_path(layer_idx)} ends before member {name!r} ({copied}/{size} bytes).")
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(src, offset, self.index.aligned(size), os.POSIX_FADV_DONTNEED)

    def _materialize_memfd(self, layer_idx: int, offset: int, size: int, name: str) -> tuple[str, Callable[[], None]]:
        dst = os.memfd_create(name, os.MFD_CLOEXEC)
        try:
            self._copy_member(layer_idx, dst, offset, size, name)
        except BaseException:
            os.close(dst)
            raise

        def _release() -> None:
            try:
                os.close(dst)
            except OSError as exc:
                logger.warning("Could not close materialized MXQ %s: %s", name, exc)

        return f"/proc/self/fd/{dst}", _release

    def _materialize_tempfile(self, layer_idx: int, offset: int, size: int, name: str) -> tuple[str, Callable[[], None]]:
        assert self.tempdir is not None
        path = os.path.join(self.tempdir, f"mblt-moe-{os.getpid()}-{uuid.uuid4().hex}-{name}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_SHORT_LIVED", 0)
        fd = os.open(path, flags, 0o600)
        try:
            self._copy_member(layer_idx, fd, offset, size, name)
        except BaseException:
            os.close(fd)
            self._remove(path)
            raise
        os.close(fd)
        return path, lambda: self._remove(path)

    def _remove(self, path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError:
            # Windows refuses to delete a file another handle still holds; retry once everything is launched.
            with self._lock:
                self._pending_removal.append(path)

    def close(self) -> None:
        """Close every pack and delete any materialized file that could not be removed earlier. Never raises."""
        with self._lock:
            for pack_map in self._maps.values():
                try:
                    pack_map.close()
                except (BufferError, OSError) as exc:
                    logger.warning("Could not unmap an MXQ pack: %s", exc)
            for fd in self._fds.values():
                try:
                    os.close(fd)
                except OSError as exc:
                    logger.warning("Could not close an MXQ pack: %s", exc)
            self._maps = {}
            self._fds = {}
            pending, self._pending_removal = self._pending_removal, []
        for path in pending:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("Could not delete materialized MXQ %s: %s", path, exc)


def _is_bad_alloc(exc: BaseException) -> bool:
    """Return True when ``exc`` reports a qbruntime device-memory ``BadAlloc``."""
    return "badalloc" in str(exc).lower().replace(" ", "")


def _scale_value(scale: Any) -> Any:
    """Return a qbruntime scale as a float (uniform) or a per-channel ``np.ndarray``."""
    if scale.is_uniform:
        return np.float64(scale.scale)
    return np.asarray(list(scale.scale_list), dtype=np.float64)


@dataclass
class MobilintMixtureOfExpertsLayer:
    """Models, placement, and seqlen=1 I/O buffers of one MoE decoder layer.

    NPU-resident ``NPUData`` is interpreted with the quantization scale of the model it was acquired from, and the
    scales differ per layer (and per expert for expert outputs), so every NPU buffer is acquired from the model
    that consumes it. Host buffers are float32 and NPU-to-host readback dequantizes with the source scale, so host
    buffers may be shared.
    """

    shared: Model
    experts: list[Model]
    shared_dev: int
    residual_idx: int
    expert_in_idx: int
    gate_idx: int
    last_only: bool
    host_in: Any = None
    npu_in: Any = None
    npu_out: list[Any] = field(default_factory=list)
    host_residual: Any = None
    host_gate: Any = None
    host_expert_in: Any = None
    # Copies of ``expert_in`` for expert devices other than ``shared_dev``. All experts of a layer share the input
    # scale, so a buffer acquired from any expert on a device serves every expert on that device.
    npu_expert_in_by_dev: dict[int, Any] = field(default_factory=dict)
    host_expert_in_by_dev: dict[int, Any] = field(default_factory=dict)


@dataclass
class _ChunkBuffers:
    """Per-layer shared-MXQ I/O buffers for one prefill chunk size."""

    host_in: list[Any]
    npu_in: list[Any]
    npu_out: list[list[Any]]
    host_residual: list[Any]
    host_gate: list[Any]
    host_expert_in: list[Any]


class MobilintMixtureOfExpertsNPUBackend:
    """Load and run a MoE release across the devices described by three role backends.

    Placement is round-robin over each role's device list: ``shared[layer]`` runs on
    ``shared_devices[layer % len(shared_devices)]``, ``expert[layer][expert]`` on
    ``expert_devices[expert % len(expert_devices)]``, and the ``lm_head`` on its single device. Every model on a
    device uses the role's cores on that device as a ``single`` core mode ``ModelConfig``.

    :meth:`create` resolves the artifact and validates the topology without touching the NPU. :meth:`launch`
    loads and launches every model, then allocates the I/O buffers. On a device-memory ``BadAlloc`` every model
    loaded so far is disposed and the failure is re-raised as :class:`MobilintBackendAllocError`.
    """

    max_batch_size = 1
    k_per_model = 1

    def __init__(
        self,
        *,
        shared_backend: MobilintNPUBackend,
        expert_backend: MobilintNPUBackend,
        lm_head_backend: MobilintNPUBackend,
        num_layers: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        vocab_size: int,
        num_expert_workers: int,
        shared_output_indices: Sequence[Optional[tuple[int, int, int]]],
        name_or_path: str = "",
        mxq_pack_index: Optional[str] = None,
    ) -> None:
        self.role_backends: dict[str, MobilintNPUBackend] = {
            "shared": shared_backend,
            "expert": expert_backend,
            "lm_head": lm_head_backend,
        }
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.num_expert_workers = int(num_expert_workers)
        if len(shared_output_indices) != self.num_layers:
            raise ValueError(
                f"shared_output_indices must have one entry per layer ({self.num_layers}), "
                f"got {len(shared_output_indices)}"
            )
        self._configured_output_indices = list(shared_output_indices)
        self.name_or_path = name_or_path
        self.mxq_pack_index = mxq_pack_index
        self.pack_index: Optional[MobilintMixtureOfExpertsPackIndex] = None
        self.revision: Optional[str] = None
        self._commit_hash: Optional[str] = None

        self.model_dir: Optional[str] = None
        self.accs: dict[int, Accelerator] = {}
        self.layers: list[MobilintMixtureOfExpertsLayer] = []
        self.lm_head: Optional[Model] = None
        self._created = False
        self._expert_pool: Optional[ThreadPoolExecutor] = None
        self._chunk_buffers: OrderedDict[int, _ChunkBuffers] = OrderedDict()
        # The cache whose sequence currently lives in the shared MXQs' KV state (``None`` after a cache-less forward).
        self.kv_owner: Any = None
        self._expert_out_pool: dict[int, list[Any]] = {}
        self._expert_out_pool_scale: dict[int, Any] = {}
        self._expert_corr: dict[tuple[int, int], Any] = {}
        self._host_expert_out: Any = None
        self._host_lm_head_in: Any = None
        self._host_lm_head_out: Any = None

    # ---- Topology -----------------------------------------------------------------------------------------------

    @property
    def target_device(self) -> str:
        return self.role_backends["shared"].target_device

    @property
    def core_mode(self) -> str:
        return self.role_backends["shared"].core_mode

    @property
    def mxq_model(self) -> None:
        """MoE releases have no single MXQ; kept so single-model probes resolve to ``None``."""
        return None

    def _role_cores_by_device(self, role: str) -> dict[int, list[Any]]:
        cores = self.role_backends[role].target_cores_by_device
        if not cores:
            raise ValueError(f"{role}_target_cores must list at least one 'd:c:k' core.")
        return cores

    @property
    def shared_devices(self) -> list[int]:
        return list(self._role_cores_by_device("shared"))

    @property
    def expert_devices(self) -> list[int]:
        return list(self._role_cores_by_device("expert"))

    @property
    def lm_head_device(self) -> int:
        return self.lm_head_devices[0]

    @property
    def lm_head_devices(self) -> list[int]:
        return list(self._role_cores_by_device("lm_head"))

    def shared_device_of(self, layer_idx: int) -> int:
        devices = self.shared_devices
        return devices[layer_idx % len(devices)]

    def expert_device_of(self, expert_idx: int) -> int:
        devices = self.expert_devices
        return devices[expert_idx % len(devices)]

    def _validate_topology(self) -> None:
        boards = {role: backend.target_device for role, backend in self.role_backends.items()}
        if len(set(boards.values())) != 1:
            raise ValueError(f"MoE roles must target the same board, got {boards}.")
        for role, backend in self.role_backends.items():
            if backend.core_mode != "single":
                raise ValueError(
                    f"MoE releases support only {role}_core_mode='single', got {backend.core_mode!r}."
                )
        if len(self.lm_head_devices) != 1:
            raise ValueError(f"lm_head_target_cores must cover exactly one device, got {self.lm_head_devices}.")
        if len(self.expert_devices) > self.num_experts:
            raise ValueError(
                f"expert_target_cores cover {len(self.expert_devices)} devices but the release has only "
                f"{self.num_experts} experts per layer."
            )

    def _make_model_config(self, role: str, dev: int) -> ModelConfig:
        mc = ModelConfig()
        mc.set_single_core_mode(None, self.role_backends[role].filter_cores_for(dev))
        mc.set_activation_slots(_ROLE_ACTIVATION_SLOTS[role])
        return mc

    # ---- Artifact resolution ------------------------------------------------------------------------------------

    def _role_template(self, role: str) -> str:
        template = self.role_backends[role].mxq_path
        if not template:
            raise ValueError(f"{role}_mxq_path must be set for a MoE release.")
        return template

    def shared_mxq_path(self, layer_idx: int) -> str:
        return self._role_template("shared").format(layer=layer_idx)

    def expert_mxq_path(self, layer_idx: int, expert_idx: int) -> str:
        return self._role_template("expert").format(layer=layer_idx, expert=expert_idx)

    def lm_head_mxq_path(self) -> str:
        return self._role_template("lm_head")

    def _resolve_artifact_dir(self) -> str:
        """Return the directory holding the release MXQs, downloading them from the Hub once if needed.

        ``MobilintNPUBackend.check_model_path`` resolves one file per call, which is impractical for thousands of
        MXQs, so the whole MXQ set is fetched with a single ``snapshot_download``. The local cache is tried first so
        ``HF_HUB_OFFLINE=1`` works without listing the repository tree.
        """
        if os.path.isdir(self.name_or_path):
            return self.name_or_path
        if not self.name_or_path:
            raise ValueError("MobilintMixtureOfExpertsNPUBackend.name_or_path is empty.")

        if self.mxq_pack_index:
            pack_dir = os.path.dirname(self.mxq_pack_index)
            pack_glob = f"{pack_dir}/*.mxqpack" if pack_dir else "*.mxqpack"
            allow_patterns = sorted({self.lm_head_mxq_path(), self.mxq_pack_index, pack_glob})
        else:
            allow_patterns = sorted(
                {_TEMPLATE_FIELD_RE.sub("*", self._role_template(role)) for role in self.role_backends}
            )
        revision = self.revision or self._commit_hash
        try:
            snapshot = snapshot_download(
                repo_id=self.name_or_path,
                revision=revision,
                allow_patterns=allow_patterns,
                local_files_only=True,
            )
            if self._missing_artifacts(snapshot) == []:
                return snapshot
        except LocalEntryNotFoundError:
            pass
        return snapshot_download(repo_id=self.name_or_path, revision=revision, allow_patterns=allow_patterns)

    def _missing_artifacts(self, root: str) -> list[str]:
        """Return the release files missing under ``root`` (a pack of the wrong size counts as missing).

        A download that stopped halfway leaves some files behind; checking every expected file keeps the local-cache
        shortcut from treating such a snapshot as complete, so the next load resumes the download instead.
        """
        missing = [path for path in (self.lm_head_mxq_path(),) if not os.path.isfile(os.path.join(root, path))]
        if self.mxq_pack_index:
            index_path = os.path.join(root, self.mxq_pack_index)
            if not os.path.isfile(index_path):
                return missing + [self.mxq_pack_index]
            try:
                index = MobilintMixtureOfExpertsPackIndex.load(index_path)
            except (OSError, ValueError, KeyError):
                return missing + [self.mxq_pack_index]
            pack_dir = os.path.dirname(index_path)
            for layer_idx in range(len(index.sizes)):
                pack_path = os.path.join(pack_dir, index.pack.format(layer=layer_idx))
                if not os.path.isfile(pack_path) or os.path.getsize(pack_path) != index.pack_size(layer_idx):
                    missing.append(os.path.relpath(pack_path, root))
            return missing
        for layer_idx in range(self.num_layers):
            paths = [self.shared_mxq_path(layer_idx)] + [
                self.expert_mxq_path(layer_idx, expert_idx) for expert_idx in range(self.num_experts)
            ]
            missing.extend(path for path in paths if not os.path.isfile(os.path.join(root, path)))
        return missing

    # ---- Lifecycle ----------------------------------------------------------------------------------------------

    def create(self) -> None:
        """Validate the topology and resolve the MXQ artifact. Does not touch the NPU."""
        self._validate_topology()
        self.model_dir = self._resolve_artifact_dir()
        if self.mxq_pack_index:
            self._load_pack_index()
        missing = self._missing_artifacts(self.model_dir)
        if missing:
            shown = ", ".join(missing[:5]) + (f" and {len(missing) - 5} more" if len(missing) > 5 else "")
            raise FileNotFoundError(f"MoE release under {self.model_dir!r} is missing {shown}.")
        self._created = True

    def _load_pack_index(self) -> None:
        """Load the pack index and check every layer pack exists with exactly the size the index describes."""
        assert self.model_dir is not None and self.mxq_pack_index
        index_path = os.path.join(self.model_dir, self.mxq_pack_index)
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"MoE release under {self.model_dir!r} is missing {self.mxq_pack_index!r}.")
        index = MobilintMixtureOfExpertsPackIndex.load(index_path)
        index.validate(self.num_layers, self.num_experts)
        pack_dir = os.path.dirname(index_path)
        for layer_idx in range(self.num_layers):
            pack_path = os.path.join(pack_dir, index.pack.format(layer=layer_idx))
            if not os.path.isfile(pack_path):
                raise FileNotFoundError(f"MoE release is missing pack {pack_path!r}.")
            actual, expected = os.path.getsize(pack_path), index.pack_size(layer_idx)
            if actual != expected:
                raise ValueError(f"{pack_path} is {actual} bytes but its index describes {expected} bytes.")
        self.pack_index = index

    def launch(self) -> None:
        """Load and launch every MXQ on its device, then allocate the I/O buffers."""
        if not self._created or self.model_dir is None:
            raise RuntimeError("MobilintMixtureOfExpertsNPUBackend.launch() requires create() to succeed first.")
        try:
            qbruntime.set_log_level(qbruntime.LogLevel.Warn)
        except AttributeError:
            pass

        model_dir = self.model_dir
        packed = self.pack_index is not None
        reader = (
            _MxqPackReader(os.path.join(model_dir, self.mxq_pack_index), self.pack_index)  # type: ignore[arg-type]
            if packed
            else None
        )
        # ``path`` is a file under ``model_dir``, or ``None`` for a member materialized from its layer pack.
        plan: list[tuple[str, int, int, int, Optional[str]]] = []
        for layer_idx in range(self.num_layers):
            plan.append(
                (
                    "shared",
                    layer_idx,
                    -1,
                    self.shared_device_of(layer_idx),
                    None if packed else self.shared_mxq_path(layer_idx),
                )
            )
            for expert_idx in range(self.num_experts):
                plan.append(
                    (
                        "expert",
                        layer_idx,
                        expert_idx,
                        self.expert_device_of(expert_idx),
                        None if packed else self.expert_mxq_path(layer_idx, expert_idx),
                    )
                )
        plan.append(("lm_head", -1, -1, self.lm_head_device, self.lm_head_mxq_path()))

        devices = sorted({item[3] for item in plan})
        self.accs = {dev: Accelerator(self.target_device, dev) for dev in devices}
        configs = {
            (role, dev): self._make_model_config(role, dev)
            for role in self.role_backends
            for dev in devices
            if dev in self._role_cores_by_device(role)
        }

        shared: list[Optional[Model]] = [None] * self.num_layers
        experts: list[list[Optional[Model]]] = [[None] * self.num_experts for _ in range(self.num_layers)]
        lm_head: list[Optional[Model]] = [None]
        loaded: list[Model] = []
        launched = [0]
        loaded_lock = threading.Lock()
        stop = threading.Event()
        failures: list[tuple[str, int, int, BaseException]] = []

        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover - tqdm ships with transformers
            tqdm = None
        progress = tqdm(total=len(plan), desc="[Mobilint] MoE load", unit="mxq") if tqdm is not None else None

        def _load_queue(items: list[tuple[int, tuple[str, int, int, int, Optional[str]]]]) -> None:
            for slot_idx, (role, layer_idx, expert_idx, dev, path) in items:
                if stop.is_set():
                    return
                phase = "create"
                release: Optional[Callable[[], None]] = None
                try:
                    if path is None:
                        assert reader is not None
                        if role == "shared":
                            member_idx, name = 0, f"block_{layer_idx}_shared.mxq"
                        else:
                            member_idx, name = 1 + expert_idx, f"block_{layer_idx}_expert_{expert_idx}.mxq"
                        model_path, release = reader.materialize(layer_idx, member_idx, name)
                    else:
                        model_path = os.path.join(model_dir, path)
                    model = Model(model_path, configs[(role, dev)])
                    with loaded_lock:
                        loaded.append(model)
                    phase = "launch"
                    # qbruntime re-reads the MXQ during launch(); a materialized member is released right after.
                    model.launch(self.accs[dev])
                    with loaded_lock:
                        launched[0] += 1
                except BaseException as exc:  # noqa: BLE001 - recorded and re-raised after every thread stops.
                    failures.append((phase, slot_idx, dev, exc))
                    stop.set()
                    return
                finally:
                    # ``release`` never raises, so it cannot mask the failure recorded above.
                    if release is not None:
                        release()
                if role == "shared":
                    shared[layer_idx] = model
                elif role == "expert":
                    experts[layer_idx][expert_idx] = model
                else:
                    lm_head[0] = model
                if progress is not None:
                    progress.update(1)

        queues: dict[int, list[tuple[int, tuple[str, int, int, int, Optional[str]]]]] = {
            dev: [] for dev in devices
        }
        for slot_idx, item in enumerate(plan):
            queues[item[3]].append((slot_idx, item))
        try:
            with ThreadPoolExecutor(max_workers=len(devices)) as executor:
                for future in as_completed([executor.submit(_load_queue, queue) for queue in queues.values()]):
                    future.result()
        finally:
            if progress is not None:
                progress.close()
            if reader is not None:
                reader.close()

        if failures:
            phase, slot_idx, dev, exc = failures[0]
            self._rollback(loaded)
            self._raise_load_failure(exc, phase=phase, slot=slot_idx, dev=dev, launched=launched[0], total=len(plan))

        # Layer setup and buffer allocation can still fail (an unresolvable output order, a buffer BadAlloc); the
        # loaded models are not reachable from ``self`` until this succeeds, so roll back here as well.
        try:
            self.lm_head = lm_head[0]
            for layer_idx in range(self.num_layers):
                self.layers.append(self._build_layer(layer_idx, shared[layer_idx], experts[layer_idx]))  # type: ignore[arg-type]
            self._allocate_shared_buffers()
            self._expert_pool = ThreadPoolExecutor(max_workers=self.num_expert_workers)
        except BaseException as exc:
            self._rollback(loaded)
            self._raise_load_failure(exc, phase="allocate", slot=len(plan), dev=-1, launched=launched[0], total=len(plan))

    def _rollback(self, loaded: Sequence[Model]) -> None:
        """Release every buffer and model of a failed :meth:`launch`."""
        self._release_buffers()
        gc.collect()
        self._dispose_models(loaded)
        gc.collect()
        self.accs = {}

    def _raise_load_failure(
        self, exc: BaseException, *, phase: str, slot: int, dev: int, launched: int, total: int
    ) -> None:
        if isinstance(exc, QbRuntimeError) and _is_bad_alloc(exc):
            raise MobilintBackendAllocError(
                phase=phase,
                slot=slot,
                dev=int(dev),
                succeeded_so_far=launched,
                n_total=total,
                max_batch_size=self.max_batch_size,
                k_per_model=self.k_per_model,
                original=exc,
            ) from exc
        raise exc

    def _release_buffers(self) -> None:
        """Drop every NPU buffer reference and the model handles that own them."""
        if self._expert_pool is not None:
            # Running expert inferences must finish before their models are disposed.
            self._expert_pool.shutdown(wait=True, cancel_futures=True)
            self._expert_pool = None
        self.layers = []
        self.lm_head = None
        self._chunk_buffers = OrderedDict()
        self._expert_out_pool = {}
        self._expert_out_pool_scale = {}
        self._expert_corr = {}
        self._host_expert_out = None
        self._host_lm_head_in = None
        self._host_lm_head_out = None
        self.kv_owner = None

    def dispose(self) -> None:
        """Release every model and accelerator handle held by this backend. Safe to call multiple times."""
        models: list[Model] = []
        for layer in self.layers:
            models.append(layer.shared)
            models.extend(layer.experts)
        if self.lm_head is not None:
            models.append(self.lm_head)
        # NPU-resident buffers must go before their models and accelerators, or qbruntime warns that the
        # accelerator was destroyed with live NPUData.
        self._release_buffers()
        gc.collect()
        self._dispose_models(models)
        del models
        gc.collect()
        self.accs = {}

    @staticmethod
    def _dispose_models(models: Sequence[Model]) -> None:
        for model in models:
            try:
                model.dispose()
            except Exception as exc:  # noqa: BLE001 - the release path must not raise.
                logger.warning("MoE model dispose failed: %s", exc)

    # ---- Layer setup --------------------------------------------------------------------------------------------

    def _detect_expert_in_index(self, shared: Model, expert: Model, gate_idx: int) -> Optional[int]:
        """Return the shared output whose scale is the reciprocal of the expert input scale, if unambiguous."""
        output_scales = shared.get_output_scale()
        expert_in_scale = expert.get_input_scale()[0]
        errors: dict[int, float] = {}
        for idx, scale in enumerate(output_scales):
            if idx == gate_idx:
                continue
            if scale.is_uniform and expert_in_scale.is_uniform:
                errors[idx] = abs(float(scale.scale) * float(expert_in_scale.scale) - 1.0)
        if not errors:
            return None
        best = min(errors, key=errors.__getitem__)
        return best if errors[best] <= _EXPERT_IN_SCALE_TOLERANCE else None

    def _resolve_output_indices(self, layer_idx: int, shared: Model, expert: Model) -> tuple[int, int, int]:
        """Resolve ``(residual, expert_in, gate)`` for one layer.

        ``config.moe_shared_output_order`` is authoritative; the quantization scales only cross-check it. Without a
        configured order, the gate is the output with ``num_experts`` channels and ``expert_in`` the output whose
        scale is the reciprocal of the expert input scale.
        """
        output_shapes = shared.get_model_output_shape()
        if len(output_shapes) != 3:
            raise ValueError(f"shared[{layer_idx}] must have 3 outputs, got {len(output_shapes)}.")
        configured = self._configured_output_indices[layer_idx]
        if configured is not None:
            residual_idx, expert_in_idx, gate_idx = configured
        else:
            gate_candidates = [idx for idx, shape in enumerate(output_shapes) if int(shape[-1]) == self.num_experts]
            if len(gate_candidates) != 1:
                raise ValueError(
                    f"Cannot identify the router gate output of shared[{layer_idx}] from shapes {output_shapes}; "
                    "set moe_shared_output_order in config.json."
                )
            gate_idx = gate_candidates[0]
            detected = self._detect_expert_in_index(shared, expert, gate_idx)
            if detected is None:
                raise ValueError(
                    f"Cannot identify the expert input output of shared[{layer_idx}] from its quantization "
                    "scales; set moe_shared_output_order in config.json."
                )
            expert_in_idx = detected
            residual_idx = ({0, 1, 2} - {gate_idx, expert_in_idx}).pop()

        if int(output_shapes[gate_idx][-1]) != self.num_experts:
            raise ValueError(
                f"shared[{layer_idx}] output {gate_idx} has {int(output_shapes[gate_idx][-1])} channels but the "
                f"router gate must have num_experts={self.num_experts}; check moe_shared_output_order."
            )
        if configured is not None:
            detected = self._detect_expert_in_index(shared, expert, gate_idx)
            if detected is not None and detected != expert_in_idx:
                logger.warning(
                    "shared[%d]: moe_shared_output_order places expert_in at output %d but the quantization "
                    "scales point to output %d.",
                    layer_idx,
                    expert_in_idx,
                    detected,
                )
        return residual_idx, expert_in_idx, gate_idx

    def _build_layer(self, layer_idx: int, shared: Model, experts: list[Model]) -> MobilintMixtureOfExpertsLayer:
        residual_idx, expert_in_idx, gate_idx = self._resolve_output_indices(layer_idx, shared, experts[0])
        # The final shared MXQ may be compiled to emit only the last token (only it feeds the lm_head).
        last_only = int(shared.get_model_output_shape()[0][_MXQ_TOKEN_AXIS_INDEX]) == 1
        return MobilintMixtureOfExpertsLayer(
            shared=shared,
            experts=experts,
            shared_dev=self.shared_device_of(layer_idx),
            residual_idx=residual_idx,
            expert_in_idx=expert_in_idx,
            gate_idx=gate_idx,
            last_only=last_only,
        )

    def _output_channels(self, layer: MobilintMixtureOfExpertsLayer, output_idx: int) -> int:
        return self.num_experts if output_idx == layer.gate_idx else self.hidden_size

    def _allocate_shared_buffers(self) -> None:
        hidden = self.hidden_size
        expert_devices = self.expert_devices
        for layer in self.layers:
            shared = layer.shared
            layer.host_in = shared.acquire_input_npu_data([1, 1, hidden], 0, False)
            layer.npu_in = shared.acquire_input_npu_data([1, 1, hidden], 0, True)
            layer.npu_out = [
                shared.acquire_output_npu_data([1, 1, self._output_channels(layer, idx)], idx, True)
                for idx in range(3)
            ]
            layer.host_residual = shared.acquire_output_npu_data([1, 1, hidden], layer.residual_idx, False)
            layer.host_gate = shared.acquire_output_npu_data([1, 1, self.num_experts], layer.gate_idx, False)
            layer.host_expert_in = shared.acquire_output_npu_data([1, 1, hidden], layer.expert_in_idx, False)
            # Expert ``i`` lives on ``expert_devices[i]`` for ``i < len(expert_devices)``, so it represents that
            # device when acquiring the per-device ``expert_in`` copies.
            for rep_idx, dev in enumerate(expert_devices):
                if dev == layer.shared_dev:
                    continue
                rep = layer.experts[rep_idx]
                layer.npu_expert_in_by_dev[dev] = rep.acquire_input_npu_data([1, 1, hidden], 0, True)
                layer.host_expert_in_by_dev[dev] = rep.acquire_input_npu_data([1, 1, hidden], 0, False)

        # Expert outputs have a distinct scale per expert, but one NPU buffer per (layer, expert) would exhaust NPU
        # DRAM. Keep ``num_experts_per_tok`` pool buffers per device instead and rescale the host readback by
        # ``expert_out_scale / pool_scale`` (the expert writes raw integers; only their interpretation differs).
        first_layer = self.layers[0]
        for rep_idx, dev in enumerate(expert_devices):
            rep = first_layer.experts[rep_idx]
            self._expert_out_pool[dev] = [
                rep.acquire_output_npu_data([1, 1, hidden], 0, True) for _ in range(self.num_experts_per_tok)
            ]
            self._expert_out_pool_scale[dev] = _scale_value(rep.get_output_scale()[0])
        self._host_expert_out = first_layer.experts[0].acquire_output_npu_data([1, 1, hidden], 0, False)

        assert self.lm_head is not None
        self._host_lm_head_in = self.lm_head.acquire_input_npu_data([1, 1, hidden], 0, False)
        self._host_lm_head_out = self.lm_head.acquire_output_npu_data([1, 1, self.vocab_size], 0, False)

    def _get_chunk_buffers(self, chunk_size: int) -> _ChunkBuffers:
        """Return the shared-MXQ I/O buffers for ``chunk_size``, acquiring them on first use.

        The shared MXQs are compiled with a dynamic sequence axis, so any chunk size works. Each size costs one input
        and three output NPU buffers per layer, so only the ``_MAX_CACHED_CHUNK_SIZES`` most recently used sizes stay
        allocated; a per-call ``npu_prefill_chunk_size`` sweep therefore cannot grow device memory without bound.
        """
        buffers = self._chunk_buffers.get(chunk_size)
        if buffers is not None:
            self._chunk_buffers.move_to_end(chunk_size)
            return buffers
        while len(self._chunk_buffers) >= _MAX_CACHED_CHUNK_SIZES:
            # Dropping the last reference frees the NPUData; collect before allocating the replacement.
            self._chunk_buffers.popitem(last=False)
            gc.collect()
        hidden = self.hidden_size
        buffers = _ChunkBuffers([], [], [], [], [], [])
        for layer in self.layers:
            shared = layer.shared
            out_len = 1 if layer.last_only else chunk_size
            buffers.host_in.append(shared.acquire_input_npu_data([1, chunk_size, hidden], 0, False))
            buffers.npu_in.append(shared.acquire_input_npu_data([1, chunk_size, hidden], 0, True))
            buffers.npu_out.append(
                [
                    shared.acquire_output_npu_data([1, out_len, self._output_channels(layer, idx)], idx, True)
                    for idx in range(3)
                ]
            )
            buffers.host_residual.append(
                shared.acquire_output_npu_data([1, out_len, hidden], layer.residual_idx, False)
            )
            buffers.host_gate.append(
                shared.acquire_output_npu_data([1, out_len, self.num_experts], layer.gate_idx, False)
            )
            buffers.host_expert_in.append(
                shared.acquire_output_npu_data([1, out_len, hidden], layer.expert_in_idx, False)
            )
        self._chunk_buffers[chunk_size] = buffers
        return buffers

    def _expert_correction(self, layer_idx: int, expert_idx: int) -> Any:
        key = (layer_idx, expert_idx)
        corr = self._expert_corr.get(key)
        if corr is None:
            expert = self.layers[layer_idx].experts[expert_idx]
            dev = self.expert_device_of(expert_idx)
            corr = _scale_value(expert.get_output_scale()[0]) / self._expert_out_pool_scale[dev]
            self._expert_corr[key] = corr
        return corr

    # ---- Inference primitives -----------------------------------------------------------------------------------

    def shared_forward_token(self, layer_idx: int, hidden_states: np.ndarray, cache_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Run ``shared[layer_idx]`` on one token and return ``(residual, gate)`` as ``[1, 1, C]`` float32.

        ``expert_in`` stays on the NPU; :meth:`experts_forward_token` consumes it directly.
        """
        layer = self.layers[layer_idx]
        layer.host_in[:] = hidden_states
        layer.npu_in.copy_from(layer.host_in)
        layer.shared.infer_npu_data([layer.npu_in], layer.npu_out, cache_size)
        layer.host_gate.copy_from(layer.npu_out[layer.gate_idx])
        layer.host_residual.copy_from(layer.npu_out[layer.residual_idx])
        gate = np.array(layer.host_gate[:], dtype=np.float32)
        residual = np.array(layer.host_residual[:], dtype=np.float32)
        return residual, gate

    def _infer_expert_token(self, layer_idx: int, expert_idx: int, out_buffer: Any) -> int:
        # Workers only call ``infer_npu_data``, which releases the GIL, so experts on different devices overlap.
        # Readback (``copy_from``) holds the GIL and stays on the caller thread.
        layer = self.layers[layer_idx]
        dev = self.expert_device_of(expert_idx)
        if dev == layer.shared_dev:
            # The shared/expert boundary scales are exact reciprocals, so the shared output feeds the expert as is.
            src = layer.npu_out[layer.expert_in_idx]
        else:
            src = layer.npu_expert_in_by_dev[dev]
        layer.experts[expert_idx].infer_npu_data([src], [out_buffer])
        return expert_idx

    def experts_forward_token(self, layer_idx: int, expert_indices: Sequence[int]) -> Iterator[tuple[int, np.ndarray]]:
        """Run the selected experts on the token from the last :meth:`shared_forward_token` call.

        Yields ``(expert_idx, output[hidden])`` in completion order so readback overlaps the remaining inferences.
        ``expert_indices`` must be unique and at most ``num_experts_per_tok`` long.
        """
        assert self._expert_pool is not None, "launch() must run before inference."
        layer = self.layers[layer_idx]
        devices = {self.expert_device_of(expert_idx) for expert_idx in expert_indices}
        remote = devices - {layer.shared_dev}
        if remote:
            layer.host_expert_in.copy_from(layer.npu_out[layer.expert_in_idx])
            for dev in remote:
                layer.host_expert_in_by_dev[dev][:] = layer.host_expert_in[:]
                layer.npu_expert_in_by_dev[dev].copy_from(layer.host_expert_in_by_dev[dev])

        used: dict[int, int] = {}
        futures = {}
        for expert_idx in expert_indices:
            dev = self.expert_device_of(expert_idx)
            out_buffer = self._expert_out_pool[dev][used.get(dev, 0)]
            used[dev] = used.get(dev, 0) + 1
            futures[self._expert_pool.submit(self._infer_expert_token, layer_idx, expert_idx, out_buffer)] = out_buffer
        for future in as_completed(futures):
            expert_idx = future.result()
            self._host_expert_out.copy_from(futures[future])
            output = np.asarray(self._host_expert_out[:], dtype=np.float32)[0, 0, :]
            yield expert_idx, output * self._expert_correction(layer_idx, expert_idx)

    def shared_forward_chunk(
        self, layer_idx: int, hidden_states: np.ndarray, cache_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run ``shared[layer_idx]`` on ``[1, C, hidden]`` and return ``(residual, gate, expert_in)``.

        Outputs are ``[1, C, ...]``, or ``[1, 1, ...]`` on a last-only layer. The input always advances the KV
        cache by ``C`` tokens.
        """
        chunk_size = int(hidden_states.shape[1])
        buffers = self._get_chunk_buffers(chunk_size)
        layer = self.layers[layer_idx]
        buffers.host_in[layer_idx][:] = hidden_states
        buffers.npu_in[layer_idx].copy_from(buffers.host_in[layer_idx])
        npu_out = buffers.npu_out[layer_idx]
        layer.shared.infer_npu_data([buffers.npu_in[layer_idx]], npu_out, cache_size)
        buffers.host_gate[layer_idx].copy_from(npu_out[layer.gate_idx])
        buffers.host_expert_in[layer_idx].copy_from(npu_out[layer.expert_in_idx])
        buffers.host_residual[layer_idx].copy_from(npu_out[layer.residual_idx])
        return (
            np.array(buffers.host_residual[layer_idx][:], dtype=np.float32),
            np.array(buffers.host_gate[layer_idx][:], dtype=np.float32),
            np.array(buffers.host_expert_in[layer_idx][:], dtype=np.float32),
        )

    def _infer_expert_chunk(self, layer_idx: int, expert_idx: int, npu_in: Any, npu_out: Any) -> int:
        self.layers[layer_idx].experts[expert_idx].infer_npu_data([npu_in], [npu_out])
        return expert_idx

    def experts_forward_chunk(
        self, layer_idx: int, expert_inputs: dict[int, np.ndarray]
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Run each expert on its routed tokens ``{expert_idx: [n_tokens, hidden]}``.

        Yields ``(expert_idx, output[n_tokens, hidden])`` in completion order. Buffers are acquired from each expert
        itself, so the readback already carries the expert's own scale.
        """
        assert self._expert_pool is not None, "launch() must run before inference."
        hidden = self.hidden_size
        futures = {}
        for expert_idx, tokens in expert_inputs.items():
            n_tokens = int(tokens.shape[0])
            expert = self.layers[layer_idx].experts[expert_idx]
            host_in = expert.acquire_input_npu_data([1, n_tokens, hidden], 0, False)
            npu_in = expert.acquire_input_npu_data([1, n_tokens, hidden], 0, True)
            npu_out = expert.acquire_output_npu_data([1, n_tokens, hidden], 0, True)
            host_out = expert.acquire_output_npu_data([1, n_tokens, hidden], 0, False)
            host_in[:] = tokens.reshape(1, n_tokens, hidden)
            npu_in.copy_from(host_in)
            future = self._expert_pool.submit(self._infer_expert_chunk, layer_idx, expert_idx, npu_in, npu_out)
            futures[future] = (npu_out, host_out)
        for future in as_completed(futures):
            expert_idx = future.result()
            npu_out, host_out = futures[future]
            host_out.copy_from(npu_out)
            yield expert_idx, np.asarray(host_out[:], dtype=np.float32)[0]

    def lm_head_forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """Run the ``lm_head`` on one ``[hidden]`` hidden state and return ``[vocab]`` logits."""
        assert self.lm_head is not None, "launch() must run before inference."
        self._host_lm_head_in[:] = hidden_states.reshape(1, 1, self.hidden_size)
        self.lm_head.infer_npu_data([self._host_lm_head_in], [self._host_lm_head_out])
        return np.array(self._host_lm_head_out[:], dtype=np.float32)[0, 0]
