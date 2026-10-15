"""Host CPU thread defaults for processes that drive Mobilint NPUs.

PyTorch sizes its intra-op (OpenMP) thread pool from the cores the process can see. That suits one process on a
host, but a serving setup often runs one engine per NPU card on the same host or in the same CPU reservation (cpuset,
container CPU limit). Each engine then starts a full-size pool, and the engines spin on the same cores.

Reference measurement (Qwen3-0.6B, runtime batch 16, three single-ARIES cards, one engine per card, 16-core CPU
reservation shared with the load clients):

- Default torch threads: 194-216 tok/s in total, with the reservation saturated.
- Two threads per engine (``OMP_NUM_THREADS=2``): 557-580 tok/s in total.
- One engine at concurrency 16: the default pool used 2.8-2.9 cores and 45 threads; two threads gave the same
  throughput at 0.9 core.
- The cost of fewer threads is host-side sampling: a 48-row sampling step took 9.5 ms with two threads instead of
  1.6 ms. A decode step on that setup takes about 80-110 ms, almost all of it NPU time, so the host thread count
  matters far less than the oversubscription it avoids.

The defaults follow from that: divide the available cores evenly among the engines on the host, never go below
:data:`DEFAULT_MIN_THREADS` (two threads kept full throughput), and never above :data:`DEFAULT_MAX_THREADS` (the
engine used about three cores even with every core available, so more threads only add spinning; eight keeps headroom
for large sampling batches on one engine). The defaults only ever lower torch's own pool size, and an explicit
``OMP_NUM_THREADS`` or ``MKL_NUM_THREADS`` always wins.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Mapping, Optional

DEFAULT_MIN_THREADS = 2
"""Floor of the per-engine thread count: two threads kept full decode throughput in the reference measurement."""

DEFAULT_MAX_THREADS = 8
"""Cap of the per-engine thread count: an engine used about three cores with every core available."""

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
"""Environment variables that set the thread count explicitly. When any is set, the defaults are not applied."""

_CGROUP_ROOT = Path("/sys/fs/cgroup")


def _cgroup_cpu_limit(cgroup_root: Optional[Path] = None) -> Optional[int]:
    """Return the CPU count allowed by a cgroup CPU quota, or ``None`` when there is no quota.

    Reads the cgroup v2 ``cpu.max`` file, then the cgroup v1 ``cpu.cfs_quota_us`` / ``cpu.cfs_period_us`` pair, at the
    cgroup root the process sees (the container's own cgroup under a cgroup namespace). A fractional quota rounds up.

    Args:
        cgroup_root: Mount point of the cgroup file system. ``None`` means ``/sys/fs/cgroup``.

    Returns:
        The number of CPUs the quota allows, at least 1, or ``None`` when no quota is set or it cannot be read.
    """
    if cgroup_root is None:
        cgroup_root = _CGROUP_ROOT
    try:
        fields = (cgroup_root / "cpu.max").read_text().split()
    except OSError:
        fields = []
    if len(fields) == 2 and fields[0] != "max":
        return _quota_cpus(fields[0], fields[1])
    if fields:
        return None
    try:
        quota = (cgroup_root / "cpu" / "cpu.cfs_quota_us").read_text().strip()
        period = (cgroup_root / "cpu" / "cpu.cfs_period_us").read_text().strip()
    except OSError:
        return None
    return _quota_cpus(quota, period)


def _quota_cpus(quota: str, period: str) -> Optional[int]:
    try:
        quota_us = int(quota)
        period_us = int(period)
    except ValueError:
        return None
    if quota_us <= 0 or period_us <= 0:
        return None
    return max(1, math.ceil(quota_us / period_us))


def available_host_cpus() -> int:
    """Return the number of CPUs this process may run on.

    Uses the scheduler affinity mask (``os.sched_getaffinity``), which reflects ``taskset`` and cpusets, and lowers it
    to a cgroup CPU quota when one is set. Falls back to ``os.cpu_count()`` where affinity is unavailable.

    Returns:
        The available CPU count, at least 1.
    """
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    quota_cpus = _cgroup_cpu_limit()
    if quota_cpus is not None:
        cpus = min(cpus, quota_cpus)
    return max(1, cpus)


def recommended_host_threads(
    n_engines_on_host: Optional[int] = None,
    *,
    min_threads: int = DEFAULT_MIN_THREADS,
    max_threads: int = DEFAULT_MAX_THREADS,
    available_cpus: Optional[int] = None,
) -> int:
    """Return a torch intra-op thread count for one engine among ``n_engines_on_host``.

    The available CPUs are divided evenly among the engines, then clamped to ``[min_threads, max_threads]``, and
    never exceed the available CPUs. See the module docstring for the measurement behind the defaults.

    Args:
        n_engines_on_host: Number of engine processes that share this host's CPUs, e.g. one per NPU card. ``None``
            means 1.
        min_threads: Lower bound per engine.
        max_threads: Upper bound per engine.
        available_cpus: CPUs to divide. ``None`` reads :func:`available_host_cpus`.

    Returns:
        The recommended thread count, at least 1.

    Raises:
        ValueError: If ``n_engines_on_host`` or ``min_threads`` is below 1, or ``max_threads`` is below
            ``min_threads``.
    """
    n_engines = 1 if n_engines_on_host is None else int(n_engines_on_host)
    if n_engines < 1:
        raise ValueError(f"n_engines_on_host must be at least 1, got {n_engines_on_host!r}.")
    if min_threads < 1:
        raise ValueError(f"min_threads must be at least 1, got {min_threads!r}.")
    if max_threads < min_threads:
        raise ValueError(f"max_threads ({max_threads!r}) must not be below min_threads ({min_threads!r}).")
    cpus = available_host_cpus() if available_cpus is None else max(1, int(available_cpus))
    threads = min(max_threads, max(min_threads, cpus // n_engines))
    return min(threads, cpus)


def explicit_thread_env(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """Return the name of the first thread-count variable in :data:`THREAD_ENV_VARS` that is set, else ``None``.

    Args:
        environ: Environment to inspect. ``None`` reads ``os.environ``.
    """
    env = os.environ if environ is None else environ
    for name in THREAD_ENV_VARS:
        if env.get(name, "").strip():
            return name
    return None


def apply_host_thread_defaults(
    n_engines_on_host: Optional[int] = None,
    *,
    min_threads: int = DEFAULT_MIN_THREADS,
    max_threads: int = DEFAULT_MAX_THREADS,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[int]:
    """Lower torch's intra-op thread count to :func:`recommended_host_threads` unless the user set it explicitly.

    Never raises the count above torch's current value. Does nothing when ``OMP_NUM_THREADS`` or ``MKL_NUM_THREADS``
    is set: those already size torch's pool, and an explicit value is the user's decision. Call it once per process,
    before the first torch CPU work.

    Args:
        n_engines_on_host: Number of engine processes that share this host's CPUs. ``None`` means 1.
        min_threads: Lower bound per engine.
        max_threads: Upper bound per engine.
        environ: Environment to inspect. ``None`` reads ``os.environ``.

    Returns:
        The thread count torch uses afterwards, or ``None`` when an explicit environment setting was respected.
    """
    if explicit_thread_env(environ) is not None:
        return None
    threads = recommended_host_threads(n_engines_on_host, min_threads=min_threads, max_threads=max_threads)
    import torch

    # Only ever lower the pool: torch's own default counts physical cores, while the CPU count above includes SMT
    # siblings, and a pool larger than the default has not been measured to help.
    threads = min(threads, torch.get_num_threads())
    torch.set_num_threads(threads)
    return threads


__all__ = [
    "DEFAULT_MAX_THREADS",
    "DEFAULT_MIN_THREADS",
    "THREAD_ENV_VARS",
    "apply_host_thread_defaults",
    "available_host_cpus",
    "explicit_thread_env",
    "recommended_host_threads",
]
