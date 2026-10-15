"""Host thread defaults: CPU discovery, per-engine division, and the explicit-environment override."""

from pathlib import Path

import pytest

from transformers_mblt.utils import host_threads


@pytest.fixture(autouse=True)
def no_proc_cgroup(tmp_path, monkeypatch):
    """Keep the host's own /proc/self/cgroup out of every test; tests that need one write it."""
    monkeypatch.setattr(host_threads, "_PROC_SELF_CGROUP", tmp_path / "proc-self-cgroup-missing")


@pytest.fixture
def no_cgroup(tmp_path, monkeypatch):
    monkeypatch.setattr(host_threads, "_CGROUP_ROOT", tmp_path / "missing")


def _set_affinity(monkeypatch, n_cpus: int) -> None:
    monkeypatch.setattr(host_threads.os, "sched_getaffinity", lambda pid: set(range(n_cpus)), raising=False)


def test_available_cpus_follow_affinity(monkeypatch, no_cgroup):
    _set_affinity(monkeypatch, 16)
    monkeypatch.setattr(host_threads.os, "cpu_count", lambda: 128)

    assert host_threads.available_host_cpus() == 16


def test_available_cpus_fall_back_to_cpu_count(monkeypatch, no_cgroup):
    monkeypatch.delattr(host_threads.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(host_threads.os, "cpu_count", lambda: 12)

    assert host_threads.available_host_cpus() == 12


@pytest.mark.parametrize(
    ("cpu_max", "expected"),
    [("400000 100000\n", 4), ("150000 100000\n", 2), ("max 100000\n", 16)],
)
def test_available_cpus_honor_cgroup_v2_quota(tmp_path, monkeypatch, cpu_max, expected):
    (tmp_path / "cpu.max").write_text(cpu_max)
    monkeypatch.setattr(host_threads, "_CGROUP_ROOT", tmp_path)
    _set_affinity(monkeypatch, 16)

    assert host_threads.available_host_cpus() == expected


def test_available_cpus_honor_cgroup_v1_quota(tmp_path, monkeypatch):
    v1 = tmp_path / "cpu"
    v1.mkdir()
    (v1 / "cpu.cfs_quota_us").write_text("300000\n")
    (v1 / "cpu.cfs_period_us").write_text("100000\n")
    monkeypatch.setattr(host_threads, "_CGROUP_ROOT", tmp_path)
    _set_affinity(monkeypatch, 16)

    assert host_threads.available_host_cpus() == 3


def test_cgroup_v1_without_quota_is_ignored(tmp_path):
    v1 = tmp_path / "cpu"
    v1.mkdir()
    (v1 / "cpu.cfs_quota_us").write_text("-1\n")
    (v1 / "cpu.cfs_period_us").write_text("100000\n")

    assert host_threads._cgroup_cpu_limit(tmp_path) is None
    assert host_threads._cgroup_cpu_limit(Path(tmp_path / "missing")) is None


def _write_proc_cgroup(tmp_path, monkeypatch, text: str) -> None:
    proc = tmp_path / "proc-self-cgroup"
    proc.write_text(text)
    monkeypatch.setattr(host_threads, "_PROC_SELF_CGROUP", proc)


def test_available_cpus_honor_nested_cgroup_v2_quota(tmp_path, monkeypatch):
    # A systemd service: the quota sits in the service's own cgroup, the root has none.
    root = tmp_path / "cgroup"
    service = root / "system.slice" / "engine.service"
    service.mkdir(parents=True)
    (root / "cpu.max").write_text("max 100000\n")
    (root / "system.slice" / "cpu.max").write_text("max 100000\n")
    (service / "cpu.max").write_text("200000 100000\n")
    monkeypatch.setattr(host_threads, "_CGROUP_ROOT", root)
    _write_proc_cgroup(tmp_path, monkeypatch, "0::/system.slice/engine.service\n")
    _set_affinity(monkeypatch, 16)

    assert host_threads.available_host_cpus() == 2


def test_nested_cgroup_v2_takes_the_tightest_ancestor_quota(tmp_path, monkeypatch):
    root = tmp_path / "cgroup"
    leaf = root / "parent" / "leaf"
    leaf.mkdir(parents=True)
    (root / "parent" / "cpu.max").write_text("300000 100000\n")
    (leaf / "cpu.max").write_text("600000 100000\n")
    monkeypatch.setattr(host_threads, "_CGROUP_ROOT", root)
    _write_proc_cgroup(tmp_path, monkeypatch, "0::/parent/leaf\n")
    _set_affinity(monkeypatch, 16)

    assert host_threads.available_host_cpus() == 3


def test_available_cpus_honor_nested_cgroup_v1_quota(tmp_path, monkeypatch):
    root = tmp_path / "cgroup"
    group = root / "cpu" / "engines" / "card0"
    group.mkdir(parents=True)
    (root / "cpu" / "cpu.cfs_quota_us").write_text("-1\n")
    (root / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    (group / "cpu.cfs_quota_us").write_text("250000\n")
    (group / "cpu.cfs_period_us").write_text("100000\n")
    monkeypatch.setattr(host_threads, "_CGROUP_ROOT", root)
    _write_proc_cgroup(tmp_path, monkeypatch, "12:memory:/engines/card0\n4:cpu,cpuacct:/engines/card0\n")
    _set_affinity(monkeypatch, 16)

    assert host_threads.available_host_cpus() == 3


def test_invisible_cgroup_levels_are_skipped(tmp_path, monkeypatch):
    # Inside a cgroup namespace the host path in /proc/self/cgroup may not exist under the mount; the root still counts.
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cpu.max").write_text("400000 100000\n")
    monkeypatch.setattr(host_threads, "_CGROUP_ROOT", root)
    _write_proc_cgroup(tmp_path, monkeypatch, "0::/host/only/path\n")
    _set_affinity(monkeypatch, 16)

    assert host_threads.available_host_cpus() == 4


@pytest.mark.parametrize(
    ("cpus", "n_engines", "expected"),
    [
        (16, None, 8),  # one engine: capped
        (16, 1, 8),
        (16, 3, 5),  # three engines on a 16-core reservation: 15 threads in total
        (16, 8, 2),
        (16, 16, 2),  # floor
        (4, 1, 4),
        (1, 1, 1),  # never more threads than CPUs
        (2, 4, 2),
    ],
)
def test_recommended_threads_divide_cpus_among_engines(cpus, n_engines, expected):
    assert host_threads.recommended_host_threads(n_engines, available_cpus=cpus) == expected


def test_recommended_threads_read_affinity_by_default(monkeypatch, no_cgroup):
    _set_affinity(monkeypatch, 6)

    assert host_threads.recommended_host_threads(2) == 3


def test_recommended_threads_custom_bounds():
    assert host_threads.recommended_host_threads(1, min_threads=1, max_threads=32, available_cpus=64) == 32
    assert host_threads.recommended_host_threads(32, min_threads=1, max_threads=32, available_cpus=64) == 2


@pytest.mark.parametrize(
    "kwargs",
    [{"n_engines_on_host": 0}, {"min_threads": 0}, {"min_threads": 4, "max_threads": 2}],
)
def test_recommended_threads_reject_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        host_threads.recommended_host_threads(available_cpus=8, **kwargs)


class _FakeTorch:
    def __init__(self, num_threads: int = 16):
        self.num_threads = num_threads
        self.calls = []

    def get_num_threads(self):
        return self.num_threads

    def set_num_threads(self, n):
        self.calls.append(n)
        self.num_threads = n


@pytest.fixture
def fake_torch(monkeypatch):
    import sys

    fake = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


def test_apply_sets_torch_threads_without_explicit_env(monkeypatch, no_cgroup, fake_torch):
    _set_affinity(monkeypatch, 16)

    assert host_threads.apply_host_thread_defaults(3, environ={}) == 5
    assert fake_torch.calls == [5]


@pytest.mark.parametrize("name", ["OMP_NUM_THREADS", "MKL_NUM_THREADS"])
def test_apply_respects_explicit_env(monkeypatch, no_cgroup, fake_torch, name):
    _set_affinity(monkeypatch, 16)

    assert host_threads.apply_host_thread_defaults(3, environ={name: "4"}) is None
    assert fake_torch.calls == []


def test_apply_ignores_empty_env_value(monkeypatch, no_cgroup, fake_torch):
    _set_affinity(monkeypatch, 16)

    assert host_threads.apply_host_thread_defaults(environ={"OMP_NUM_THREADS": " "}) == 8
    assert fake_torch.calls == [8]


def test_apply_reads_process_env_by_default(monkeypatch, no_cgroup, fake_torch):
    _set_affinity(monkeypatch, 16)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")

    assert host_threads.apply_host_thread_defaults() is None
    assert fake_torch.calls == []


def test_apply_never_raises_torch_default(monkeypatch, no_cgroup, fake_torch):
    _set_affinity(monkeypatch, 16)  # SMT siblings counted: torch defaults to the 4 physical cores
    fake_torch.num_threads = 4

    assert host_threads.apply_host_thread_defaults(1, environ={}) == 4
    assert fake_torch.calls == [4]
