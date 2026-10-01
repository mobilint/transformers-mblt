"""Tests for the ``transformers-mblt`` entry point."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys

import pytest

# ``transformers_mblt.cli`` re-exports the ``main`` function, which shadows the ``main`` module attribute.
cli_main = importlib.import_module("transformers_mblt.cli.main")


def test_build_parser_exposes_list_and_tps() -> None:
    parser = cli_main.build_parser()
    assert parser.prog == "transformers-mblt"
    args = parser.parse_args(["list", "--task", "text-generation", "--json"])
    assert args.task == ["text-generation"]
    assert args.json is True
    args = parser.parse_args(["tps", "measure", "--model", "mobilint/Llama-3.2-1B-Instruct"])
    assert args.model == "mobilint/Llama-3.2-1B-Instruct"


def test_list_rejects_unknown_task() -> None:
    with pytest.raises(SystemExit):
        cli_main.build_parser().parse_args(["list", "--task", "object-detection"])


def test_list_prints_models(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    seen: dict[str, object] = {}

    def _fake_list_models(tasks, include_private=False):
        seen["tasks"] = tasks
        seen["include_private"] = include_private
        return {"text-generation": ["mobilint/Llama-3.2-1B-Instruct"]}

    monkeypatch.setattr("transformers_mblt.utils.api.list_models", _fake_list_models)
    monkeypatch.setattr(sys, "argv", ["transformers-mblt", "list", "--task", "text-generation", "--json"])

    assert cli_main.main() == 0
    assert json.loads(capsys.readouterr().out) == {"text-generation": ["mobilint/Llama-3.2-1B-Instruct"]}
    assert seen == {"tasks": ["text-generation"], "include_private": False}


def test_list_tasks_only(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from transformers_mblt.utils.api import list_tasks

    monkeypatch.setattr(sys, "argv", ["transformers-mblt", "list", "--tasks"])
    assert cli_main.main() == 0
    assert capsys.readouterr().out.split() == list_tasks()


def test_no_command_prints_help(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["transformers-mblt"])
    assert cli_main.main() == 1
    assert "usage: transformers-mblt" in capsys.readouterr().out


def test_module_entry_point_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "transformers_mblt.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "transformers-mblt" in result.stdout


def _fake_offline_cache(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Point the HF cache at ``tmp_path`` holding one cached Mobilint repo and make the Hub unreachable."""
    snapshot = tmp_path / "models--mobilint--Llama-3.2-1B-Instruct" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "README.md").write_text("---\npipeline_tag: text-generation\n---\n")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    def _hub_down(tasks, *, include_private):
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr("transformers_mblt.utils.api._list_models_from_hub", _hub_down)


@pytest.mark.parametrize("extra_args", [["--include-private"], []])
def test_list_json_stays_valid_on_hub_fallback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
    extra_args: list[str],
) -> None:
    """Fallback diagnostics go to stderr, and the offline cache lists every cached Mobilint repo."""
    _fake_offline_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["transformers-mblt", "list", "--task", "text-generation", "--json", *extra_args])

    assert cli_main.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"text-generation": ["mobilint/Llama-3.2-1B-Instruct"]}
    assert "Falling back to local cache" in captured.err
