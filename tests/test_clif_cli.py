"""``icutwin_clif`` task dispatch (steps replaced by recorders; nothing is built)."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from icutwin.clif import cli


def _record(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    for step in list(cli.STEPS):
        monkeypatch.setitem(cli.STEPS, step, lambda cfg, name, step=step: calls.append((step, name)))
    return calls


def test_config_composes_with_every_dataset() -> None:
    with initialize_config_module("icutwin.configs", version_base="1.3"):
        cfg = compose("clif", overrides=["task=pipeline"])
    assert list(cfg.datasets) == ["mimic", "aumc", "eicu", "hirid", "sicdb", "nwicu", "inspire"]
    assert cfg.utc_offset_hours.mimic == -5 and cfg.strict is True


def test_pipeline_runs_every_step_for_one_dataset_before_the_next(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record(monkeypatch)
    cli.run(OmegaConf.create({"task": "pipeline", "datasets": ["mimic", "eicu"]}))
    assert calls == [(s, d) for d in ("mimic", "eicu") for s in cli.PIPELINE]


def test_reference_tool_task_runs_only_for_its_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record(monkeypatch)
    cli.run(OmegaConf.create({"task": "aumc_tool", "datasets": ["mimic", "aumc", "eicu"]}))
    assert calls == [("aumc_tool", "aumc")]


def test_unknown_task_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record(monkeypatch)
    with pytest.raises(ValueError, match="unknown task"):
        cli.run(OmegaConf.create({"task": "adapters", "datasets": ["eicu"]}))
    assert calls == []


def test_build_is_tool_plus_supplement_or_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []
    for fn in ("_tool", "_supplement", "_adapter"):
        monkeypatch.setattr(cli, fn, lambda cfg, name, fn=fn: calls.append((fn, name)))
    cli._build(None, "mimic")
    cli._build(None, "hirid")
    assert calls == [("_tool", "mimic"), ("_supplement", "mimic"), ("_adapter", "hirid")]


def test_validate_step_stops_the_pipeline_on_hard_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "validate_canonical_dir",
                        lambda clif_dir, work: {"passed": False, "n_hard_errors": 2, "flags": {}})
    cfg = OmegaConf.create({"harmonized_root": str(tmp_path), "paths": {"tmp": str(tmp_path)}})
    with pytest.raises(ValueError, match="validation failed"):
        cli._validate(cfg, "eicu")
