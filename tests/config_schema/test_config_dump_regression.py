"""Regression: merged config JSON matches golden (env merge + flatten)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def test_merged_config_minimal_snapshot_matches_golden() -> None:
    """Subprocess + --env-clear isolates from developer shell env vars."""
    yaml_p = _ROOT / "tests" / "fixtures" / "config" / "minimal.yaml"
    golden = _ROOT / "tests" / "fixtures" / "config" / "merged_snapshot_minimal.json"
    script = _ROOT / "scripts" / "dump_merged_config_snapshot.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--yaml",
            str(yaml_p),
            "--env-clear",
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.fail(proc.stderr or f"exit {proc.returncode}")
    assert proc.stdout == golden.read_text(encoding="utf-8")


def test_dump_script_env_override_changes_gpu_vendor() -> None:
    """Sanity: --env is applied after --env-clear (merge path exercised)."""
    yaml_p = _ROOT / "tests" / "fixtures" / "config" / "minimal.yaml"
    script = _ROOT / "scripts" / "dump_merged_config_snapshot.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--yaml",
            str(yaml_p),
            "--env-clear",
            "--env",
            "GPU_VENDOR=intel",
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert '"GPU_VENDOR": "intel"' in proc.stdout
