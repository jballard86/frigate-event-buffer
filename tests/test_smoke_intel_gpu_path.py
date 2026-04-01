"""Tests for scripts/smoke_intel_gpu_path.py (CLI and _run_vainfo)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "smoke_intel_gpu_path.py"


def _load_smoke_module() -> object:
    spec = importlib.util.spec_from_file_location("smoke_intel_gpu_path_cli", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_help_exits_zero() -> None:
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert "vainfo" in r.stdout


def test_strict_dri_without_vainfo_exits_2() -> None:
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--strict-dri"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 2


def test_run_vainfo_strict_fails_when_missing_binary() -> None:
    mod = _load_smoke_module()
    with patch.object(mod.shutil, "which", return_value=None):
        assert mod._run_vainfo(strict_dri=True) == 3
        assert mod._run_vainfo(strict_dri=False) == 0


def test_run_vainfo_strict_fails_on_nonzero_exit() -> None:
    mod = _load_smoke_module()
    proc = MagicMock()
    proc.returncode = 1
    proc.stdout = ""
    proc.stderr = "err"
    with patch.object(mod.shutil, "which", return_value="/bin/vainfo"):
        with patch.object(mod.subprocess, "run", return_value=proc):
            assert mod._run_vainfo(strict_dri=True) == 3
            assert mod._run_vainfo(strict_dri=False) == 0
