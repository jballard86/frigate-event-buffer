#!/usr/bin/env python3
"""Dump canonical JSON for merged config (YAML + environment).

Used to detect unintended changes to env merge or flattening before refactoring
``config.py``. Regenerate the golden file after intentional behavior changes::

    python scripts/dump_merged_config_snapshot.py \\
        --yaml tests/fixtures/config/minimal.yaml --env-clear \\
        > tests/fixtures/config/merged_snapshot_minimal.json

Why: :func:`frigate_buffer.config.load_config` has a long env-merge tail; a
stable JSON diff catches regressions when schema slices move between modules.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _bootstrap_src_path() -> None:
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _apply_env_clear() -> None:
    keep = (
        "PATH",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "USERPROFILE",
        "HOME",
        "TEMP",
        "TMP",
    )
    saved = {k: os.environ[k] for k in keep if k in os.environ}
    os.environ.clear()
    os.environ.update(saved)


def _parse_env_pairs(pairs: list[str]) -> None:
    for raw in pairs:
        if "=" not in raw:
            raise SystemExit(f"Invalid --env (expected KEY=VAL): {raw!r}")
        key, _, val = raw.partition("=")
        os.environ[key] = val


def _canonical_json(config: dict) -> str:
    return json.dumps(config, sort_keys=True, indent=2, default=str) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaml",
        required=True,
        type=Path,
        help="Path to YAML passed to load_config(yaml_path=...).",
    )
    parser.add_argument(
        "--env-clear",
        action="store_true",
        help="Clear os.environ except a small OS allowlist before merge.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Set environment variable after optional --env-clear (repeatable).",
    )
    args = parser.parse_args()
    yml = args.yaml.resolve()
    if not yml.is_file():
        print(f"YAML not found: {yml}", file=sys.stderr)
        return 2

    if args.env_clear:
        _apply_env_clear()
    _parse_env_pairs(args.env)

    _bootstrap_src_path()
    from frigate_buffer.config import load_config

    cfg = load_config(yaml_path=str(yml))
    sys.stdout.write(_canonical_json(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
