# Project rules

**map.md** is the project’s source of truth. It defines architecture, directory structure, file placement, data flows, and coding conventions. Read it before making changes.

- **Structure and flows:** Follow the directory layout and file placement rules in map.md. Do not add root-level folders or new patterns without explicit permission.
- **Ruff:** Follow Ruff lint and format rules when making or editing Python files (see map.md §6 and **.cursor/rules/ruff-rules.mdc**). Run `ruff check src tests` and `ruff format src tests`; do not introduce new violations.
- **Cursor rules:** For Cursor-specific guidance, see **.cursor/rules/** (`.mdc` files). Those rules reinforce map.md and add Python-style conventions.
- **Conventions (see map.md §6):** Type hints on public APIs; Python 3.10+ union syntax (`str | None`); early returns; docstrings that explain *why* (Google or NumPy style); constants in `constants.py`; no magic numbers; tests as Setup → Execute → Verify; KISS.
- **Map maintenance:** When you add or remove files, change core flows, or rename important components, update **map.md** so it stays accurate.

This file is a short index. For full detail, use map.md and .cursor/rules/.
