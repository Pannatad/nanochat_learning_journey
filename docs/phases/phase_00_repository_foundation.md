# Phase 00: Repository Foundation

## What We Built

We created the project foundation: README, agent instructions, Python project
configuration, Git ignore rules, Python version marker, importable package,
first test, and GitHub Actions CI.

## What I Learned

- A Python project needs configuration before model code is useful.
- The `src/` layout keeps package code separate from tests and scripts.
- A tiny import test catches broken packaging early.
- Ruff and pytest give the repo a basic quality gate.

## High-Level Understanding

Phase 0 is the empty lab bench. It does not build the model yet. It proves the
repo can be installed, imported, linted, and tested.

## Intuition / Small Example

The most important check is simple:

```bash
python -c "import llm_lab"
```

If Python cannot import the package, later model code will be hard to run,
test, or share.

## Detailed Explanation

The package starts at `src/llm_lab/__init__.py`. The test in
`tests/test_import.py` imports that package and confirms the project setup is
valid.

`pyproject.toml` defines the package metadata, Python version requirement,
development dependencies, pytest settings, and Ruff settings.

## Experiments To Try

- Change the package version in `src/llm_lab/__init__.py` and check that the
  import command prints the new value.
- Temporarily break the package name in the test to see how pytest reports the
  failure.
- Run Ruff after adding an unused import to understand lint feedback.

## Tests / Checks

```bash
.venv/bin/python -c "import llm_lab; print(llm_lab.__version__)"
.venv/bin/pytest
.venv/bin/ruff check .
```

Expected result:

- import prints `0.1.0`;
- pytest passes;
- Ruff reports all checks passed.

## Open Questions

- Should this repo use plain `pip` or `uv` long term?
- When should the first real dependency, PyTorch, be added?
- How much CI should be added before the model code starts?

