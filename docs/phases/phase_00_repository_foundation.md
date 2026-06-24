# Phase 00: Project Setup

## What We Built

We initialized the project foundation before writing model code. This phase set
up the repo files, Python package layout, basic testing/linting tools, and the
GitHub push workflow.

## What I Learned

- `README.md` stores the project details and overview.
- `AGENTS.md` stores instructions for the coding agent.
- `pyproject.toml` is the main project setup/config file.
- `ruff` checks code quality and catches suspicious mistakes or messy code.
- `pytest` runs tests using `assert`.
- `black` is a code formatter, even though this phase uses Ruff for checks.
- `.gitignore` tells Git which files/folders not to track.
- `.python-version` records the Python version for the project.
- `src/` is the source code folder.
- `src/llm_lab/` is the Python package, so the project can be imported with
  `import llm_lab`.
- `src/llm_lab/__init__.py` marks the folder as a Python package and stores the
  package version.
- `tests/` contains tests to check whether important parts work properly.
- GitHub should be created as an empty repo for this case: no README, no
  license, and no `.gitignore`, because those files already exist locally.
- If pushing a GitHub Actions workflow fails, the GitHub token needs workflow
  permission enabled.

## High-Level Understanding

Phase 0 is project setup. It does not build the LLM yet. It makes sure the repo
has a clean structure, can be imported as a Python package, can run tests, can
run code checks, and can be pushed to GitHub.

## Intuition / Small Example

The simplest proof that the setup works is importing the package:

```bash
python -c "import llm_lab"
```

If this works, Python can find the package inside `src/llm_lab`. Later, when
the model code is added, it can live inside the same package structure.

## Detailed Explanation

The repo starts with documentation and config files:

- `README.md` explains the project.
- `AGENTS.md` explains how the agent should help in this repo.
- `pyproject.toml` defines package metadata, Python version, dev tools, pytest
  config, and Ruff config.
- `.gitignore` keeps local/generated files out of Git.
- `.python-version` records the Python version.

The source code starts in `src/llm_lab/`. The `__init__.py` file makes
`llm_lab` importable and currently stores the version.

The first test is in `tests/test_import.py`. It imports `llm_lab` to confirm
the package setup works before any model code exists.

The GitHub setup flow was:

```bash
git add .
git commit -m "Complete phase 0 repository foundation"
git remote add origin "<repo link>"
git push -u origin main
```

The GitHub repo should be created empty first. Do not initialize it with a
README, license, or `.gitignore`, because those files are already in the local
repo.

One issue came up during push: GitHub rejected the workflow file because the
token did not have workflow permission. The fix was to edit/refresh the token
and enable workflow permission, then push again.

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
- GitHub Actions CI passes after pushing.

## Open Questions

- Should this repo use plain `pip` or `uv` long term?
- When should the first real dependency, PyTorch, be added?
- Should we add `black`, or keep Ruff as the only formatting/checking tool for
  now?
