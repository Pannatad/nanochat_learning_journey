# Phase 00: Project Foundation

This phase set up the project before any model code existed. The goal was to
make the repo importable, testable, lintable, and ready to grow phase by phase.

The important idea is that an ML project is still a software project first. If
the package layout, tests, and tooling are clean early, every later model change
is easier to verify.

## Repository Files

`README.md` explains what the project is. It is the first place a reader should
look for the project overview.

`AGENTS.md` stores local instructions for how the coding agent should work in
this repo. In this project, those instructions say to keep explanations concise
and to build the main structure before refining smaller pieces.

`pyproject.toml` is the main Python project configuration file. It records the
package name, Python version, dependencies, pytest settings, and Ruff settings.
Instead of scattering tool configuration across many files, this repo keeps the
core Python setup in one place.

`.gitignore` tells Git which files should not be tracked. Typical examples are
virtual environments, caches, generated outputs, and local machine files.

`.python-version` records the Python version expected by the project. This
helps avoid the "works on my Python version" problem.

## Package Layout

The source code lives under:

```text
src/llm_lab/
```

The `src/` layout keeps package code separate from tests, docs, notebooks, and
scripts. The package itself is named `llm_lab`, so project code can be imported
with:

```python
import llm_lab
```

The file:

```text
src/llm_lab/__init__.py
```

makes `llm_lab` a Python package. It also stores the package version.

A small import test proves the project is wired correctly:

```python
import llm_lab
```

If this import works, Python can find the project package.

## Tests And Linting

`pytest` runs tests. Tests are small pieces of code that check expected
behavior with `assert`.

Example:

```python
assert llm_lab.__version__ == "0.1.0"
```

Ruff checks code quality. It catches issues like unused imports, formatting
problems, suspicious patterns, and style mistakes. Ruff gives fast feedback
before code becomes messy.

The basic local checks are:

```bash
.venv/bin/pytest
.venv/bin/ruff check .
```

These commands became the standard safety check for later phases.

## Git And GitHub Setup

Git tracks local file changes. GitHub stores the remote copy.

For this project, the GitHub repo should be created empty. Do not initialize it
with a README, license, or `.gitignore`, because those already exist locally.

The basic push flow is:

```bash
git add .
git commit -m "Complete phase 0 repository foundation"
git remote add origin "<repo link>"
git push -u origin main
```

One issue came up during push: GitHub rejected the workflow file because the
token did not have workflow permission. The fix was to enable workflow
permission for the token and push again.

## Small Example

The simplest proof that Phase 0 works is:

```bash
.venv/bin/python -c "import llm_lab; print(llm_lab.__version__)"
```

Expected output:

```text
0.1.0
```

That tiny command proves the package is importable, the `src/` layout works,
and the project has a version.
