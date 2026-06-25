# LLM From Scratch Lab

A small, readable repository for rebuilding a decoder-only language model from
scratch with PyTorch. This project is inspired by Andrej Karphaty nanochat repo

The project is built phase by phase. Phase 0 only sets up the repository
foundation: package layout, tests, linting, and CI.

## Phase 0 Checks

```bash
python -c "import llm_lab"
pytest
ruff check .
```

