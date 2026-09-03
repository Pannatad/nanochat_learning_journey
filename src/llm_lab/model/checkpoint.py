from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from llm_lab.model.config import ModernModelConfig
from llm_lab.model.modern_lm import ModernLanguageModel


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    payload: dict[str, Any]
    config: ModernModelConfig
    model: ModernLanguageModel


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Rebuild and load a ModernLanguageModel checkpoint."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a dictionary payload")

    raw_config = payload.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("Checkpoint is missing a model_config dictionary")

    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing a model_state_dict dictionary")

    try:
        config = ModernModelConfig(**raw_config)
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint contains an invalid model_config") from error

    model = ModernLanguageModel(config)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise ValueError("Checkpoint weights do not match model_config") from error

    model.to(device)
    model.eval()
    return LoadedCheckpoint(
        path=checkpoint_path,
        payload=payload,
        config=config,
        model=model,
    )
