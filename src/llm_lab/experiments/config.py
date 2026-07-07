import json
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def create_run_dir(root_dir: str | Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root_dir) / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_resolved_config(config: dict[str, Any], run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    output_path = run_dir / "resolved_config.yaml"

    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return output_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_environment_metadata(seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }


def save_metadata(metadata: dict[str, Any], run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    output_path = run_dir / "metadata.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return output_path


def diff_configs(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    diffs = {}

    def walk(
        baseline_part: dict[str, Any],
        candidate_part: dict[str, Any],
        prefix: str,
    ) -> None:
        for key in baseline_part.keys() | candidate_part.keys():
            path = f"{prefix}.{key}" if prefix else key
            baseline_value = baseline_part.get(key)
            candidate_value = candidate_part.get(key)
            if isinstance(baseline_value, dict) and isinstance(candidate_value, dict):
                walk(baseline_value, candidate_value, path)
            elif baseline_value != candidate_value:
                diffs[path] = {
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                }

    walk(baseline, candidate, "")
    return diffs
