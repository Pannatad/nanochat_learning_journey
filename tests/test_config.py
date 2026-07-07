import json

import torch
import yaml

from llm_lab.experiments.config import (
    collect_environment_metadata,
    create_run_dir,
    diff_configs,
    load_config,
    save_metadata,
    save_resolved_config,
    set_seed,
)


def test_load_config_reads_phase_06_smoke_values():
    config = load_config("configs/phase_06_smoke.yaml")

    assert config["training"]["steps"] == 10
    assert config["optimizer"]["lr"] == 0.01
    assert config["generation"]["prompt"] == "My name is"
    assert config["output"]["root_dir"] == "outputs/runs"


def test_create_run_dir_creates_timestamped_directory(tmp_path):
    run_dir = create_run_dir(tmp_path, "phase_06_smoke")

    assert run_dir.exists()
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path
    assert run_dir.name.startswith("phase_06_smoke_")


def test_save_resolved_config_saves_yaml(tmp_path):
    config = load_config("configs/phase_06_smoke.yaml")
    run_dir = create_run_dir(tmp_path, "phase_06_smoke")
    output_path = save_resolved_config(config, run_dir)

    assert output_path.exists()
    assert output_path.is_file()

    with output_path.open("r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)

    assert loaded_config == config


def test_set_seed_makes_torch_randomness_repeatable():
    seed = 42
    set_seed(seed)

    rand_tensor1 = torch.rand(3)
    set_seed(seed)
    rand_tensor2 = torch.rand(3)

    assert torch.equal(rand_tensor1, rand_tensor2)


def test_collect_environment_metadata_includes_reproducibility_fields():
    metadata = collect_environment_metadata(seed=1234)

    assert metadata["seed"] == 1234
    assert "python_version" in metadata
    assert "platform" in metadata
    assert "torch_version" in metadata
    assert "cuda_available" in metadata
    assert "mps_available" in metadata


def test_save_metadata_writes_json(tmp_path):
    metadata = collect_environment_metadata(seed=1234)
    run_dir = create_run_dir(tmp_path, "phase_06_smoke")

    output_path = save_metadata(metadata, run_dir)

    assert output_path.exists()
    assert output_path.is_file()
    assert output_path.name == "metadata.json"

    with output_path.open("r", encoding="utf-8") as file:
        loaded_metadata = json.load(file)

    assert loaded_metadata == metadata


def test_diff_configs_reports_nested_value_changes():
    baseline = {
        "optimizer": {
            "lr": 0.01,
            "weight_decay": 0.1,
        },
        "training": {
            "steps": 10,
        },
    }
    candidate = {
        "optimizer": {
            "lr": 0.001,
            "weight_decay": 0.1,
        },
        "training": {
            "steps": 20,
        },
    }

    diffs = diff_configs(baseline, candidate)

    assert diffs == {
        "optimizer.lr": {
            "baseline": 0.01,
            "candidate": 0.001,
        },
        "training.steps": {
            "baseline": 10,
            "candidate": 20,
        },
    }


def test_diff_configs_returns_empty_dict_when_configs_match():
    baseline = {
        "training": {
            "steps": 10,
        },
    }
    candidate = {
        "training": {
            "steps": 10,
        },
    }

    diffs = diff_configs(baseline, candidate)

    assert diffs == {}
