import pytest
import torch

from llm_lab.model.checkpoint import load_checkpoint
from llm_lab.model.config import ModernModelConfig
from llm_lab.model.modern_lm import ModernLanguageModel


def test_load_checkpoint_rebuilds_modern_model(tmp_path):
    config = ModernModelConfig(
        vocab_size=32,
        block_size=8,
        d_model=8,
        n_head=2,
        n_layer=1,
        normalization="rms_norm",
        positional_embedding="rope",
    )
    model = ModernLanguageModel(config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_config": config.__dict__,
            "model_state_dict": model.state_dict(),
            "variant": {
                "name": "rms_norm_rope",
                "normalization": "rms_norm",
                "positional_embedding": "rope",
            },
            "seed": 0,
            "final_metrics": {"val_loss_8": 1.0},
        },
        checkpoint_path,
    )

    loaded = load_checkpoint(checkpoint_path)

    assert loaded.path == checkpoint_path
    assert loaded.config == config
    assert loaded.model.position_embeddings is None
    assert not loaded.model.training
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, loaded.model.state_dict()[name])


def test_load_checkpoint_rejects_missing_weights(tmp_path):
    checkpoint_path = tmp_path / "invalid.pt"
    torch.save({"model_config": {"vocab_size": 8}}, checkpoint_path)

    with pytest.raises(ValueError, match="model_state_dict"):
        load_checkpoint(checkpoint_path)
