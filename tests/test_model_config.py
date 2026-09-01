import pytest

from llm_lab.model.config import ModernModelConfig


def test_modern_model_config_stores_architecture_values():
    config = ModernModelConfig(
        vocab_size=512,
        block_size=128,
        d_model=64,
        n_head=4,
        n_layer=3,
    )

    # Verify the five required values.
    assert config.vocab_size == 512
    assert config.block_size == 128
    assert config.d_model == 64
    assert config.n_head == 4
    assert config.n_layer == 3
    # Also verify the component defaults.
    assert config.normalization == "layer_norm"
    assert config.activation == "relu"
    assert config.attention == "causal_mha"
    assert config.positional_embedding == "learned"


def test_modern_model_config_rejects_non_positive_vocab_size():
    with pytest.raises(ValueError, match="vocab_size"):
        ModernModelConfig(
            vocab_size=0,
            block_size=128,
            d_model=64,
            n_head=4,
            n_layer=3,
        )


def test_modern_model_config_rejects_non_positive_block_size():
    with pytest.raises(ValueError, match="block_size"):
        ModernModelConfig(
            vocab_size=512,
            block_size=-1,
            d_model=64,
            n_head=4,
            n_layer=3,
        )


def test_modern_model_config_rejects_non_positive_d_model():
    with pytest.raises(ValueError, match="d_model"):
        ModernModelConfig(
            vocab_size=512,
            block_size=128,
            d_model=-1,
            n_head=4,
            n_layer=3,
        )


def test_modern_model_config_rejects_non_positive_n_head():
    with pytest.raises(ValueError, match="n_head"):
        ModernModelConfig(
            vocab_size=512,
            block_size=128,
            d_model=64,
            n_head=0,
            n_layer=3,
        )


def test_modern_model_config_rejects_non_positive_n_layer():
    with pytest.raises(ValueError, match="n_layer"):
        ModernModelConfig(
            vocab_size=512,
            block_size=128,
            d_model=64,
            n_head=4,
            n_layer=0,
        )


def test_modern_model_config_requires_even_head_split():
    with pytest.raises(ValueError, match="divisible"):
        ModernModelConfig(
            vocab_size=512,
            block_size=128,
            d_model=64,
            n_head=3,
            n_layer=3,
        )


def test_modern_model_config_rejects_unknown_normalization():
    with pytest.raises(ValueError, match="normalization"):
        ModernModelConfig(
            vocab_size=512,
            block_size=128,
            d_model=64,
            n_head=4,
            n_layer=3,
            normalization="unknown_norm",
        )


def test_modern_model_config_rejects_unknown_activation():
    with pytest.raises(ValueError, match="activation"):
        ModernModelConfig(
            vocab_size=512,
            block_size=128,
            d_model=64,
            n_head=4,
            n_layer=3,
            activation="unknown_activation",
        )


def test_modern_model_config_rejects_unknown_attention():
    with pytest.raises(ValueError, match="attention"):
        ModernModelConfig(
            vocab_size=512,
            block_size=128,
            d_model=64,
            n_head=4,
            n_layer=3,
            attention="unknown_attention",
        )


def test_modern_model_config_accepts_rms_norm():
    config = ModernModelConfig(
        vocab_size=512,
        block_size=128,
        d_model=64,
        n_head=4,
        n_layer=3,
        normalization="rms_norm",
    )

    assert config.normalization == "rms_norm"


def test_modern_model_config_accepts_rope_position_encoding():
    config = ModernModelConfig(
        vocab_size=512,
        block_size=128,
        d_model=64,
        n_head=4,
        n_layer=3,
        positional_embedding="rope",
    )

    assert config.positional_embedding == "rope"


def test_modern_model_config_rejects_unknown_position_encoding():
    with pytest.raises(ValueError, match="pos_embedding"):
        ModernModelConfig(
            vocab_size=512,
            block_size=128,
            d_model=64,
            n_head=4,
            n_layer=3,
            positional_embedding="unknown_position_encoding",
        )
