from dataclasses import dataclass

SUPPORTED_NORMALIZATIONS = frozenset({"layer_norm", "rms_norm"})
SUPPORTED_ACTIVATIONS = frozenset({"relu"})
SUPPORTED_ATTENTION_TYPES = frozenset({"causal_mha"})
SUPPORTED_POSITIONAL_EMBEDDING = frozenset({"learned", "rope"})


@dataclass
class ModernModelConfig:
    # Required size fields
    vocab_size: int
    block_size: int
    d_model: int
    n_head: int
    n_layer: int

    # Component choices with Phase 9 defaults
    normalization: str = "layer_norm"
    activation: str = "relu"
    attention: str = "causal_mha"
    positional_embedding: str = "learned"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.normalization not in SUPPORTED_NORMALIZATIONS:
            raise ValueError(
                f"Unsupported normalization: {self.normalization}. Supported normalizations are: {SUPPORTED_NORMALIZATIONS}"
            )
        if self.activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation: {self.activation}. Supported activations are: {SUPPORTED_ACTIVATIONS}"
            )

        if self.attention not in SUPPORTED_ATTENTION_TYPES:
            raise ValueError(
                f"Unsupported attention type: {self.attention}. Supported attention types are: {SUPPORTED_ATTENTION_TYPES}"
            )

        if self.positional_embedding not in SUPPORTED_POSITIONAL_EMBEDDING:
            raise ValueError(
                f"Unsupported pos_embedding type: {self.positional_embedding}. Supported attention types are: {SUPPORTED_POSITIONAL_EMBEDDING}"
            )
