import torch


def build_rope_cache(
    sequence_length: int,
    head_dim: int,
    base: float = 10_000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cosine and sine values for every position and feature pair."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    if base <= 0:
        raise ValueError("base must be positive")

    pair_indices = torch.arange(
        0,
        head_dim,
        step=2,
        dtype=torch.float32,
        device=device,
    )
    frequencies = base ** (-pair_indices / head_dim)

    positions = torch.arange(
        0,
        sequence_length,
        dtype=torch.float32,
        device=device,
    )

    # (T, 1) * (1, head_dim / 2) -> (T, head_dim / 2)
    angles = positions[:, None] * frequencies[None, :]

    return angles.cos(), angles.sin()


def apply_rotary_embedding(
    x: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Apply pairwise rotary position embeddings to a query or key tensor."""
    if x.ndim != 3:
        raise ValueError("x must have shape (B, T, head_dim)")
    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    if not torch.is_floating_point(x):
        raise TypeError("x must be a floating-point tensor")

    expected_cache_shape = (x.shape[1], x.shape[2] // 2)
    if cosine.shape != expected_cache_shape:
        raise ValueError(f"cosine must have shape {expected_cache_shape}")
    if sine.shape != expected_cache_shape:
        raise ValueError(f"sine must have shape {expected_cache_shape}")
    if cosine.device != x.device or sine.device != x.device:
        raise ValueError("x, cosine, and sine must be on the same device")

    cosine = cosine.to(dtype=x.dtype)
    sine = sine.to(dtype=x.dtype)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rotated_even = x_even * cosine - x_odd * sine
    rotated_odd = x_even * sine + x_odd * cosine

    rotated_pairs = torch.stack((rotated_even, rotated_odd), dim=-1)
    return rotated_pairs.flatten(start_dim=-2)
