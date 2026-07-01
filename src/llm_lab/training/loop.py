import torch
import torch.nn.functional as F


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, vocab_size = logits.shape
    logits_flat = logits.view(batch_size * sequence_length, vocab_size)
    targets_flat = targets.view(batch_size * sequence_length)
    loss = F.cross_entropy(logits_flat, targets_flat)
    return loss


def make_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    model.train()
    optimizer.zero_grad()
    logits = model(x)
    loss = compute_loss(logits, y)
    loss.backward()
    optimizer.step()
    return loss


def validation_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = compute_loss(logits, y)
    return loss


def generate(
    model: torch.nn.Module,
    start_token_ids: torch.Tensor,
    max_tokens: int,
) -> torch.Tensor:
    model.eval()
    token_ids = start_token_ids
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(token_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            next_token = next_token.unsqueeze(-1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
    return token_ids
