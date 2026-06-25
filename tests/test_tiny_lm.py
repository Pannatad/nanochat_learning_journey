import torch
import torch.nn.functional as F

from llm_lab.model.tiny_lm import TinyLanguageModel


def test_tiny_language_model_created():
    model = TinyLanguageModel(256, 16)
    assert model is not None


def test_logits_shape():
    model = TinyLanguageModel(256, 16)
    x = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    logits = model(x)
    assert logits.shape == (2, 3, 256)


def test_finite_loss():
    model = TinyLanguageModel(256, 16)
    x = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    target = torch.tensor(
        [
            [2, 3, 4],
            [5, 6, 7],
        ]
    )
    logits = model(x)
    batch_size, sequence_length, vocab_size = logits.shape
    loss = F.cross_entropy(
        logits.view(batch_size * sequence_length, vocab_size),
        target.view(batch_size * sequence_length),
    )

    assert torch.isfinite(loss)
