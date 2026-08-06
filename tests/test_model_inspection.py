from torch import nn

from llm_lab.model.config import ModernModelConfig
from llm_lab.model.inspection import (
    ParameterReport,
    build_parameter_report,
    count_parameters,
    count_parameters_by_component,
)
from llm_lab.model.modern_lm import ModernLanguageModel
from llm_lab.model.modern_block import ModernTransformerBlock


def test_count_parameters_matches_hand_calculated_total():
    model = nn.Sequential(
        nn.Linear(
            in_features=3,
            out_features=4,
        ),
        nn.Linear(
            in_features=4,
            out_features=2,
            bias=False,
        ),
    )

    parameter_count = count_parameters(model)

    assert parameter_count == 24
    assert isinstance(parameter_count, int)


def test_count_parameters_can_include_only_trainable_parameters():
    model = nn.Sequential(
        nn.Linear(
            in_features=3,
            out_features=4,
        ),
        nn.Linear(
            in_features=4,
            out_features=2,
            bias=False,
        ),
    )

    model[0].weight.requires_grad_(False)

    total = count_parameters(model)
    trainable = count_parameters(
        model,
        trainable_only=True,
    )
    frozen = total - trainable

    assert total == 24
    assert trainable == 12
    assert frozen == 12


def test_count_parameters_counts_shared_weight_only_once():
    embedding = nn.Embedding(
        num_embeddings=5,
        embedding_dim=3,
    )
    output_head = nn.Linear(
        in_features=3,
        out_features=5,
        bias=False,
    )

    output_head.weight = embedding.weight

    model = nn.ModuleDict(
        {
            "embedding": embedding,
            "output_head": output_head,
        }
    )

    assert model["embedding"].weight is model["output_head"].weight

    parameter_count = count_parameters(model)

    assert parameter_count == 15


def test_count_parameters_deduplicates_real_model_tied_weight():
    config = ModernModelConfig(
        vocab_size=16,
        block_size=8,
        d_model=8,
        n_head=2,
        n_layer=1,
    )
    model = ModernLanguageModel(config)

    assert model.output_head.weight is model.token_embeddings.weight

    unique_count = count_parameters(model)
    count_with_duplicate_paths = sum(
        parameter.numel()
        for _, parameter in model.named_parameters(remove_duplicate=False)
    )

    assert (
        count_with_duplicate_paths - unique_count == config.vocab_size * config.d_model
    )


def test_parameter_breakdown_counts_real_model_components():
    config = ModernModelConfig(
        vocab_size=16,
        block_size=8,
        d_model=8,
        n_head=2,
        n_layer=1,
    )
    model = ModernLanguageModel(config)

    breakdown = count_parameters_by_component(model)

    assert set(breakdown) == {
        "token_embeddings",
        "position_embeddings",
        "blocks",
        "final_norm",
        "output_head",
    }

    assert breakdown["token_embeddings"] == config.vocab_size * config.d_model
    assert breakdown["position_embeddings"] == config.block_size * config.d_model
    assert breakdown["output_head"] == 0

    assert sum(breakdown.values()) == count_parameters(model)


def test_build_parameter_report_combines_parameter_statistics():
    config = ModernModelConfig(
        vocab_size=16,
        block_size=8,
        d_model=8,
        n_head=2,
        n_layer=1,
    )
    model = ModernLanguageModel(config)

    model.final_norm.weight.requires_grad_(False)

    report = build_parameter_report(model)

    assert isinstance(report, ParameterReport)
    assert report.total == count_parameters(model)
    assert report.trainable == count_parameters(
        model,
        trainable_only=True,
    )
    assert report.frozen == model.final_norm.weight.numel()
    assert report.by_component == count_parameters_by_component(model)
    assert sum(report.by_component.values()) == report.total


def test_parameter_count_scales_with_vocabulary_size():
    small_config = ModernModelConfig(
        vocab_size=16,
        block_size=8,
        d_model=8,
        n_head=2,
        n_layer=1,
    )
    large_config = ModernModelConfig(
        vocab_size=24,
        block_size=8,
        d_model=8,
        n_head=2,
        n_layer=1,
    )

    small_model = ModernLanguageModel(small_config)
    large_model = ModernLanguageModel(large_config)

    difference = count_parameters(large_model) - count_parameters(small_model)

    expected = (
        large_config.vocab_size - small_config.vocab_size
    ) * small_config.d_model

    assert difference == expected


def test_parameter_count_scales_with_context_length():
    short_config = ModernModelConfig(
        vocab_size=16,
        block_size=8,
        d_model=8,
        n_head=2,
        n_layer=1,
    )
    long_config = ModernModelConfig(
        vocab_size=16,
        block_size=12,
        d_model=8,
        n_head=2,
        n_layer=1,
    )

    short_model = ModernLanguageModel(short_config)
    long_model = ModernLanguageModel(long_config)

    difference = count_parameters(long_model) - count_parameters(short_model)
    expected = (long_config.block_size - short_config.block_size) * short_config.d_model

    assert difference == expected


def test_parameter_count_scales_with_number_of_layers():
    shallow_config = ModernModelConfig(
        vocab_size=16,
        block_size=8,
        d_model=8,
        n_head=2,
        n_layer=1,
    )
    deep_config = ModernModelConfig(
        vocab_size=16,
        block_size=8,
        d_model=8,
        n_head=2,
        n_layer=2,
    )

    shallow_model = ModernLanguageModel(shallow_config)
    deep_model = ModernLanguageModel(deep_config)

    added_parameters = count_parameters(deep_model) - count_parameters(shallow_model)
    one_block_parameters = count_parameters(ModernTransformerBlock(shallow_config))

    assert added_parameters == one_block_parameters
