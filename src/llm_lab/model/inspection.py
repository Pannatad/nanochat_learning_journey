from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ParameterReport:
    total: int
    trainable: int
    frozen: int
    by_component: dict[str, int]


def count_parameters(
    module: nn.Module,
    *,
    trainable_only: bool = False,
) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def count_parameters_by_component(
    module: nn.Module,
) -> dict[str, int]:
    counts = {name: 0 for name, _ in module.named_children()}
    seen_parameter_ids: set[int] = set()

    for component_name, component in module.named_children():
        for parameter in component.parameters():
            parameter_id = id(parameter)

            if parameter_id in seen_parameter_ids:
                continue

            counts[component_name] += parameter.numel()
            seen_parameter_ids.add(parameter_id)

    return counts


def build_parameter_report(
    module: nn.Module,
) -> ParameterReport:
    total = count_parameters(module)
    trainable = count_parameters(
        module,
        trainable_only=True,
    )

    return ParameterReport(
        total=total,
        trainable=trainable,
        frozen=total - trainable,
        by_component=count_parameters_by_component(module),
    )
