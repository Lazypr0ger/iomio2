from __future__ import annotations

import numpy as np

from src.functions.base import ObjectiveFunction
from src.models import OptimizationParams


def armijo_backtracking(
    function: ObjectiveFunction,
    x: np.ndarray,
    direction: np.ndarray,
    gradient: np.ndarray,
    params: OptimizationParams,
) -> float:
    """
    Бэктрекинг Армихо.

    Подбирает длину шага lambda для направления direction.
    Направление должно быть направлением убывания.
    """

    step_size = params.initial_step
    current_value = function.value(x)
    directional_derivative = float(np.dot(gradient, direction))

    if directional_derivative >= 0:
        return params.min_step

    while step_size >= params.min_step:
        candidate_x = x + step_size * direction
        candidate_value = function.value(candidate_x)

        armijo_right_side = (
            current_value
            + params.armijo_c1 * step_size * directional_derivative
        )

        if candidate_value <= armijo_right_side:
            return step_size

        step_size *= params.backtracking_factor

    return params.min_step