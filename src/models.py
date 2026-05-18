from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class IterationRecord:
    """Данные одной итерации метода оптимизации."""

    k: int
    x: np.ndarray
    f_value: float
    grad_norm: float
    step_size: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OptimizationResult:
    """Итог работы метода оптимизации."""

    method_name: str
    function_name: str
    x_star: np.ndarray
    f_star: float
    grad_norm: float
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    hessian_evaluations: int
    history: list[IterationRecord]
    converged: bool
    message: str


@dataclass(slots=True)
class OptimizationParams:
    """Параметры методов оптимизации."""

    epsilon: float = 1e-3
    max_iterations: int = 1000
    initial_step: float = 1.0
    armijo_c1: float = 1e-4
    backtracking_factor: float = 0.5
    min_step: float = 1e-12
    cg_restart_frequency: int | None = None
    newton_regularization: float = 1e-6


@dataclass(slots=True)
class FunctionEvaluationCounter:
    """Счётчики вызовов функции, градиента и Гессиана."""

    function_calls: int = 0
    gradient_calls: int = 0
    hessian_calls: int = 0

    def reset(self) -> None:
        self.function_calls = 0
        self.gradient_calls = 0
        self.hessian_calls = 0