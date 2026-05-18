from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.models import FunctionEvaluationCounter


class ObjectiveFunction(ABC):
    """Базовый интерфейс целевой функции."""

    def __init__(
        self,
        name: str,
        dimension: int,
        default_x0: np.ndarray,
        known_minimum: np.ndarray | None = None,
        known_minimum_value: float | None = None,
    ) -> None:
        self.name = name
        self.dimension = dimension
        self.default_x0 = np.array(default_x0, dtype=float)
        self.known_minimum = (
            None if known_minimum is None else np.array(known_minimum, dtype=float)
        )
        self.known_minimum_value = known_minimum_value
        self.counter = FunctionEvaluationCounter()

    def value(self, x: np.ndarray) -> float:
        self.counter.function_calls += 1
        return float(self._value(np.array(x, dtype=float)))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        self.counter.gradient_calls += 1
        return np.array(self._gradient(np.array(x, dtype=float)), dtype=float)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        self.counter.hessian_calls += 1
        return np.array(self._hessian(np.array(x, dtype=float)), dtype=float)

    def reset_counter(self) -> None:
        self.counter.reset()

    def validate_x(self, x: np.ndarray) -> None:
        if x.shape != (self.dimension,):
            raise ValueError(
                f"Для функции {self.name} ожидался вектор размера "
                f"{self.dimension}, получен размер {x.shape}."
            )

    @abstractmethod
    def _value(self, x: np.ndarray) -> float:
        """Значение функции без увеличения счётчика вызовов."""

    @abstractmethod
    def _gradient(self, x: np.ndarray) -> np.ndarray:
        """Аналитический или численный градиент без увеличения счётчика вызовов."""

    @abstractmethod
    def _hessian(self, x: np.ndarray) -> np.ndarray:
        """Аналитический или численный Гессиан без увеличения счётчика вызовов."""