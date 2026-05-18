from __future__ import annotations

import numpy as np


def gradient_norm(gradient: np.ndarray) -> float:
    """Евклидова норма градиента."""

    return float(np.linalg.norm(gradient))


def is_converged(gradient: np.ndarray, epsilon: float) -> bool:
    """Проверка критерия остановки по норме градиента."""

    return gradient_norm(gradient) <= epsilon


def make_positive_definite(
    matrix: np.ndarray,
    regularization: float,
    max_attempts: int = 12,
) -> np.ndarray:
    """
    Модификация матрицы до положительно определённой.

    Используется в модифицированном методе Ньютона.
    Если Гессиан не положительно определён, к нему добавляется mu * I.
    """

    n = matrix.shape[0]
    identity = np.eye(n)
    mu = regularization

    for _ in range(max_attempts):
        candidate = matrix + mu * identity

        try:
            np.linalg.cholesky(candidate)
            return candidate
        except np.linalg.LinAlgError:
            mu *= 10.0

    return matrix + mu * identity


def safe_solve(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Решение системы matrix * x = vector.

    Если матрица плохо обусловлена или вырождена, используется псевдообратная.
    """

    try:
        return np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix) @ vector