from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.functions.base import ObjectiveFunction


@dataclass(frozen=True, slots=True)
class CrossTerm:
    """Перекрёстный член c * x_i * x_j."""

    i: int
    j: int
    coefficient: float


@dataclass(frozen=True, slots=True)
class HighPowerTerm:
    """Высокостепенной член alpha * x_k^power."""

    index: int
    coefficient: float
    power: int


class Polynomial10DFunction(ObjectiveFunction):
    """Полином 10-й степени от 10 переменных."""

    def __init__(
        self,
        name: str,
        diagonal: np.ndarray,
        cross_terms: list[CrossTerm],
        linear: np.ndarray,
        high_power_term: HighPowerTerm,
        default_x0: np.ndarray,
    ) -> None:
        if diagonal.shape != (10,):
            raise ValueError("Массив diagonal должен иметь размер 10.")

        if linear.shape != (10,):
            raise ValueError("Массив linear должен иметь размер 10.")

        if default_x0.shape != (10,):
            raise ValueError("Начальная точка default_x0 должна иметь размер 10.")

        self.diagonal = diagonal.astype(float)
        self.cross_terms = cross_terms
        self.linear = linear.astype(float)
        self.high_power_term = high_power_term

        super().__init__(
            name=name,
            dimension=10,
            default_x0=default_x0.astype(float),
            known_minimum=None,
            known_minimum_value=None,
        )

    def _value(self, x: np.ndarray) -> float:
        quadratic_sum = float(np.sum(self.diagonal * x**2))

        cross_sum = 0.0
        for term in self.cross_terms:
            cross_sum += term.coefficient * x[term.i] * x[term.j]

        linear_sum = float(np.sum(self.linear * x))

        hp = self.high_power_term
        high_power_sum = hp.coefficient * x[hp.index] ** hp.power

        return quadratic_sum + cross_sum + linear_sum + high_power_sum

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        gradient = 2 * self.diagonal * x + self.linear

        for term in self.cross_terms:
            gradient[term.i] += term.coefficient * x[term.j]
            gradient[term.j] += term.coefficient * x[term.i]

        hp = self.high_power_term
        gradient[hp.index] += hp.coefficient * hp.power * x[hp.index] ** (hp.power - 1)

        return gradient

    def _hessian(self, x: np.ndarray) -> np.ndarray:
        hessian = np.diag(2 * self.diagonal)

        for term in self.cross_terms:
            hessian[term.i, term.j] += term.coefficient
            hessian[term.j, term.i] += term.coefficient

        hp = self.high_power_term
        hessian[hp.index, hp.index] += (
            hp.coefficient
            * hp.power
            * (hp.power - 1)
            * x[hp.index] ** (hp.power - 2)
        )

        return hessian


def get_variant_3_polynomial() -> Polynomial10DFunction:
    """
    Полином 10D для варианта 3.

    В таблице варианта указано:
    diag = [9.8, 5.1, 7.5, ...],
    5 cross,
    0.043 * x3^6,
    x0 = (-1.5, -2.4, 2.5, ...).

    Так как оставшиеся коэффициенты не раскрыты в таблице, они заданы явно.
    Это допустимо по условию: не заданные коэффициенты можно приравнять к 0
    или взять случайным образом. Здесь используется воспроизводимый вариант.
    Индексы в коде начинаются с 0, поэтому x3 соответствует index=2.
    """

    diagonal = np.array(
        [
            9.8,
            5.1,
            7.5,
            4.2,
            6.4,
            3.8,
            8.1,
            2.9,
            5.7,
            4.6,
        ],
        dtype=float,
    )

    cross_terms = [
        CrossTerm(i=0, j=1, coefficient=-1.3),
        CrossTerm(i=1, j=2, coefficient=0.8),
        CrossTerm(i=3, j=4, coefficient=-1.1),
        CrossTerm(i=5, j=7, coefficient=1.4),
        CrossTerm(i=8, j=9, coefficient=-0.9),
    ]

    linear = np.array(
        [
            1.2,
            -2.0,
            0.7,
            0.0,
            -1.5,
            2.1,
            0.0,
            -0.8,
            1.4,
            0.0,
        ],
        dtype=float,
    )

    high_power_term = HighPowerTerm(
        index=2,
        coefficient=0.043,
        power=6,
    )

    default_x0 = np.array(
        [
            -1.5,
            -2.4,
            2.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    return Polynomial10DFunction(
        name="Polynomial10DVariant3",
        diagonal=diagonal,
        cross_terms=cross_terms,
        linear=linear,
        high_power_term=high_power_term,
        default_x0=default_x0,
    )