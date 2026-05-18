from __future__ import annotations

import numpy as np

from src.functions.base import ObjectiveFunction


class HimmelblauFunction(ObjectiveFunction):
    """Функция Химмельблау."""

    def __init__(self, default_x0: np.ndarray | None = None) -> None:
        super().__init__(
            name="Himmelblau",
            dimension=2,
            default_x0=np.array([0.0, 0.0]) if default_x0 is None else default_x0,
            known_minimum=np.array([3.0, 2.0]),
            known_minimum_value=0.0,
            known_minima=[
                np.array([3.0, 2.0]),
                np.array([-2.805118, 3.131312]),
                np.array([-3.779310, -3.283186]),
                np.array([3.584428, -1.848126]),
            ],
        )

    def _value(self, x: np.ndarray) -> float:
        x1, x2 = x
        a = x1**2 + x2 - 11
        b = x1 + x2**2 - 7
        return a**2 + b**2

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x
        a = x1**2 + x2 - 11
        b = x1 + x2**2 - 7

        df_dx1 = 4 * x1 * a + 2 * b
        df_dx2 = 2 * a + 4 * x2 * b

        return np.array([df_dx1, df_dx2])

    def _hessian(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x

        h11 = 12 * x1**2 + 4 * x2 - 42
        h22 = 4 * x1 + 12 * x2**2 - 26
        h12 = 4 * x1 + 4 * x2

        return np.array(
            [
                [h11, h12],
                [h12, h22],
            ]
        )


class RosenbrockFunction(ObjectiveFunction):
    """Функция Розенброка."""

    def __init__(self, default_x0: np.ndarray | None = None) -> None:
        super().__init__(
            name="Rosenbrock",
            dimension=2,
            default_x0=np.array([-1.2, 1.0]) if default_x0 is None else default_x0,
            known_minimum=np.array([1.0, 1.0]),
            known_minimum_value=0.0,
        )

    def _value(self, x: np.ndarray) -> float:
        x1, x2 = x
        return (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x

        df_dx1 = 2 * (x1 - 1) - 400 * x1 * (x2 - x1**2)
        df_dx2 = 200 * (x2 - x1**2)

        return np.array([df_dx1, df_dx2])

    def _hessian(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x

        h11 = 2 - 400 * x2 + 1200 * x1**2
        h12 = -400 * x1
        h22 = 200

        return np.array(
            [
                [h11, h12],
                [h12, h22],
            ]
        )


class BoothFunction(ObjectiveFunction):
    """Функция Бута."""

    def __init__(self, default_x0: np.ndarray | None = None) -> None:
        super().__init__(
            name="Booth",
            dimension=2,
            default_x0=np.array([0.0, 0.0]) if default_x0 is None else default_x0,
            known_minimum=np.array([1.0, 3.0]),
            known_minimum_value=0.0,
        )

    def _value(self, x: np.ndarray) -> float:
        x1, x2 = x
        return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x

        df_dx1 = 10 * x1 + 8 * x2 - 34
        df_dx2 = 8 * x1 + 10 * x2 - 38

        return np.array([df_dx1, df_dx2])

    def _hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [10.0, 8.0],
                [8.0, 10.0],
            ]
        )


class BealeFunction(ObjectiveFunction):
    """Функция Била."""

    def __init__(self, default_x0: np.ndarray | None = None) -> None:
        super().__init__(
            name="Beale",
            dimension=2,
            default_x0=np.array([0.0, 0.0]) if default_x0 is None else default_x0,
            known_minimum=np.array([3.0, 0.5]),
            known_minimum_value=0.0,
        )

    def _value(self, x: np.ndarray) -> float:
        x1, x2 = x

        a = 1.5 - x1 + x1 * x2
        b = 2.25 - x1 + x1 * x2**2
        c = 2.625 - x1 + x1 * x2**3

        return a**2 + b**2 + c**2

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x

        powers = np.array([1.0, 2.0, 3.0])
        constants = np.array([1.5, 2.25, 2.625])
        y_powers = np.array([x2, x2**2, x2**3])

        residuals = constants - x1 + x1 * y_powers

        df_dx1 = 2 * np.sum(residuals * (-1 + y_powers))
        df_dx2 = 2 * np.sum(residuals * x1 * powers * x2 ** (powers - 1))

        return np.array([df_dx1, df_dx2])

    def _hessian(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x

        h11 = 0.0
        h12 = 0.0
        h22 = 0.0

        constants = [1.5, 2.25, 2.625]

        for i, constant in enumerate(constants, start=1):
            residual = constant - x1 + x1 * x2**i

            residual_x1 = -1 + x2**i
            residual_x2 = x1 * i * x2 ** (i - 1)

            residual_x1x2 = i * x2 ** (i - 1)
            residual_x2x2 = x1 * i * (i - 1) * x2 ** (i - 2) if i >= 2 else 0.0

            h11 += 2 * residual_x1 * residual_x1
            h12 += 2 * (residual_x1 * residual_x2 + residual * residual_x1x2)
            h22 += 2 * (residual_x2 * residual_x2 + residual * residual_x2x2)

        return np.array(
            [
                [h11, h12],
                [h12, h22],
            ]
        )


class MatyasFunction(ObjectiveFunction):
    """Функция Матьяса."""

    def __init__(self, default_x0: np.ndarray | None = None) -> None:
        super().__init__(
            name="Matyas",
            dimension=2,
            default_x0=np.array([5.0, 5.0]) if default_x0 is None else default_x0,
            known_minimum=np.array([0.0, 0.0]),
            known_minimum_value=0.0,
        )

    def _value(self, x: np.ndarray) -> float:
        x1, x2 = x
        return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x

        df_dx1 = 0.52 * x1 - 0.48 * x2
        df_dx2 = 0.52 * x2 - 0.48 * x1

        return np.array([df_dx1, df_dx2])

    def _hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [0.52, -0.48],
                [-0.48, 0.52],
            ]
        )


class ThreeHumpCamelFunction(ObjectiveFunction):
    """Трёхгорбый верблюд."""

    def __init__(self, default_x0: np.ndarray | None = None) -> None:
        super().__init__(
            name="ThreeHumpCamel",
            dimension=2,
            default_x0=np.array([2.0, 2.0]) if default_x0 is None else default_x0,
            known_minimum=np.array([0.0, 0.0]),
            known_minimum_value=0.0,
        )

    def _value(self, x: np.ndarray) -> float:
        x1, x2 = x
        return 2 * x1**2 - 1.05 * x1**4 + (x1**6) / 6 + x1 * x2 + x2**2

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x

        df_dx1 = 4 * x1 - 4.2 * x1**3 + x1**5 + x2
        df_dx2 = x1 + 2 * x2

        return np.array([df_dx1, df_dx2])

    def _hessian(self, x: np.ndarray) -> np.ndarray:
        x1, _ = x

        h11 = 4 - 12.6 * x1**2 + 5 * x1**4
        h12 = 1.0
        h22 = 2.0

        return np.array(
            [
                [h11, h12],
                [h12, h22],
            ]
        )


class SphereFunction(ObjectiveFunction):
    """Сфера n=2."""

    def __init__(self, default_x0: np.ndarray | None = None) -> None:
        super().__init__(
            name="Sphere",
            dimension=2,
            default_x0=np.array([5.0, 5.0]) if default_x0 is None else default_x0,
            known_minimum=np.array([0.0, 0.0]),
            known_minimum_value=0.0,
        )

    def _value(self, x: np.ndarray) -> float:
        return float(np.sum(x**2))

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        return 2 * x

    def _hessian(self, x: np.ndarray) -> np.ndarray:
        return 2 * np.eye(2)


class RastriginFunction(ObjectiveFunction):
    """Функция Растригина n=2."""

    def __init__(self, default_x0: np.ndarray | None = None) -> None:
        super().__init__(
            name="Rastrigin",
            dimension=2,
            default_x0=np.array([2.5, 2.5]) if default_x0 is None else default_x0,
            known_minimum=np.array([0.0, 0.0]),
            known_minimum_value=0.0,
        )

    def _value(self, x: np.ndarray) -> float:
        x1, x2 = x
        return (
            20
            + x1**2
            + x2**2
            - 10 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))
        )

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x

        df_dx1 = 2 * x1 + 20 * np.pi * np.sin(2 * np.pi * x1)
        df_dx2 = 2 * x2 + 20 * np.pi * np.sin(2 * np.pi * x2)

        return np.array([df_dx1, df_dx2])

    def _hessian(self, x: np.ndarray) -> np.ndarray:
        x1, x2 = x

        h11 = 2 + 40 * np.pi**2 * np.cos(2 * np.pi * x1)
        h22 = 2 + 40 * np.pi**2 * np.cos(2 * np.pi * x2)

        return np.array(
            [
                [h11, 0.0],
                [0.0, h22],
            ]
        )


def get_all_builtin_functions() -> dict[str, ObjectiveFunction]:
    """Все встроенные тестовые функции."""

    functions = [
        HimmelblauFunction(),
        RosenbrockFunction(),
        BoothFunction(),
        BealeFunction(),
        MatyasFunction(),
        ThreeHumpCamelFunction(),
        SphereFunction(),
        RastriginFunction(),
    ]

    return {function.name: function for function in functions}


def get_variant_3_functions() -> list[ObjectiveFunction]:
    """Четыре 2D-функции индивидуального варианта 3."""

    return [
        BoothFunction(default_x0=np.array([-5.0, -5.0])),
        SphereFunction(default_x0=np.array([-3.0, -3.0])),
        RosenbrockFunction(default_x0=np.array([-2.0, 2.0])),
        HimmelblauFunction(default_x0=np.array([-1.0, 0.0])),
    ]