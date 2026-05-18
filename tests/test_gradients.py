from __future__ import annotations

import numpy as np

from src.functions.builtin_2d import get_all_builtin_functions


def numerical_gradient(function, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    gradient = np.zeros_like(x, dtype=float)

    for i in range(x.size):
        step = np.zeros_like(x, dtype=float)
        step[i] = h

        f_plus = function.value(x + step)
        f_minus = function.value(x - step)

        gradient[i] = (f_plus - f_minus) / (2 * h)

    return gradient


def numerical_hessian(function, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    n = x.size
    hessian = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            step_i = np.zeros_like(x, dtype=float)
            step_j = np.zeros_like(x, dtype=float)

            step_i[i] = h
            step_j[j] = h

            f_pp = function.value(x + step_i + step_j)
            f_pm = function.value(x + step_i - step_j)
            f_mp = function.value(x - step_i + step_j)
            f_mm = function.value(x - step_i - step_j)

            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h**2)

    return hessian


def check_builtin_derivatives() -> None:
    functions = get_all_builtin_functions()

    test_points = {
        "Himmelblau": np.array([1.0, -1.0]),
        "Rosenbrock": np.array([-1.0, 1.5]),
        "Booth": np.array([2.0, -1.0]),
        "Beale": np.array([1.0, 0.5]),
        "Matyas": np.array([3.0, -2.0]),
        "ThreeHumpCamel": np.array([1.0, -1.0]),
        "Sphere": np.array([2.0, -3.0]),
        "Rastrigin": np.array([0.7, -1.2]),
    }

    for name, function in functions.items():
        x = test_points[name]

        analytical_gradient = function.gradient(x)
        numerical_grad = numerical_gradient(function, x)

        analytical_hessian = function.hessian(x)
        numerical_hess = numerical_hessian(function, x)

        gradient_ok = np.allclose(
            analytical_gradient,
            numerical_grad,
            atol=1e-4,
            rtol=1e-4,
        )

        hessian_ok = np.allclose(
            analytical_hessian,
            numerical_hess,
            atol=1e-3,
            rtol=1e-3,
        )

        print(f"{name}")
        print(f"  gradient ok: {gradient_ok}")
        print(f"  hessian ok:  {hessian_ok}")

        if not gradient_ok:
            print(f"  analytical gradient: {analytical_gradient}")
            print(f"  numerical gradient:  {numerical_grad}")

        if not hessian_ok:
            print(f"  analytical hessian:\n{analytical_hessian}")
            print(f"  numerical hessian:\n{numerical_hess}")

        print()


if __name__ == "__main__":
    check_builtin_derivatives()