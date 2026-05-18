from __future__ import annotations
from src.visualization.contours import plot_all_2d_contours
from src.visualization.convergence import plot_all_convergence_graphs
from src.visualization.polynomial_sections import plot_polynomial_sections
from src.reporting.export import export_all_tables
from collections.abc import Callable

import numpy as np

from src.config import ensure_output_dirs
from src.functions.base import ObjectiveFunction
from src.functions.builtin_2d import get_variant_3_functions
from src.functions.polynomial_10d import get_variant_3_polynomial
from src.methods.bfgs import bfgs
from src.methods.conjugate_gradient import conjugate_gradient_fletcher_reeves
from src.methods.newton import modified_newton
from src.models import OptimizationParams, OptimizationResult


OptimizationMethod = Callable[
    [ObjectiveFunction, np.ndarray, OptimizationParams],
    OptimizationResult,
]


def get_required_methods() -> list[OptimizationMethod]:
    """Три обязательных метода лабораторной работы."""

    return [
        conjugate_gradient_fletcher_reeves,
        modified_newton,
        bfgs,
    ]


def run_methods_for_function(
    function: ObjectiveFunction,
    params: OptimizationParams,
) -> list[OptimizationResult]:
    """Запуск всех обязательных методов для одной функции."""

    results: list[OptimizationResult] = []

    for method in get_required_methods():
        result = method(function, function.default_x0, params)
        results.append(result)

    return results


def run_variant_3() -> list[OptimizationResult]:
    """Запуск всех расчётов для варианта 3."""

    ensure_output_dirs()

    params = OptimizationParams(
        epsilon=1e-3,
        max_iterations=1000,
        initial_step=1.0,
        armijo_c1=1e-4,
        backtracking_factor=0.5,
        min_step=1e-12,
        cg_restart_frequency=None,
        newton_regularization=1e-6,
    )

    variant_2d_functions = get_variant_3_functions()

    functions: list[ObjectiveFunction] = [
    *variant_2d_functions,
    get_variant_3_polynomial(),
    ]

    all_results: list[OptimizationResult] = []

    for function in functions:
        print("=" * 80)
        print(f"Функция: {function.name}")
        print(f"x0 = {function.default_x0}")

        results = run_methods_for_function(function, params)
        all_results.extend(results)

        for result in results:
            print("-" * 80)
            print(f"Метод: {result.method_name}")
            print(f"Сошёлся: {result.converged}")
            print(f"Сообщение: {result.message}")
            print(f"Итерации: {result.iterations}")
            print(f"x* = {np.array2string(result.x_star, precision=6)}")
            print(f"f(x*) = {result.f_star:.10g}")
            print(f"||grad|| = {result.grad_norm:.10g}")
            print(f"Вычисления f: {result.function_evaluations}")
            print(f"Вычисления grad: {result.gradient_evaluations}")
            print(f"Вычисления Hessian: {result.hessian_evaluations}")
    contour_paths = plot_all_2d_contours(variant_2d_functions, all_results)

    print("=" * 80)
    print("Контурные графики построены")
    for path in contour_paths:
        print(path)

    convergence_paths = plot_all_convergence_graphs(
        functions=functions,
        results=all_results,
        epsilon=params.epsilon,
    )

    print("=" * 80)
    print("Графики сходимости построены")
    for path in convergence_paths:
        print(path)

    section_paths = plot_polynomial_sections(
        functions=functions,
        results=all_results,
    )

    print("=" * 80)
    print("2D-сечения полинома построены")
    for path in section_paths:
        print(path)

    export_all_tables(all_results)

    return all_results