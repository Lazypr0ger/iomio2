from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.config import ensure_output_dirs
from src.functions.base import ObjectiveFunction
from src.functions.builtin_2d import get_all_builtin_functions, get_variant_3_functions
from src.functions.polynomial_10d import get_variant_3_polynomial
from src.methods.bfgs import bfgs
from src.methods.conjugate_gradient import conjugate_gradient_fletcher_reeves
from src.methods.newton import modified_newton
from src.models import OptimizationParams, OptimizationResult
from src.reporting.export import export_all_tables
from src.visualization.contours import plot_all_2d_contours
from src.visualization.convergence import plot_all_convergence_graphs
from src.visualization.polynomial_sections import plot_polynomial_sections


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


def get_default_params() -> OptimizationParams:
    """Параметры методов по умолчанию."""

    return OptimizationParams(
        epsilon=1e-3,
        max_iterations=1000,
        initial_step=1.0,
        armijo_c1=1e-4,
        backtracking_factor=0.5,
        min_step=1e-12,
        cg_restart_frequency=None,
        newton_regularization=1e-6,
    )


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


def print_result(result: OptimizationResult) -> None:
    """Печать результата одного метода."""

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


def run_variant_3() -> list[OptimizationResult]:
    """Запуск всех расчётов для варианта 3."""

    ensure_output_dirs()

    params = get_default_params()

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
            print_result(result)

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


def read_float(prompt: str, default: float) -> float:
    """Ввод числа с возможностью оставить значение по умолчанию."""

    raw_value = input(f"{prompt} [{default}]: ").strip()

    if not raw_value:
        return default

    return float(raw_value.replace(",", "."))


def read_int(prompt: str, default: int) -> int:
    """Ввод целого числа с возможностью оставить значение по умолчанию."""

    raw_value = input(f"{prompt} [{default}]: ").strip()

    if not raw_value:
        return default

    return int(raw_value)


def read_optional_int(prompt: str, default: int | None) -> int | None:
    """Ввод целого числа или None."""

    default_text = "None" if default is None else str(default)
    raw_value = input(f"{prompt} [{default_text}]: ").strip()

    if not raw_value:
        return default

    if raw_value.lower() in {"none", "нет", "no", "-"}:
        return None

    return int(raw_value)


def read_vector(prompt: str, default: np.ndarray) -> np.ndarray:
    """Ввод вектора через пробел или запятую."""

    default_text = np.array2string(default, precision=4)
    raw_value = input(f"{prompt} {default_text}: ").strip()

    if not raw_value:
        return default.copy()

    cleaned = raw_value.replace(",", " ")
    values = [float(item) for item in cleaned.split()]

    vector = np.array(values, dtype=float)

    if vector.shape != default.shape:
        raise ValueError(
            f"Ожидался вектор размера {default.size}, получен размер {vector.size}."
        )

    return vector


def configure_params(default_params: OptimizationParams) -> OptimizationParams:
    """Изменение гиперпараметров перед запуском."""

    print("=" * 80)
    print("Настройка гиперпараметров")
    print("Нажмите Enter, чтобы оставить значение по умолчанию.")

    return OptimizationParams(
        epsilon=read_float("epsilon", default_params.epsilon),
        max_iterations=read_int("max_iterations", default_params.max_iterations),
        initial_step=read_float("initial_step", default_params.initial_step),
        armijo_c1=read_float("armijo_c1", default_params.armijo_c1),
        backtracking_factor=read_float(
            "backtracking_factor",
            default_params.backtracking_factor,
        ),
        min_step=read_float("min_step", default_params.min_step),
        cg_restart_frequency=read_optional_int(
            "cg_restart_frequency",
            default_params.cg_restart_frequency,
        ),
        newton_regularization=read_float(
            "newton_regularization",
            default_params.newton_regularization,
        ),
    )


def get_functions_for_menu() -> list[ObjectiveFunction]:
    """Список функций для интерактивного меню."""

    builtin_functions = list(get_all_builtin_functions().values())
    return [*builtin_functions, get_variant_3_polynomial()]


def choose_function(functions: list[ObjectiveFunction]) -> ObjectiveFunction:
    """Выбор функции из меню."""

    print("=" * 80)
    print("Выбор функции")

    for index, function in enumerate(functions, start=1):
        print(f"{index}. {function.name}, x0 = {function.default_x0}")

    selected_index = read_int("Номер функции", 1)

    if selected_index < 1 or selected_index > len(functions):
        raise ValueError("Некорректный номер функции.")

    return functions[selected_index - 1]


def run_interactive_single_function() -> list[OptimizationResult]:
    """Интерактивный запуск одной функции всеми методами."""

    ensure_output_dirs()

    functions = get_functions_for_menu()
    function = choose_function(functions)

    x0 = read_vector("Начальная точка x0", function.default_x0)
    function.default_x0 = x0

    params = configure_params(get_default_params())

    print("=" * 80)
    print(f"Функция: {function.name}")
    print(f"x0 = {function.default_x0}")

    results = run_methods_for_function(function, params)

    for result in results:
        print_result(result)

    if function.dimension == 2:
        contour_paths = plot_all_2d_contours([function], results)
        print("=" * 80)
        print("Контурный график построен")
        for path in contour_paths:
            print(path)

    convergence_paths = plot_all_convergence_graphs(
        functions=[function],
        results=results,
        epsilon=params.epsilon,
    )

    print("=" * 80)
    print("Графики сходимости построены")
    for path in convergence_paths:
        print(path)

    if function.dimension == 10:
        section_paths = plot_polynomial_sections(
            functions=[function],
            results=results,
        )

        print("=" * 80)
        print("2D-сечение полинома построено")
        for path in section_paths:
            print(path)

    export_all_tables(results)

    return results


def run_menu() -> None:
    """Главное меню программы."""

    while True:
        print("=" * 80)
        print("Лабораторная работа №2. Нелинейная оптимизация")
        print("1. Запустить вариант 3 полностью")
        print("2. Запустить одну функцию с настройками")
        print("0. Выход")

        choice = input("Выберите пункт: ").strip()

        if choice == "1":
            run_variant_3()
        elif choice == "2":
            run_interactive_single_function()
        elif choice == "0":
            print("Завершение работы.")
            break
        else:
            print("Некорректный пункт меню.")