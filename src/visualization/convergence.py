from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import PLOTS_DIR
from src.functions.base import ObjectiveFunction
from src.models import OptimizationResult


def _safe_log10(values: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    """Безопасный log10 для неотрицательных величин."""

    return np.log10(np.maximum(values, eps))


def plot_gradient_norm_convergence(
    function: ObjectiveFunction,
    results: list[OptimizationResult],
    epsilon: float,
    output_path: Path | None = None,
) -> Path:
    """График log10(||grad f(x_k)||) по итерациям."""

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = PLOTS_DIR / f"convergence_grad_norm_{function.name}.png"

    plt.figure(figsize=(10, 6))

    for result in results:
        grad_norms = np.array(
            [record.grad_norm for record in result.history],
            dtype=float,
        )

        iterations = np.arange(grad_norms.size)

        plt.plot(
            iterations,
            _safe_log10(grad_norms),
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"{result.method_name}, k={result.iterations}",
        )

    plt.axhline(
        y=np.log10(epsilon),
        linestyle="--",
        linewidth=1.2,
        label=f"ε = {epsilon:g}",
    )

    plt.title(f"{function.name}: сходимость по норме градиента")
    plt.xlabel("Номер итерации k")
    plt.ylabel("log10(||∇f(xᵏ)||)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def plot_function_decrease(
    function: ObjectiveFunction,
    results: list[OptimizationResult],
    output_path: Path | None = None,
) -> Path:
    """График log10(|f(x_k) - f*|) по итерациям."""

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = PLOTS_DIR / f"decrease_f_value_{function.name}.png"

    if function.known_minimum_value is not None:
        reference_value = function.known_minimum_value
    else:
        reference_value = min(result.f_star for result in results)

    plt.figure(figsize=(10, 6))

    for result in results:
        f_values = np.array(
            [record.f_value for record in result.history],
            dtype=float,
        )

        differences = np.abs(f_values - reference_value)
        iterations = np.arange(f_values.size)

        plt.plot(
            iterations,
            _safe_log10(differences),
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"{result.method_name}, k={result.iterations}",
        )

    plt.title(f"{function.name}: убывание значения функции")
    plt.xlabel("Номер итерации k")
    plt.ylabel("log10(|f(xᵏ) - f*|)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def plot_all_convergence_graphs(
    functions: list[ObjectiveFunction],
    results: list[OptimizationResult],
    epsilon: float,
) -> list[Path]:
    """Построение графиков сходимости и убывания для всех функций."""

    paths: list[Path] = []

    for function in functions:
        function_results = [
            result
            for result in results
            if result.function_name == function.name
        ]

        if not function_results:
            continue

        paths.append(
            plot_gradient_norm_convergence(
                function=function,
                results=function_results,
                epsilon=epsilon,
            )
        )

        paths.append(
            plot_function_decrease(
                function=function,
                results=function_results,
            )
        )

    return paths