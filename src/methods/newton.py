from __future__ import annotations

import numpy as np

from src.functions.base import ObjectiveFunction
from src.methods.line_search import armijo_backtracking
from src.methods.utils import gradient_norm, make_positive_definite, safe_solve
from src.models import IterationRecord, OptimizationParams, OptimizationResult


def modified_newton(
    function: ObjectiveFunction,
    x0: np.ndarray,
    params: OptimizationParams,
) -> OptimizationResult:
    """
    Модифицированный метод Ньютона.

    На каждой итерации:
        H_k = ∇²f(x_k)
        H_k_modified = H_k + μI, если H_k не положительно определён
        p_k = -H_k_modified^{-1} ∇f(x_k)

    Длина шага выбирается бэктрекингом Армихо.
    """

    function.reset_counter()

    x = np.array(x0, dtype=float)
    function.validate_x(x)

    history: list[IterationRecord] = []

    gradient = function.gradient(x)
    grad_norm = gradient_norm(gradient)

    history.append(
        IterationRecord(
            k=0,
            x=x.copy(),
            f_value=function.value(x),
            grad_norm=grad_norm,
            step_size=None,
            extra={"regularization": None},
        )
    )

    converged = grad_norm <= params.epsilon
    message = "Начальная точка удовлетворяет критерию остановки."

    for k in range(1, params.max_iterations + 1):
        if grad_norm <= params.epsilon:
            converged = True
            message = "Достигнут критерий остановки по норме градиента."
            break

        hessian = function.hessian(x)
        modified_hessian = make_positive_definite(
            matrix=hessian,
            regularization=params.newton_regularization,
        )

        direction = -safe_solve(modified_hessian, gradient)

        if float(np.dot(direction, gradient)) >= 0:
            direction = -gradient

        step_size = armijo_backtracking(
            function=function,
            x=x,
            direction=direction,
            gradient=gradient,
            params=params,
        )

        x_next = x + step_size * direction
        gradient_next = function.gradient(x_next)
        grad_norm_next = gradient_norm(gradient_next)

        regularization_norm = float(np.linalg.norm(modified_hessian - hessian))

        history.append(
            IterationRecord(
                k=k,
                x=x_next.copy(),
                f_value=function.value(x_next),
                grad_norm=grad_norm_next,
                step_size=step_size,
                extra={"regularization": regularization_norm},
            )
        )

        x = x_next
        gradient = gradient_next
        grad_norm = grad_norm_next

    else:
        converged = False
        message = "Достигнут максимум итераций без выполнения критерия остановки."

    final_f = function.value(x)

    return OptimizationResult(
        method_name="Modified Newton",
        function_name=function.name,
        x_star=x.copy(),
        f_star=final_f,
        grad_norm=grad_norm,
        iterations=len(history) - 1,
        function_evaluations=function.counter.function_calls,
        gradient_evaluations=function.counter.gradient_calls,
        hessian_evaluations=function.counter.hessian_calls,
        history=history,
        converged=converged,
        message=message,
    )