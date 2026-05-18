from __future__ import annotations

import numpy as np

from src.functions.base import ObjectiveFunction
from src.methods.line_search import armijo_backtracking
from src.methods.utils import gradient_norm
from src.models import IterationRecord, OptimizationParams, OptimizationResult


def bfgs(
    function: ObjectiveFunction,
    x0: np.ndarray,
    params: OptimizationParams,
) -> OptimizationResult:
    """
    Метод BFGS.

    В реализации хранится приближение обратного Гессиана H_inv.
    Направление:
        p_k = -H_inv_k * g_k

    Обновление:
        s_k = x_{k+1} - x_k
        y_k = g_{k+1} - g_k
        rho_k = 1 / (y_k^T s_k)

        H_inv_{k+1} =
            (I - rho*s*y^T) H_inv_k (I - rho*y*s^T) + rho*s*s^T
    """

    function.reset_counter()

    x = np.array(x0, dtype=float)
    function.validate_x(x)

    n = function.dimension
    identity = np.eye(n)
    inverse_hessian_approx = np.eye(n)

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
            extra={"rho": None, "update": None},
        )
    )

    converged = grad_norm <= params.epsilon
    message = "Начальная точка удовлетворяет критерию остановки."

    for k in range(1, params.max_iterations + 1):
        if grad_norm <= params.epsilon:
            converged = True
            message = "Достигнут критерий остановки по норме градиента."
            break

        direction = -inverse_hessian_approx @ gradient

        if float(np.dot(direction, gradient)) >= 0:
            direction = -gradient
            inverse_hessian_approx = np.eye(n)

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

        s = x_next - x
        y = gradient_next - gradient
        ys = float(np.dot(y, s))

        update_status = "skipped"
        rho = None

        if ys > 1e-12:
            rho = 1.0 / ys

            left = identity - rho * np.outer(s, y)
            right = identity - rho * np.outer(y, s)

            inverse_hessian_approx = (
                left @ inverse_hessian_approx @ right
                + rho * np.outer(s, s)
            )

            update_status = "updated"
        else:
            inverse_hessian_approx = np.eye(n)

        history.append(
            IterationRecord(
                k=k,
                x=x_next.copy(),
                f_value=function.value(x_next),
                grad_norm=grad_norm_next,
                step_size=step_size,
                extra={"rho": rho, "update": update_status},
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
        method_name="BFGS",
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