from __future__ import annotations

import numpy as np

from src.functions.base import ObjectiveFunction
from src.methods.line_search import armijo_backtracking
from src.methods.utils import gradient_norm
from src.models import IterationRecord, OptimizationParams, OptimizationResult


def conjugate_gradient_fletcher_reeves(
    function: ObjectiveFunction,
    x0: np.ndarray,
    params: OptimizationParams,
) -> OptimizationResult:
    """
    Метод сопряжённых градиентов Флетчера–Ривза.

    Направление:
        p_0 = -g_0
        p_k = -g_k + beta_k * p_{k-1}

    Коэффициент:
        beta_k = ||g_k||^2 / ||g_{k-1}||^2

    Рестарт:
        каждые n шагов, где n — размерность задачи.
    """

    function.reset_counter()

    x = np.array(x0, dtype=float)
    function.validate_x(x)

    restart_frequency = (
        function.dimension
        if params.cg_restart_frequency is None
        else params.cg_restart_frequency
    )

    history: list[IterationRecord] = []

    gradient = function.gradient(x)
    grad_norm = gradient_norm(gradient)
    direction = -gradient

    history.append(
        IterationRecord(
            k=0,
            x=x.copy(),
            f_value=function.value(x),
            grad_norm=grad_norm,
            step_size=None,
            extra={"beta": None, "restart": True},
        )
    )

    converged = grad_norm <= params.epsilon
    message = "Начальная точка удовлетворяет критерию остановки."

    for k in range(1, params.max_iterations + 1):
        if grad_norm <= params.epsilon:
            converged = True
            message = "Достигнут критерий остановки по норме градиента."
            break

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

        denominator = float(np.dot(gradient, gradient))

        if denominator <= 1e-30:
            beta = 0.0
        else:
            beta = float(np.dot(gradient_next, gradient_next) / denominator)

        need_restart = k % restart_frequency == 0

        if need_restart:
            beta = 0.0

        direction_next = -gradient_next + beta * direction

        history.append(
            IterationRecord(
                k=k,
                x=x_next.copy(),
                f_value=function.value(x_next),
                grad_norm=grad_norm_next,
                step_size=step_size,
                extra={"beta": beta, "restart": need_restart},
            )
        )

        x = x_next
        gradient = gradient_next
        grad_norm = grad_norm_next
        direction = direction_next

    else:
        converged = False
        message = "Достигнут максимум итераций без выполнения критерия остановки."

    final_f = function.value(x)

    return OptimizationResult(
        method_name="CG Fletcher-Reeves",
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