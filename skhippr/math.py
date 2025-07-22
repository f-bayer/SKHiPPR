import numpy as np
from skhippr.systems.AbstractSystems import AbstractEquation


def finite_difference_derivative(
    equation: AbstractEquation, variable, h_step=1e-4
) -> np.ndarray:

    x_orig = getattr(equation, variable)
    x = np.atleast_2d(x_orig)
    n = x.shape[0]
    if n == 1 and x.shape[1] > 1:
        x = x.T
        n = x.shape[0]

    f = equation.residual(update=False)
    delta = h_step * np.eye(n)
    derivative = np.zeros((f.shape[0], n, *f.shape[1:]))

    for k in range(n):
        setattr(equation, variable, np.squeeze(x + delta[:, [k]]))
        derivative[:, k, ...] = (equation.residual_function() - f) / h_step

    setattr(equation, variable, x_orig)
    return derivative
