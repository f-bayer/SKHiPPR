import numpy as np
import pytest

from skhippr.equations.odes.FirstOrderODE import Equation


def my_function(x=np.array([0, 1]), a=1, b=2):
    x = np.atleast_1d(x)
    if x.ndim == 1:
        x = x[:, np.newaxis]

    return np.squeeze(
        np.vstack(
            (x, a * np.ones((1, *x.shape[1:])), 2 * b * np.ones((1, *x.shape[1:])))
        )
    )


def my_derivative(variable, x, a, b):
    I = np.eye(len(x) + 2)
    I[-1, -1] = 2  # b is multiplied by two

    match variable:
        case "x":
            return I[:, : x.shape[0]]
        case "a":
            return I[:, [x.shape[0]]]
        case "b":
            return I[:, [x.shape[0] + 1]]
        case _:
            raise NotImplementedError


@pytest.fixture
def equation(request):
    x = np.random.rand(3)
    return Equation(
        residual_function=my_function,
        closed_form_derivative=my_derivative,
        x=x,
        a=0.5,
        b=3,
    )


@pytest.mark.parametrize("variable", ["x", "a", "b"])
def test_finite_differences(equation: Equation, variable):
    finite_diff = equation.finite_difference_derivative(variable, 1e-5)
    true_diff = equation.closed_form_derivative(variable=variable)

    assert np.allclose(
        finite_diff, true_diff, 1e-4, 1e-4
    ), f"derivative w.r.t {variable} does not match!"
