import numpy as np
import pytest

from skhippr.systems.AbstractSystems import EquationSystem


def my_function(x=np.array([0, 1]), a=1, b=2):
    x = np.atleast_2d(x)
    if x.shape[0] == 1 and x.shape[1] > 1:
        x = x.T

    return np.vstack(
        (x, a * np.ones((1, *x.shape[1:])), 2 * b * np.ones((1, *x.shape[1:])))
    )


def my_derivative(variable, x, a, b):
    I = np.eye(len(x) + 2)
    I[-1, -1] = 2  # b is multiplied by two

    I = I[:, :, np.newaxis] * np.ones((1, 1, *x.shape[1:]))

    match variable:
        case "x":
            return np.squeeze(I[:, : x.shape[0], ...])
        case "a":
            return np.squeeze(I[:, x.shape[0], ...])
        case "b":
            return np.squeeze(I[:, x.shape[0] + 1, ...])
        case _:
            raise NotImplementedError


@pytest.fixture(params=[1, 10])
def equation_system(request):
    n_samples = request.param
    x = np.squeeze(np.random.rand(3, n_samples))
    return EquationSystem(
        residual_function=my_function,
        closed_form_derivative=my_derivative,
        x=x,
        a=0.5,
        b=3,
    )


@pytest.mark.parametrize("variable", ["x", "a", "b"])
def test_finite_differences(equation_system: EquationSystem, variable):
    finite_diff = equation_system.finite_difference_derivative(variable, 1e-5)
    true_diff = equation_system.closed_form_derivative(variable=variable)

    assert np.allclose(
        finite_diff, true_diff, 1e-4, 1e-4
    ), f"derivative w.r.t {variable} does not match!"
