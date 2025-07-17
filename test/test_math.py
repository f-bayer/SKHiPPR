import numpy as np
import pytest

from skhippr.math import finite_differences


def my_function(x=np.array([0, 1]), a=1, b=2):
    return np.array([*x, a, b])


def my_derivative(x, a, b, variable):
    I = np.eye(len(x) + 2)
    match variable:
        case "x":
            return I[:, : len(x)]
        case "a":
            return I[:, len(x)]
        case "b":
            return I[:, len(x) + 1]


@pytest.mark.parametrize("variable", ["x", "a", "b"])
def test_finite_differences(variable):
    parameters = {"a": 1, "b": 3, "x": [1, 2, 3]}
    finite_diff = finite_differences(my_function, parameters, variable, 1e-5)
    true_diff = my_derivative(variable=variable, **parameters)
    assert np.allclose(
        finite_diff, true_diff, 1e-4, 1e-4
    ), f"derivative w.r.t {variable} does not match!"
