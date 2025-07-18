import pytest
import numpy as np

from skhippr.systems.nonautonomous import Duffing
from skhippr.math import finite_differences


def test_duffing_array_inputs(params_duffing):
    """
    Test the duffing function when t is an array and x is a 2xL array.
    """
    t = np.linspace(0, 10, 100)  # Array of time points
    x = np.random.rand(2, len(t))  # 2xL array

    duff = Duffing(t=t, x=x, **params_duffing[1])

    f = duff.dynamics()

    # Check dimensions of the output
    assert f.shape == x.shape, f"Expected f to have shape {x.shape}, but got {f.shape}"

    df_dx = duff.derivative("x")
    assert df_dx.shape == (
        2,
        2,
        len(t),
    ), f"Expected df_dx to have shape {(2, 2, len(t))}, but got {f.shape}"
    for key in ["F", "omega", "alpha"]:
        df_dkey = duff.derivative(key)
        assert (
            df_dkey.shape == x.shape
        ), f"Expected df_d{key} to have shape {x.shape}, but got {df_dkey.shape}"


def test_duffing_scalar_inputs(params_duffing):
    """
    Test the duffing function when t is a float and x is a 1-dimensional array of length 2.
    """
    t = 1.0  # Scalar time point
    x = np.random.rand(2)  # 1-dimensional array of length 2

    duff = Duffing(t=t, x=x, **params_duffing[1])
    f = duff.dynamics()

    assert f.shape == x.shape, f"Expected f to have shape {x.shape}, but got {f.shape}"

    df_dx = duff.derivative("x")
    assert df_dx.shape == (
        2,
        2,
    ), f"Expected df_dx to have shape {(2, 2)}, but got {f.shape}"

    for key in ["F", "omega", "alpha"]:
        df_dkey = duff.derivative(key)
        assert (
            df_dkey.shape == x.shape
        ), f"Expected df_d{key} to have shape {x.shape}, but got {df_dkey.shape}"


def test_duffing_mismatched_ndof(params_duffing):
    """
    Test the duffing function with mismatched n_dof.
    """
    t = np.linspace(0, 10, 100)  # Array of time points
    x = np.random.rand(3, len(t))  # 3xL array (incorrect dimension)

    duff = Duffing(t=t, x=x, **params_duffing[1])  # no error detected yet

    try:
        # Call the duffing function
        f = duff.dynamics()
        assert (
            False
        ), "Expected ValueError for mismatched dimensions, but no error was raised."
    except ValueError:
        pass  # Expected behavior


def test_duffing_mismatched_dimensions(params_duffing):
    """
    Test the duffing function with mismatched dimensions for t and x.
    """
    t = np.linspace(0, 10, 100)  # Array of time points
    x = np.random.rand(2, len(t) + 1)  # 2x(L+1) array (incorrect dimension)

    duff = Duffing(t=t, x=x, **params_duffing[1])  # no error detected yet

    try:
        # Call the duffing function
        f = duff.dynamics()
        assert (
            False
        ), "Expected ValueError for mismatched dimensions, but no error was raised."
    except ValueError:
        pass  # Expected behavior


@pytest.mark.parametrize(
    "variable", ["x", "omega", "F", "alpha", "beta", "delta", "bad_var"]
)
def test_duffing_derivatives(params_duffing, variable):
    """Verify that the returned derivative matches finite difference derivative"""
    t = np.linspace(0, 10, 100)  # Array of time points
    x = np.random.rand(2, len(t))  # 2xL array
    duff = Duffing(t, x, **params_duffing[1])

    kwargs = params_duffing[1].copy()
    kwargs["x"] = x
    kwargs["t"] = t

    if variable != "x" and variable not in params_duffing[1]:
        with pytest.raises(AttributeError):
            deriv_duff = duff.derivative(variable, **kwargs)
        return

    deriv_duff = duff.derivative(variable, **kwargs)
    deriv_finite_diff = finite_differences(duff.dynamics, kwargs, variable, 1e-5)

    assert np.allclose(
        deriv_finite_diff, deriv_duff, 1e-3, 1e-3
    ), f"{variable} derivative does not match FD derivative with error {np.linalg.norm(deriv_finite_diff - deriv_duff)}"


if __name__ == "__main__":
    pytest.main([__file__])
