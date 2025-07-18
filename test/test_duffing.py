import pytest
import numpy as np

from skhippr.systems.nonautonomous import Duffing


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
    Test the duffing function with mismatched dimensions for t and x.
    """
    t = np.linspace(0, 10, 100)  # Array of time points
    x = np.random.rand(3, len(t))  # 3xL array (incorrect dimension)

    try:
        # Call the duffing function
        f, derivatives = duffing(t, x, **params_duffing[1])
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

    try:
        # Call the duffing function
        f, derivatives = duffing(t, x, **params_duffing[1])
        assert (
            False
        ), "Expected ValueError for mismatched dimensions, but no error was raised."
    except ValueError:
        pass  # Expected behavior


if __name__ == "__main__":
    pytest.main([__file__])
