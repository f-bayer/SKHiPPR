import numpy as np
import pytest
from skhippr.systems.autonomous import Vanderpol, Truss, BlockOnBelt
from skhippr.math import finite_differences


@pytest.fixture(scope="module", params=["vanderpol", "truss", "blockonbelt"])
def ode_setting(x, request):

    match request.param:
        case "vanderpol":
            params = {"nu": 1.5}
            return params, Vanderpol(x=x, **params)
        case "truss":
            params = {"k": 100.0, "c": 0.5, "F": 1.0, "a": 0.1, "l_0": 1.0, "m": 1.0}
            return params, Truss(x=x, **params)
        case "blockonbelt":
            params = {"epsilon": 0.1, "k": 1, "m": 1, "Fs": 0.1, "vdr": 2, "delta": 0.5}
            return params, BlockOnBelt(x=x, **params)


@pytest.fixture(scope="module", params=[1, 100])
def x(request):
    n_samples = request.param
    if n_samples == 1:
        x = np.random.rand(2)
    else:
        x = np.random.rand(2, n_samples)

    return x


@pytest.mark.parametrize("n_samples", [1, 100])
def test_f(ode_setting, n_samples):
    """
    Test the ODE function when x is either a 1-D or 2-D array.
    """
    _, ode = ode_setting
    x = ode.x
    df_dx_shape_expected = (2, *x.shape)

    f = ode.dynamics()
    assert f.shape == x.shape, f"Expected f to have shape {x.shape}, but got {f.shape}"


def test_mismatched_ndof(ode_setting):
    """
    Test the duffing function with mismatched n_dof.
    """

    params, ode = ode_setting
    x = np.random.rand(3, len(t))  # 3xL array (incorrect dimension)
    ode.x = x  # no error detected yet

    with pytest.raises(ValueError):
        f = ode.dynamics()

    with pytest.raises(ValueError):
        df_dx = ode.derivative("x")
        pass

    with pytest.raises(ValueError):
        df_dx = ode.derivative("nu")
        pass


def test_derivatives(ode_setting):
    """Verify that all derivatives match finite difference derivative"""
    params, ode = ode_setting
    x = ode.x

    # Check derivative w.r.t. x
    df_dx_shape_expected = (2, *x.shape)
    df_dx = ode.derivative("x")
    assert (
        df_dx.shape == df_dx_shape_expected
    ), f"Expected df_dx to have shape {df_dx_shape_expected}, but got {df_dx.shape}"

    params["x"] = x
    df_dx_fd = finite_differences(ode.dynamics, params, "x", 1e-5)
    assert np.allclose(
        df_dx_fd, df_dx, 1e-3, 1e-3
    ), f"df_dx does not match FD derivative with max error {np.max(np.abs(df_dx_fd - df_dx))}"

    del params["x"]
    # Check derivatives w.r.t scalar parameters
    for variable in params:
        df_dvar = ode.derivative(variable)
        assert (
            df_dvar.shape == x.shape
        ), f"Expected df_dmu to have shape {x.shape}, but got {df_dvar.shape}"

        df_dvar_fd = finite_differences(ode.dynamics, params, variable, 1e-5)
        assert np.allclose(
            df_dvar_fd, df_dvar, 1e-3, 1e-3
        ), f"{variable} derivative does not match FD derivative with max error {np.max(np.abs(df_dvar_fd - df_dvar))}"

    # Check that derivative w.r.t a non-parameter fails
    with pytest.raises(AttributeError):
        df_dvar = ode.derivative("no_param")


if __name__ == "__main__":
    pytest.main([__file__])
