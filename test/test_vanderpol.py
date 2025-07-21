import numpy as np
import pytest
from skhippr.systems.autonomous import Vanderpol, Truss, BlockOnBelt

# TODO find errors here on Monday


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


def test_f(ode_setting):
    """
    Test the ODE function when x is either a 1-D or 2-D array.
    """
    _, ode = ode_setting
    x = ode.x

    f = ode.dynamics()
    assert f.shape == x.shape, f"Expected f to have shape {x.shape}, but got {f.shape}"


# TODO implement self.check_dimensions into subclasses
def test_mismatched_ndof(ode_setting):
    """
    Test the duffing function with mismatched n_dof.
    """

    params, ode = ode_setting
    x = ode.x
    x = np.random.rand(*(3, *x.shape[1:]))  # 3xL array (incorrect dimension)
    ode.x = x  # no error detected yet

    with pytest.raises(ValueError):
        f = ode.dynamics()

    with pytest.raises(ValueError):
        df_dx = ode.derivative("x")
        pass

    for variable in params:
        with pytest.raises(ValueError):
            df_dvar = ode.derivative(variable)
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

    df_dx_fd = ode.finite_difference_derivative("x", h_step=1e-6)
    assert np.allclose(
        df_dx_fd, df_dx, 1e-3, 1e-3
    ), f"df_dx does not match FD derivative with max error {np.max(np.abs(df_dx_fd - df_dx))}"

    # Check derivatives w.r.t scalar parameters
    for variable in params:
        df_dvar = ode.derivative(variable)
        assert (
            df_dvar.shape == x.shape
        ), f"Expected df_dmu to have shape {x.shape}, but got {df_dvar.shape}"

        df_dvar_fd = ode.finite_difference_derivative(variable, h_step=1e-5)
        assert np.allclose(
            df_dvar_fd, df_dvar, 1e-3, 1e-3
        ), f"{variable} derivative does not match FD derivative with max error {np.max(np.abs(df_dvar_fd - df_dvar))}"

    # Check that derivative w.r.t a non-parameter fails
    with pytest.raises(AttributeError):
        df_dvar = ode.derivative("no_param")


if __name__ == "__main__":
    pytest.main([__file__])
