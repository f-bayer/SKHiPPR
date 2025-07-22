import numpy as np
import pytest


def test_f(ode_setting_vectorized):
    """
    Test the ODE function when x is either a 1-D or 2-D array.
    """
    _, ode = ode_setting_vectorized
    x = ode.x

    f = ode.dynamics()
    assert f.shape == x.shape, f"Expected f to have shape {x.shape}, but got {f.shape}"


def test_mismatched_ndof(ode_setting_vectorized):
    """
    Test the duffing function with mismatched n_dof.
    """

    params, ode = ode_setting_vectorized
    x_old = ode.x
    assert x_old.shape[0] == 2
    x = np.random.rand(*(3, *x_old.shape[1:]))  # 3xL array (incorrect dimension)
    ode.x = x  # no error detected yet

    with pytest.raises(ValueError):
        f = ode.dynamics()

    with pytest.raises(ValueError):
        df_dx = ode.derivative("x", update=True)
        pass

    for variable in params:
        with pytest.raises(ValueError):
            df_dvar = ode.derivative(variable, update=True)
            pass

    # Revert the ode object to correct setting
    ode.x = x_old


def test_derivatives(ode_setting_vectorized):
    """Verify that all derivatives match finite difference derivative"""
    params, ode = ode_setting_vectorized
    x = ode.x
    f = ode.residual(update=True)

    # Check derivative w.r.t. x
    df_dx_shape_expected = (2, *x.shape)
    df_dx = ode.derivative("x", update=True)
    assert (
        df_dx.shape == df_dx_shape_expected
    ), f"Expected df_dx to have shape {df_dx_shape_expected}, but got {df_dx.shape}"

    df_dx_fd = ode.finite_difference_derivative("x", h_step=1e-6)
    assert np.allclose(
        df_dx_fd, df_dx, 1e-3, 1e-3
    ), f"df_dx does not match FD derivative with max error {np.max(np.abs(df_dx_fd - df_dx))}"

    # Check derivatives w.r.t scalar parameters
    shape_expected = (x.shape[0], 1, *x.shape[1:])
    for variable in params:

        try:
            df_dvar = ode.closed_form_derivative(variable)
        except NotImplementedError:
            # derivative w.r.t this variable is not implemented
            continue

        assert (
            df_dvar.shape == shape_expected
        ), f"Expected df_dmu to have shape {shape_expected}, but got {df_dvar.shape}"

        df_dvar_fd = ode.finite_difference_derivative(variable, h_step=1e-5)
        assert np.allclose(
            df_dvar_fd, df_dvar, 1e-3, 1e-3
        ), f"{variable} derivative does not match FD derivative with max error {np.max(np.abs(df_dvar_fd - df_dvar))}"

    # Check that derivative w.r.t a non-parameter fails with an AttributeError and not a NotImplementedError
    with pytest.raises(AttributeError):
        df_dvar = ode.derivative("no_param")


if __name__ == "__main__":
    pytest.main([__file__])
