import numpy as np
import pytest
from skhippr.systems.autonomous import Vanderpol
from skhippr.math import finite_differences


@pytest.mark.parametrize("n_samples", [1, 100])
def test_vanderpol_inputs(n_samples):
    """
    Test the vanderpol function when t is an array and x is a 2xL array.
    """

    if n_samples == 1:
        t = 1.0
        x = np.random.rand(2)
        df_dx_shape_expected = (2, 2)
    else:
        t = np.linspace(1, 10, n_samples)
        x = np.random.rand(2, len(t))
        df_dx_shape_expected = (2, 2, n_samples)

    nu = 1.5

    vdp = Vanderpol(t=t, x=x, nu=nu)
    f = vdp.dynamics()

    assert f.shape == x.shape, f"Expected f to have shape {x.shape}, but got {f.shape}"

    df_dx = vdp.derivative("x")

    assert (
        df_dx.shape == df_dx_shape_expected
    ), f"Expected df_dx to have shape {df_dx_shape_expected}, but got {df_dx.shape}"

    df_dnu = vdp.derivative("nu")
    assert (
        df_dnu.shape == x.shape
    ), f"Expected df_dmu to have shape {x.shape}, but got {df_dnu.shape}"


def test_vdp_mismatched_ndof():
    """
    Test the duffing function with mismatched n_dof.
    """
    t = np.linspace(0, 10, 100)  # Array of time points
    x = np.random.rand(3, len(t))  # 3xL array (incorrect dimension)
    nu = 1.5

    vdp = Vanderpol(t=t, x=x, nu=nu)  # no error detected yet

    with pytest.raises(ValueError):
        f = vdp.dynamics()

    with pytest.raises(ValueError):
        df_dx = vdp.derivative("x")
        pass

    with pytest.raises(ValueError):
        df_dx = vdp.derivative("nu")
        pass


@pytest.mark.parametrize("variable", ["x", "nu", "t"])
def test_duffing_derivatives(params_duffing, variable):
    """Verify that the returned derivative matches finite difference derivative"""
    t = np.linspace(0, 10, 100)  # Array of time points
    x = np.random.rand(2, len(t))  # 2xL array
    nu = 1.5

    vdp = Vanderpol(x=x, nu=nu, t=t)

    kwargs = {"nu": nu, "x": x, "t": t}

    if variable not in ("x", "nu"):
        with pytest.raises(AttributeError):
            deriv_vdp = vdp.derivative(variable, **kwargs)
        return

    deriv_vdp = vdp.derivative(variable, **kwargs)
    deriv_finite_diff = finite_differences(vdp.dynamics, kwargs, variable, 1e-5)

    assert np.allclose(
        deriv_finite_diff, deriv_vdp, 1e-3, 1e-3
    ), f"{variable} derivative does not match FD derivative with error {np.linalg.norm(deriv_finite_diff - deriv_vdp)}"


if __name__ == "__main__":
    pytest.main([__file__])
