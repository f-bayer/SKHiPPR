import pytest
import numpy as np
from skhippr.problems.shooting import ShootingProblem
from skhippr.systems.ltp import (
    lti,
    lti_fundamental_matrix,
    meissner,
    meissner_fundamental_matrix,
)


@pytest.mark.parametrize("t_over_period", np.arange(0, 3))
@pytest.mark.parametrize("d", [0.0, 0.1])
@pytest.mark.parametrize("gamma_sq", [-2.0, 0.0, 2.0])
def test_lti(gamma_sq, d, t_over_period):
    parameters = {"gamma_sq": gamma_sq, "d": d}
    _test_closed_form_fundamat(lti, lti_fundamental_matrix, t_over_period, parameters)


@pytest.mark.parametrize("t_over_period", np.arange(0, 10) / 8)
@pytest.mark.parametrize("d", [0.0, 0.1])
@pytest.mark.parametrize("a", [-2.0, 0.0, 2.0])
@pytest.mark.parametrize("b", [-2.0, 0.0, 2.0])
def test_meissner(a, b, d, t_over_period):
    omega = np.sqrt(2)
    parameters = {"a": a, "b": b, "d": d, "omega": omega}
    _test_closed_form_fundamat(
        meissner, meissner_fundamental_matrix, t_over_period, parameters
    )


def _test_closed_form_fundamat(f_ode, f_fundamat, t_over_period, params):

    t_0 = 0
    if "omega" in params:
        omega = params["omega"]
    else:
        omega = np.sqrt(2)

    T = 2 * np.pi / omega
    y0 = np.array([0.0, 0.0])

    t = T * t_over_period
    assert t_0 <= t

    # Set up "trivial" shooting problem which is solved by initial guess [0,0]
    prblm = ShootingProblem(
        f=f_ode,
        x0=y0,
        T=T,
        autonomous=False,
        variable="y",
        tolerance=1e-10,
        parameters=params,
        verbose=True,
        kwargs_odesolver={"rtol": 1e-10, "atol": 1e-10},
        max_iterations=1,
    )
    prblm.solve()
    assert prblm.converged

    _, _, Phis_t = prblm.integrate_with_fundamental_matrix(x0=y0, t=t)
    Phi_t_ref = Phis_t[..., -1]
    Phi_t_cmp = f_fundamat(t, t_0, **params)
    print(Phi_t_ref)
    print(Phi_t_cmp)
    assert np.allclose(Phi_t_ref, Phi_t_cmp, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
