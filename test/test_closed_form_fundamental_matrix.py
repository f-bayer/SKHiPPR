import pytest
import numpy as np
from skhippr.cycles.shooting import ShootingBVP
from skhippr.odes.ltp import HillLTI, SmoothedMeissner


@pytest.mark.parametrize("t_over_period", np.arange(0, 3))
@pytest.mark.parametrize("d", [0.0, 0.1])
@pytest.mark.parametrize("gamma_sq", [-2.0, 0.0, 2.0])
def test_lti(solver, gamma_sq, d, t_over_period):
    ode = HillLTI(t=0, x=np.array([0.0, 0.0]), damping=d, b=gamma_sq, omega=np.sqrt(2))
    _test_closed_form_fundamat(solver, ode, t_over_period)


@pytest.mark.parametrize("t_over_period", np.arange(0, 10) / 8)
@pytest.mark.parametrize("d", [0.0, 0.1])
@pytest.mark.parametrize("a", [-2.0, 0.0, 2.0])
@pytest.mark.parametrize("b", [-2.0, 0.0, 2.0])
def test_meissner(solver, a, b, d, t_over_period):
    ode = SmoothedMeissner(
        t=0, x=np.array([0.0, 0.0]), smoothing=0, a=a, b=b, omega=np.sqrt(2), damping=d
    )
    _test_closed_form_fundamat(solver, ode, t_over_period)


def _test_closed_form_fundamat(solver, ode, t_over_period):

    T = 2 * np.pi / ode.omega

    t = T * t_over_period

    # Set up "trivial" shooting problem which is solved by initial guess [0,0]
    shoot = ShootingBVP(ode, T, atol=1e-8, rtol=1e-8)
    solver.solve_equation(shoot, "x")

    _, _, Phis_t = shoot.integrate_with_fundamental_matrix(x_0=ode.x, t=ode.t + t)
    Phi_t_ref = Phis_t[..., -1]
    Phi_t_cmp = ode.fundamental_matrix(ode.t + t, ode.t)
    print(Phi_t_ref)
    print(Phi_t_cmp)
    assert np.allclose(Phi_t_ref, Phi_t_cmp, atol=5e-5, rtol=5e-5)


if __name__ == "__main__":
    pytest.main([__file__])
