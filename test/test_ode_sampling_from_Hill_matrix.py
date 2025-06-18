import pytest
import numpy as np
import matplotlib.pyplot as plt

from skhippr.systems.autonomous import vanderpol
from skhippr.problems.HBM import HBMProblem_autonomous
from skhippr.stability.SinglePass import SinglePassRK4


def f(t, x):
    return vanderpol(t, x, nu=0.5)


def test_ode_sampling(fourier):
    omega0 = 1

    ts_samp = fourier.time_samples(omega=omega0)

    x0_samples = np.vstack(
        (2 * np.cos(omega0 * ts_samp), -2 * omega0 * np.sin(omega0 * ts_samp))
    )
    X0 = fourier.DFT(x0_samples)

    prb = HBMProblem_autonomous(f=f, initial_guess=X0, omega=omega0, fourier=fourier)
    prb.solve()
    assert prb.converged
    x_samp = prb.x_time()
    assert np.max(np.abs(x_samp)) > 1e-1

    _, derivatives_ref = f(ts_samp, x_samp)
    J_ref = derivatives_ref["x"]
    _, derivatives = prb.residual_function()

    J_samp = derivatives["x_samp"]
    assert np.allclose(J_ref, J_samp)

    SP = SinglePassRK4(fourier)
    J_SP = prb.ode_samples()
    assert np.allclose(J_ref, J_SP)


if __name__ == "__main__":
    pytest.main([__file__])
