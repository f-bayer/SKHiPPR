"""Test HBM using the Duffing oscillator and the Vanderpol oscillator. See also: Matlab workshop, Task x.x"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

# from skhippr.problems.shooting import shooting
from skhippr.problems.HBM import HBMProblem, HBMProblem_autonomous
from skhippr.problems.shooting import ShootingProblem
from skhippr.Fourier import Fourier
from skhippr.systems.nonautonomous import duffing
from skhippr.systems.autonomous import vanderpol


@pytest.mark.parametrize("period_k", [1, 5])
def test_HBM_duffing(fourier, params_duffing, period_k, visualize=False):

    if visualize:
        print("Duffing oscillator")

    params = params_duffing[period_k]
    omega = params["omega"]

    """ Compute periodic solution using shooting """
    # Find period-5 solution using shooting method
    ode_kwargs = {"atol": 1e-7, "rtol": 1e-7}
    sol_shoot = ShootingProblem(
        f=duffing,
        x0=[1.0, 0.0],
        T=2 * period_k * np.pi / omega,
        kwargs_odesolver=ode_kwargs,
        parameters=params,
        period_k=period_k,
    )
    sol_shoot.solve()
    assert sol_shoot.converged

    """ Compute periodic solution using HBM """
    t_eval = fourier.time_samples(omega, periods=period_k)[::period_k]
    x_shoot = sol_shoot.x_time(t_eval=t_eval)
    X0 = fourier.DFT(x_shoot + np.random.rand(*x_shoot.shape) * 1e-1)

    HBMsol = HBMProblem(
        f=duffing,
        initial_guess=X0,
        omega=omega,
        fourier=fourier,
        variable="x",
        stability_method=None,
        tolerance=1e-8,
        max_iterations=30,
        verbose=visualize,
        period_k=period_k,
        parameters_f=params,
    )

    if visualize:
        print(HBMsol)

    HBMsol.solve()

    if visualize:
        print(HBMsol)

    assert HBMsol.converged

    # Assert that the HBM solution is indeed a solution

    x_shoot = sol_shoot.x_time(t_eval=t_eval)
    x_hbm = HBMsol.x_time()
    if visualize:
        plt.figure()
        plt.plot(x_shoot[0, :], x_shoot[1, :], label="ODE")
        plt.plot(np.real(x_hbm[0, :]), np.real(x_hbm[1, :]), "--", label="HBM")
        plt.legend()
        plt.title(
            f"{'real' if fourier.real_formulation else 'complex'} Duffing \n {HBMsol}"
        )

    # assert np.allclose(x_hbm, x_shoot, atol=1e-1)


def test_HBM_aut(fourier, visualize=True):

    params = {"nu": 1}
    omega0 = 1.0

    print("Van der Pol oscillator")

    """ Compute periodic solution using HBM """
    t0 = fourier.time_samples(omega0)
    x0_hbm = np.vstack((np.cos(omega0 * t0), -omega0 * np.sin(omega0 * t0)))

    sol_HBM = HBMProblem_autonomous(
        f=vanderpol,
        initial_guess=x0_hbm,
        omega=omega0,
        fourier=fourier,
        verbose=True,
        parameters_f=params,
    )
    print(sol_HBM)
    sol_HBM.solve()
    print(sol_HBM)
    assert sol_HBM.converged
    x_p_hbm = sol_HBM.x_time()

    # Solve the shooting problem for reference
    sol_shoot = ShootingProblem(
        f=vanderpol,
        x0=np.real(x_p_hbm[:, 0]),
        T=2 * np.pi / np.real(sol_HBM.omega),
        autonomous=True,
        variable="x",
        verbose=visualize,
        kwargs_odesolver={"atol": 1e-7, "rtol": 1e-7},
        parameters=params,
    )
    x_p_shoot = sol_shoot.x_time(fourier.time_samples(np.real(sol_HBM.omega)))
    try:
        assert np.allclose(x_p_hbm, x_p_shoot, atol=1e-1, rtol=1e-1)
    except AssertionError as AE:
        if visualize:
            plt.figure()
            plt.plot(fourier.time_samples(sol_HBM.omega), x_p_hbm.T, label="HBM")
            plt.plot(
                fourier.time_samples(sol_HBM.omega), x_p_shoot.T, "--", label="Shooting"
            )
            plt.show()
            pass
        raise AE

    if visualize:
        print(f"Max imaginary part: {max(abs(np.imag(x_p_hbm[0,:])))}")
        plt.figure()
        plt.plot(x_p_shoot[0, :], x_p_shoot[1, :], label="ODE")
        plt.plot(x_p_hbm[0, :], x_p_hbm[1, :], "--", label="HBM")
        plt.legend()
        plt.title(
            f"{'real' if fourier.real_formulation else 'complex'} van der Pol \n {sol_HBM}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
