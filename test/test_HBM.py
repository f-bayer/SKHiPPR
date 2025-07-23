"""Test HBM using the Duffing oscillator and the Vanderpol oscillator. See also: Matlab workshop, Task x.x"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from skhippr.problems.HBM import HBMEquation  # , HBMProblem_autonomous
from skhippr.problems.shooting import ShootingBVP
from skhippr.Fourier import Fourier
from skhippr.systems.nonautonomous import Duffing
from skhippr.problems.newton import NewtonSolver

# from skhippr.systems.autonomous import vanderpol


@pytest.mark.parametrize("period_k", [1, 5])
def test_HBM_duffing(solver, fourier, period_k, visualize=False):

    if visualize:
        print("Duffing oscillator")

    x_0 = np.array([1.0, 0.0])

    match period_k:
        case 1:
            ode = Duffing(t=0, x=x_0, alpha=1, beta=3, F=1, delta=1, omega=1.3)
        case 5:
            ode = Duffing(t=0, x=x_0, alpha=-1, beta=1, F=0.37, delta=0.3, omega=1.2)
        case _:
            raise ValueError(f"Unknown value '{period_k}' for period-k solution")

    T = 2 * np.pi / ode.omega

    # Reference: Find periodic solution using shooting method
    shooting = ShootingBVP(ode, T=T, period_k=period_k, atol=1e-7, rtol=1e-7)

    solver.solve_equation(shooting, "x")

    """ Compute periodic solution using HBM """
    t_eval = fourier.time_samples(ode.omega, periods=period_k)[::period_k]
    x_shoot = shooting.x_time(t_eval=t_eval)
    X0 = fourier.DFT(x_shoot + np.random.rand(*x_shoot.shape) * 1e-1)

    hbm = HBMEquation(
        ode=ode,
        omega=ode.omega,
        fourier=fourier,
        initial_guess=X0,
        period_k=period_k,
        stability_method=None,
    )

    if visualize:
        print(hbm)

    solver.solve_equation(hbm, "X")

    if visualize:
        print(hbm)

    # Assert that the HBM solution is indeed a solution

    x_hbm = hbm.x_time()
    if visualize:
        plt.figure()
        plt.plot(x_shoot[0, :], x_shoot[1, :], label="ODE")
        plt.plot(np.real(x_hbm[0, :]), np.real(x_hbm[1, :]), "--", label="HBM")
        plt.legend()
        plt.title(
            f"{'real' if fourier.real_formulation else 'complex'} Duffing \n {hbm}"
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
    my_solver = NewtonSolver(verbose=True)
    my_fourier = Fourier(N_HBM=15, L_DFT=128, n_dof=2, real_formulation=True)
    for period_k in [1, 5]:
        test_HBM_duffing(my_solver, my_fourier, period_k, visualize=True)
    plt.show()
