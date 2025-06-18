import numpy as np
import matplotlib.pyplot as plt

from skhippr.Fourier import Fourier
from skhippr.systems.nonautonomous import duffing

from skhippr.problems.shooting import ShootingProblem
from skhippr.problems.HBM import HBMProblem
from skhippr.problems.continuation import pseudo_arclength_continuator


def shooting_frc():

    params = {
        "alpha": 1,
        "beta": 0.04,
        "F": 1,
        "delta": 0.1,
        "omega": 3,
    }
    T0 = 2 * np.pi / params["omega"]

    ode_kwargs = {"rtol": 1e-7, "atol": 1e-7}

    x0 = np.array([1.0, 0.0])

    prb = ShootingProblem(
        f=duffing,
        x0=x0,
        T=T0,
        autonomous=False,
        variable="x",
        verbose=False,
        kwargs_odesolver=ode_kwargs,
        parameters=params,
    )
    prb.solve()
    assert prb.converged
    print(prb)

    omega_range = (0.1, 3)
    print("HBM reference")
    frc_ref = frc_HBM(params, prb)
    print("HBM reference done.")
    frc = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    tangent_old = 0
    for branch_point in pseudo_arclength_continuator(
        initial_problem=prb,
        stepsize=0.05,
        initial_direction=1,
        num_steps=1000,
        key_param="T",
        value_param=T0,
        stepsize_range=(0.01, 0.1),
        verbose=True,
    ):
        frc.append(branch_point)
        branch_point.determine_tangent()
        x_tng = branch_point.x + 0.1 * branch_point.tangent

        Ts = (branch_point.T, x_tng[-1])
        amp_1 = (branch_point.x[0], x_tng[0])
        amp_2 = (branch_point.x[1], x_tng[1])
        ax.plot(Ts, amp_1, amp_2, "gray")
        ax.plot(branch_point.x0[-1], branch_point.x0[0], branch_point.x0[1], "r+")

        if len(frc) > 1:
            assert (
                abs(np.inner(branch_point.x - branch_point.x0, frc[-2].tangent)) < 1e-3
            )

        if branch_point.omega < omega_range[0] or branch_point.omega > omega_range[1]:
            break

    print("Continuation terminated.")

    Ts = [branch_point.T for branch_point in frc]
    omegas = [branch_point.omega for branch_point in frc]
    x1s = [branch_point.x[0] for branch_point in frc]
    x2s = [branch_point.x[1] for branch_point in frc]
    amplitudes = [np.max(branch_point.x_time()[0, :]) for branch_point in frc]

    Ts_ref = [2 * np.pi / branch_point.omega for branch_point in frc_ref]
    omegas_ref = [branch_point.omega for branch_point in frc_ref]
    x1s_ref = [branch_point.x_time()[0, 0] for branch_point in frc_ref]
    x2s_ref = [branch_point.x_time()[1, 0] for branch_point in frc_ref]
    amplitudes_ref = [np.max(branch_point.x_time()[0, :]) for branch_point in frc_ref]

    ax.plot(Ts_ref, x1s_ref, x2s_ref, label="HBM")
    ax.plot(Ts, x1s, x2s, "--", label="shooting")
    ax.set_xlabel("T")
    ax.set_ylabel("x_1(0)")
    ax.set_zlabel("x_2[0]")

    plt.figure()
    plt.plot(omegas_ref, amplitudes_ref)
    plt.plot(omegas, amplitudes, "--")

    print(frc[0].T)
    # print(frc[100].T)
    x1 = frc[0].x_time()
    # x2 = frc[100].x_time(T=frc[100].T)

    # np.allclose(x1, x2)
    print(np.max(x1[0, :]))
    # print(np.max(x2[0, :]))
    pass


def frc_HBM(params, sol_init):
    """Reference FRC determined using HBM"""
    N_HBM = 25
    L_DFT = 150
    fourier = Fourier(N_HBM, L_DFT, 2, True)

    X0 = fourier.DFT(sol_init.x_time(t_eval=fourier.time_samples(omega=sol_init.omega)))
    HBM_init = HBMProblem(
        f=duffing,
        initial_guess=X0,
        omega=sol_init.omega,
        fourier=fourier,
        parameters_f=params,
    )
    HBM_init.solve()
    assert HBM_init.converged

    frc = list(
        pseudo_arclength_continuator(
            initial_problem=HBM_init,
            stepsize=0.4,
            stepsize_range=(0.05, 0.6),
            key_param="omega",
            value_param=HBM_init.omega,
            verbose=True,
            num_steps=80,
            initial_direction=-1,
        )
    )

    return frc


if __name__ == "__main__":
    shooting_frc()
    plt.show()
