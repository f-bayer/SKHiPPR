import numpy as np
import matplotlib.pyplot as plt

from skhippr.Fourier import Fourier
from skhippr.systems.nonautonomous import Duffing

from skhippr.problems.shooting import ShootingSystem
from skhippr.problems.HBM import HBMSystem
from skhippr.problems.continuation import pseudo_arclength_continuator
from skhippr.problems.newton import NewtonSolver


def shooting_frc(solver=None, visualize=True):

    if solver is None:
        solver = NewtonSolver(verbose=True)

    x0 = np.array([1.0, 0.0])
    ode = Duffing(t=0, x=x0, omega=3, alpha=1, beta=0.04, F=1, delta=0.1)
    T0 = 2 * np.pi / ode.omega

    initial_system = ShootingSystem(ode, T=T0, period_k=1, atol=1e-7, rtol=1e-7)

    solver.solve(initial_system)
    assert initial_system.solved

    omega_range = (0.1, 3)

    frc = []
    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = None

    # print("HBM reference")
    # t_init = np.linspace(0, 2 * np.pi / ode.omega, 150, endpoint=False)
    # x_init = initial_system.equations[0].x_time(t_init)
    # frc_ref = frc_HBM(solver, ode, x_init=x_init, ax=ax)
    # print("HBM reference done.")

    # solver.verbose = False
    for branch_point in pseudo_arclength_continuator(
        initial_system=initial_system,
        solver=solver,
        stepsize=0.05,
        stepsize_range=(0.01, 0.1),
        initial_direction=1,
        continuation_parameter="T",
        verbose=True,
        num_steps=100,
    ):
        frc.append(branch_point)
        branch_point.determine_tangent()

        if visualize:
            x_tng = branch_point.vector_of_unknowns + 0.1 * branch_point.tangent
            Ts = (np.squeeze(branch_point.T), x_tng[-1])
            amp_1 = (branch_point.x[0], x_tng[0])
            amp_2 = (branch_point.x[1], x_tng[1])
            ax.plot(Ts, amp_1, amp_2, "gray")
            ax.plot(Ts[0], amp_1[0], amp_2[0], ".", color="red")

            pass

        # ax.plot(
        #     solver.initial_guess[-1],
        #     solver.initial_guess[0],
        #     solver.initial_guess[1],
        #     "r+",
        # )

        # if len(frc) > 1:
        #     assert (
        #         abs(
        #             np.inner(
        #                 branch_point.x - branch_point.initial_guess, frc[-2].tangent
        #             )
        #         )
        #         < 1e-3
        #     )

        if (
            branch_point.equations[0].omega < omega_range[0]
            or branch_point.equations[0].omega > omega_range[1]
        ):
            break

    print("Continuation terminated.")

    return

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


def frc_HBM(solver, ode, x_init, ax=None):
    """Reference FRC determined using HBM"""
    N_HBM = 25
    L_DFT = x_init.shape[1]
    fourier = Fourier(N_HBM, L_DFT, 2, True)

    X0 = fourier.DFT(x_init)
    HBM_init = HBMSystem(ode, ode.omega, fourier, X0, 1, None)
    solver.solve(HBM_init)
    assert HBM_init.solved

    frc = list(
        pseudo_arclength_continuator(
            initial_system=HBM_init,
            solver=solver,
            stepsize=0.2,
            stepsize_range=(0.05, 0.6),
            continuation_parameter="omega",
            verbose=True,
            num_steps=50,
            initial_direction=-1,
        )
    )

    if ax is not None:
        for bp in frc:
            x = bp.equations[0].x_time()[:, 0]
            T = 2 * np.pi / bp.omega
            ax.plot(T, x[0], x[1], ".", color="green")
            pass

    return frc


if __name__ == "__main__":
    shooting_frc(solver=None, visualize=True)
    plt.show()
