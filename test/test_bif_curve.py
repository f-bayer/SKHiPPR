"""Test HBM using the Duffing oscillator. See also: Matlab workshop, Task x.x"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pytest

from skhippr.Fourier import Fourier
from skhippr.problems.HBM import HBMProblem, HBMProblem_autonomous
from skhippr.problems.shooting import ShootingProblem
from skhippr.systems.nonautonomous import duffing
from skhippr.systems.autonomous import vanderpol
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic
from skhippr.problems.continuation import pseudo_arclength_continuator


def bif_curve_nonaut(sparse=False, real_formulation=False):

    alpha = 1
    beta = 0.2
    F_0 = 0.05
    delta = 0.1
    omega = 0.8

    N_HBM = 25
    L_DFT = 300

    fourier = Fourier(
        N_HBM=N_HBM,
        L_DFT=L_DFT,
        n_dof=2,
        sparse_formulation=sparse,
        real_formulation=real_formulation,
    )

    print("Duffing oscillator")

    def f(t, x, omega=omega, alpha=alpha, beta=beta, F=F_0, delta=delta):
        return duffing(t, x, omega, alpha, beta, F, delta)

    ts = fourier.time_samples(omega)
    x0_samples = np.vstack((np.cos(omega * ts), -omega * np.sin(omega * ts)))
    X0 = fourier.DFT(x0_samples)

    stability_method = KoopmanHillSubharmonic(fourier=fourier)

    initial_problem_HBM = HBMProblem(
        f=f,
        initial_guess=X0,
        omega=omega,
        fourier=fourier,
        variable="x",
        stability_method=stability_method,
        verbose=False,
        parameters_f={"omega": omega, "F": F_0},
    )

    kwargs_ode = {"atol": 1e-5, "rtol": 1e-5}

    initial_problem_shoot = ShootingProblem(
        f=f,
        x0=x0_samples[:, 0],
        T=2 * np.pi / omega,
        autonomous=False,
        variable="x",
        verbose=False,
        parameters={"omega": omega, "F": F_0},
        kwargs_odesolver=kwargs_ode,
    )

    F_range = (0, 20)

    fig_3d = plt.figure()
    ax = fig_3d.add_subplot(111, projection="3d")
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.set_zlabel("F")

    fig_stability = plt.figure()

    for initial_problem in (initial_problem_HBM, initial_problem_shoot):

        xs_time = []
        amplitudes = []
        stable = []
        Fs = []

        for branch_point in pseudo_arclength_continuator(
            initial_problem=initial_problem,
            stepsize=0.1,
            stepsize_range=(0.001, 0.15),
            initial_direction=1,
            num_steps=1000,
            key_param="F",
            value_param=F_0,
            verbose=True,
        ):
            xs_time.append(branch_point.x_time())
            amplitudes.append(np.max(xs_time[-1][0, :]))
            Fs.append(branch_point.F)
            stable.append(branch_point.stable)

            branch_point.determine_tangent()
            x_tng = branch_point.x + 0.2 * branch_point.tangent
            x0 = np.append(xs_time[-1][:, 0], branch_point.F)
            try:
                # HBM case
                x0_tng = np.append(fourier.inv_DFT(x_tng)[:, 0], x_tng[-1])
            except:
                # Shooting case
                x0_tng = x_tng
                ax.plot(*zip(x0, x0_tng), "gray")

            if branch_point.F < F_range[0] or branch_point.F > F_range[1]:
                break

        x0s = np.array([x_time[:, 0] for x_time in xs_time])
        amplitudes = np.array(amplitudes)
        stable = np.array(stable)
        unstable = np.invert(stable)
        Fs = np.array(Fs)
        F_stable = Fs.copy()
        F_stable[unstable] = np.nan
        F_unstable = Fs.copy()
        F_unstable[stable] = np.nan

        ax.plot(x0s[:, 0], x0s[:, 1], Fs, linewidth=1)

        plt.figure(fig_stability)

        plt.plot(F_stable, amplitudes, label="stable")
        plt.plot(F_unstable, amplitudes, "--", label="unstable")
        # plt.title(f"Amplitude of first DOF -- {solver}")
    plt.legend()


def bif_curve_aut(sparse=False, real_formulation=False):
    nu_0 = 0.1
    omega_0 = 1
    N_HBM = 45
    L_DFT = 1000

    print("Van der Pol oscillator")

    fourier = Fourier(
        N_HBM=N_HBM,
        L_DFT=L_DFT,
        n_dof=2,
        sparse_formulation=sparse,
        real_formulation=real_formulation,
    )

    ts = fourier.time_samples(omega_0)
    x0_samples = np.vstack(
        (2 * np.cos(omega_0 * ts), -2 * omega_0 * np.sin(omega_0 * ts))
    )
    X0 = fourier.DFT(x0_samples)

    stability_method = KoopmanHillSubharmonic(
        fourier=fourier, autonomous=True, tol=1e-4
    )

    initial_problem = HBMProblem_autonomous(
        f=vanderpol,
        initial_guess=X0,
        omega=omega_0,
        fourier=fourier,
        variable="x",
        stability_method=stability_method,
        verbose=False,
        parameters_f={"nu": nu_0},
    )

    initial_problem.solve()
    print(f"Initial problem convergence: {initial_problem.converged}")

    nu_range = (nu_0, 10)

    xs_time = []
    amplitudes = []
    nus = []
    stable = []
    omegas = []
    floquet_multipliers = []
    x0s = []

    for branch_point in pseudo_arclength_continuator(
        initial_problem=initial_problem,
        stepsize=0.1,
        stepsize_range=(0.001, 0.15),
        initial_direction=1,
        num_steps=1000,
        key_param="nu",
        value_param=nu_0,
        verbose=True,
    ):
        xs_time.append(branch_point.x_time())
        amplitudes.append(np.max(xs_time[-1][0, :]))
        nus.append(branch_point.nu)
        stable.append(branch_point.stable)
        omegas.append(branch_point.omega)
        floquet_multipliers.append(branch_point.eigenvalues)

        x0 = xs_time[-1][:, 0]

        if branch_point.nu < nu_range[0] or branch_point.nu > nu_range[1]:
            break

    amplitudes = np.array(amplitudes)
    stable = np.array(stable)
    unstable = np.invert(stable)
    nus = np.array(nus)
    nu_stable = nus.copy()
    nu_stable[unstable] = np.nan
    nu_unstable = nus.copy()
    nu_unstable[stable] = np.nan

    plt.plot(nu_stable, amplitudes, "r+-", label="stable")
    plt.plot(nu_unstable, amplitudes, "bx-", label="unstable")
    # plt.title(f"Amplitude of first DOF -- {solver}")
    plt.legend()
    plt.xlabel("\nu")
    plt.ylabel("|x_1|")

    plt.figure()
    plt.plot(nu_stable, omegas, "r", label="stable")
    plt.plot(nu_unstable, omegas, "b", label="unstable")
    plt.xlabel("\nu")
    plt.ylabel("\omega")

    fig_anim, axs = plt.subplots(nrows=1, ncols=2)
    # Create an animation
    (line2,) = axs[0].plot(xs_time[0][0, :], xs_time[0][1, :])
    axs[0].axis("equal")
    (plot_floquet_multiplier,) = axs[1].plot(
        np.real(floquet_multipliers[0]),
        np.imag(floquet_multipliers[0]),
        "x",
    )
    phis = np.linspace(0, 2 * np.pi)
    axs[1].plot(np.cos(phis), np.sin(phis), "k-")
    axs[1].axis("equal")

    def update(frame):
        line2.set_xdata(xs_time[frame][0, :])
        line2.set_ydata(xs_time[frame][1, :])
        plot_floquet_multiplier.set_xdata(np.real(floquet_multipliers[frame]))
        plot_floquet_multiplier.set_ydata(np.imag(floquet_multipliers[frame]))
        return (line2, plot_floquet_multiplier)

    animation = FuncAnimation(
        fig=fig_anim, func=update, frames=len(xs_time), interval=20
    )
    return animation


if __name__ == "__main__":
    bif_curve_nonaut(real_formulation=True, sparse=False)
    plt.figure()
    animation = bif_curve_aut(real_formulation=True, sparse=False)
    plt.show()
    print("All tests successful.")
