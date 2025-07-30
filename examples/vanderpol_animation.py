import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from skhippr.Fourier import Fourier
from skhippr.systems.autonomous import Vanderpol
from skhippr.problems.HBM import HBMSystem
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic
from skhippr.problems.newton import NewtonSolver
from skhippr.problems.continuation import pseudo_arclength_continuator


def plot_bif_curve_aut():

    ode = Vanderpol(x=[2.0, 0.0], nu=0.1)
    omega_0 = 1

    print("Van der Pol oscillator")

    fourier = Fourier(N_HBM=45, L_DFT=1000, n_dof=ode.n_dof, real_formulation=True)
    solver = NewtonSolver()

    ts = fourier.time_samples(omega_0)
    x0_samples = np.vstack(
        (2 * np.cos(omega_0 * ts), -2 * omega_0 * np.sin(omega_0 * ts))
    )
    X0 = fourier.DFT(x0_samples)

    stability_method = KoopmanHillSubharmonic(fourier=fourier, tol=1e-4)

    hbm = HBMSystem(ode, omega_0, fourier, X0, stability_method=stability_method)
    solver.solve(hbm)

    print(f"Initial problem convergence: {hbm.solved}")

    nu_range = (ode.nu, 10)

    xs_time = []
    amplitudes = []
    nus = []
    stable = []
    omegas = []
    floquet_multipliers = []
    x0s = []

    for branch_point in pseudo_arclength_continuator(
        initial_system=hbm,
        solver=solver,
        stepsize=0.1,
        stepsize_range=(0.001, 0.15),
        initial_direction=1,
        num_steps=1000,
        continuation_parameter="nu",
        verbose=True,
    ):
        xs_time.append(branch_point.equations[0].x_time())
        amplitudes.append(np.max(xs_time[-1][0, :]))
        nus.append(branch_point.nu)
        stable.append(branch_point.stable)
        omegas.append(branch_point.omega)
        floquet_multipliers.append(branch_point.eigenvalues)

        if not nu_range[0] <= branch_point.nu <= nu_range[1]:
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
    plt.xlabel("$\\nu$")
    plt.ylabel("$|x_1|$")

    plt.figure()
    plt.plot(nu_stable, omegas, "r", label="stable")
    plt.plot(nu_unstable, omegas, "b", label="unstable")
    plt.xlabel(xlabel="$\\nu$")
    plt.ylabel("$\\omega$")

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
    animation = plot_bif_curve_aut()
    plt.show()
