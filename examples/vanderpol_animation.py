import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# ODE
from skhippr.odes.autonomous import Vanderpol

# FFT configuration
from skhippr.Fourier import Fourier

# HBM and stability
from skhippr.cycles.hbm import HBMSystem
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

# Solution procedure
from skhippr.solvers.newton import NewtonSolver
from skhippr.solvers.continuation import pseudo_arclength_continuator

# only for type hinting
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.odes.AbstractODE import AbstractODE
from skhippr.solvers.continuation import BranchPoint


def main():

    print("Van der Pol oscillator: continuation w.r.t. mu")

    # setup
    newton_solver = NewtonSolver(verbose=True)
    ode = Vanderpol(x=[2.0, 0.0], nu=0.1)
    hbm_system: EquationSystem = setup_hbm_system(ode, newton_solver)

    # continuation
    branch: list[BranchPoint] = []
    nu_range = (ode.nu, 10)
    newton_solver.verbose = False

    for branch_point in pseudo_arclength_continuator(
        initial_system=hbm_system,
        solver=newton_solver,
        stepsize=0.1,
        stepsize_range=(0.001, 0.15),
        initial_direction=1,
        num_steps=1000,
        continuation_parameter="nu",
        verbose=True,
    ):
        branch.append(branch_point)
        if not nu_range[0] <= branch_point.nu <= nu_range[1]:
            break

    # analysis
    xs_time, amplitudes, nus, stable, FMs, omegas = parse_branch(branch)

    # visualization
    animation = animate_phase_portrait_and_FMs(nus, xs_time, FMs)
    plot_with_stability(nus, amplitudes, stable, "$\\nu$", "$|x_1|$")
    plot_with_stability(nus, omegas, stable, "$\\nu$", "$\\omega$")

    return animation


def setup_hbm_system(ode: AbstractODE, solver: NewtonSolver = None):

    omega_0 = 1
    fourier = Fourier(N_HBM=45, L_DFT=1000, n_dof=ode.n_dof, real_formulation=True)
    stability_method = KoopmanHillSubharmonic(fourier=fourier, tol=1e-4)
    X0 = generate_initial_condition(fourier, omega_0)

    hbm_system = HBMSystem(ode, omega_0, fourier, X0, stability_method=stability_method)
    if solver:
        solver.solve(hbm_system)
        print(
            f"Initial problem convergence: {hbm_system.solved}. omega = {hbm_system.omega}"
        )
    return hbm_system


def generate_initial_condition(fourier, omega_0):
    ts = fourier.time_samples(omega_0)
    x0_samples = np.vstack(
        (2 * np.cos(omega_0 * ts), -2 * omega_0 * np.sin(omega_0 * ts))
    )
    X0 = fourier.DFT(x0_samples)
    return X0


def parse_branch(branch: list[BranchPoint]):
    xs_time = [point.equations[0].x_time() for point in branch]
    amplitudes = np.array([np.max(x_time[0, :]) for x_time in xs_time])
    stable = np.array([point.stable for point in branch])
    nus = np.array([np.squeeze(point.nu) for point in branch])
    floquet_multipliers = [point.eigenvalues for point in branch]
    omegas = [np.squeeze(point.omega) for point in branch]

    return xs_time, amplitudes, nus, stable, floquet_multipliers, omegas


def plot_with_stability(x_values, y_values, stable, xlabel, ylabel):

    x_stable = np.where(stable, x_values, np.nan)
    x_unstable = np.where(~stable, x_values, np.nan)

    plt.figure()
    plt.plot(x_stable, y_values, "r-", label="stable")
    plt.plot(x_unstable, y_values, "b--", label="unstable")
    # plt.title(f"Amplitude of first DOF -- {solver}")
    plt.legend()
    plt.xlabel("$\\nu$")
    plt.ylabel(ylabel)


def animate_phase_portrait_and_FMs(nus, xs_time, floquet_multipliers):

    fig_anim, axs = plt.subplots(nrows=1, ncols=2)

    # Initially populate the plots
    (line_period,) = axs[0].plot(xs_time[0][0, :], xs_time[0][1, :])
    axs[0].axis("equal")
    title_phase = axs[0].set_title("Phase portrait")

    (plot_floquet_multiplier,) = axs[1].plot(
        np.real(floquet_multipliers[0]),
        np.imag(floquet_multipliers[0]),
        "x",
    )
    phis = np.linspace(0, 2 * np.pi)
    axs[1].plot(np.cos(phis), np.sin(phis), "k-")
    axs[1].axis("equal")
    title_FM = axs[1].set_title("Floquet multipliers")

    # Animation function
    def update(frame):
        line_period.set_xdata(xs_time[frame][0, :])
        line_period.set_ydata(xs_time[frame][1, :])
        plot_floquet_multiplier.set_xdata(np.real(floquet_multipliers[frame]))
        plot_floquet_multiplier.set_ydata(np.imag(floquet_multipliers[frame]))
        title_phase.set_text(f"Phase portrait: $\\nu$ = {nus[frame]:.2f}")
        title_FM.set_text(f"Floquet multipliers: $\\nu$ = {nus[frame]:.2f}")
        return (line_period, plot_floquet_multiplier, title_phase, title_FM)

    animation = FuncAnimation(
        fig=fig_anim, func=update, frames=len(xs_time), interval=20
    )
    return animation, xs_time


if __name__ == "__main__":
    animation = main()
    plt.show()
