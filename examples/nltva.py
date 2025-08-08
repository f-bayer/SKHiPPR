"""Demonstration / Template for creating a simple frequency response curve of a non-autonomous dynamical system with HBM.
This example uses a Duffing oscillator, but ``ode`` can be instantiated as any non-autonomous :py:class:`~skhippr.odes.AbstractODE.AbstractODE`.
"""

import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import tikzplotlib

# --- Fourier configuration ---
from skhippr.Fourier import Fourier

# --- Differential equation ---
from skhippr.odes.nonautonomous import Duffing, NLTVA_FO

# --- HBM equation system ---
from skhippr.cycles.hbm import HBMSystem

# --- Stability method ---
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

# --- Continuation ---
from skhippr.solvers.continuation import pseudo_arclength_continuator, BranchPoint

# --- Newton solver ---
from skhippr.solvers.newton import NewtonSolver


def main():
    """
    Demonstration for creating a simple frequency response curve of a non-autonomous dynamical system with HBM.

    This function performs the following steps:

    #. Creation of :py:class:`~skhippr.Fourier`, :py:class:`~skhippr.solvers.newton.NewtonSolver`, and :py:class:`~skhippr.stability.KoopmmanHillProjection.KoopmanHillSubharmonic` objects to collect method parameters.
    #. Instantiation of a :py:class:`~skhippr.odes.AbstractODE.AbstractODE` (here: :py:class:`~skhippr.odes.nonautonomous.Duffing`) object which contains the ODE.
    #. Setup of an initial guess
    #. Setup and solution of the :py:class:`~skhippr.cycles.hbm.HBMEquation`, which formalizes the Harmonic Balance equations.
    #. Creation of an :py:class:`~skhippr.equations.EquationSystem.EquationSystem` containing only the HBM equations as input to the continuation method.
    #. Continuation of the frequency response curve using :py:func:`~skhippr.cycles.continuation.pseudo_arclength_continuator` and collecting the branch points
    #. Analyzing and plotting the branch points

    Returns
    -------

    None
    """

    # --- FFT, stability method and Newton solver configuration ---
    fourier = Fourier(N_HBM=25, L_DFT=300, n_dof=2, real_formulation=True)
    solver = NewtonSolver(verbose=True)

    eps = 0.05
    m1 = 1
    m2 = eps * m1
    k1 = 1
    k2 = (
        (8 * eps * k1)
        * (16 + 23 * eps + 9 * eps**2 + 2 * (2 + eps) * np.sqrt(4 + 3 * eps))
        / (3 * (1 + eps) ** 2 * (64 + 80 * eps + 27 * eps**2))
    )
    c1 = 0.002
    c2 = np.sqrt(k2 * m2 * (8 + 9 * eps - 4 * np.sqrt(4 + 3 * eps)) / (4 * (1 + eps)))

    knl1 = 1
    knl2 = 2 * eps**2 * knl1 / (1 + 4 * eps)

    F = 0.14
    omega_init = 0.5

    # Duffing, cf. Duffing example ---
    ode = Duffing(t=0, x=[1.0, 0.0], alpha=k1, beta=k2, delta=c1, F=F, omega=omega_init)

    ts = fourier.time_samples(omega_init)
    x0_samples = np.array(
        [np.cos(omega_init * ts), -omega_init * np.sin(omega_init * ts)]
    )

    plt.figure()
    # continue_and_plot(ode, x0_samples, fourier, solver, omega_init)

    # NLTVA
    nltva = NLTVA_FO(
        t=0,
        x=np.array([1.0, eps, 0.0, 0.0]),
        omega=omega_init,
        m=[m1, m2],
        c=[c1, c2],
        k=[k1, k2],
        k_nl=[knl1, knl2],
        F=F,
    )
    x0_samples = np.array(
        [
            np.cos(omega_init * ts),
            eps * np.cos(omega_init * ts),
            -omega_init * np.sin(omega_init * ts),
            -eps * omega_init * np.sin(omega_init * ts),
        ]
    )
    continue_and_plot(nltva, x0_samples, fourier, solver, omega_init)


def continue_and_plot(ode, x0_samples, fourier, solver, omega_init):

    fourier = Fourier(
        N_HBM=fourier.N_HBM, L_DFT=fourier.L_DFT, n_dof=ode.n_dof, real_formulation=True
    )
    stability_method = KoopmanHillSubharmonic(fourier, tol=1e-4, autonomous=False)
    X0 = fourier.DFT(x0_samples)

    # --- Set up the Harmonic Balance equations
    hbm = HBMSystem(
        ode=copy(ode),
        omega=omega_init,
        fourier=fourier,
        initial_guess=X0,
        period_k=1,
        stability_method=stability_method,
    )

    # --- Solve initial HBM problem. ---
    # As hbm is here an AbstractEquation, we can use solver.solve_equation()
    # and specify the unknown variable explicitly.
    # solver.solve_equation() internally instantiates an EquationSystem.
    solver.solve(hbm)
    solver.verbose = False

    # --- Preallocate the result of the FRC continuation ---
    # BranchPoints (a subclass of EquationSystem) extend the initial EquationSystem with one added equation for the tangency condition.
    frc: list[BranchPoint] = []

    # --- Iterate through the branch. Almost all arguments are optional. ---
    for branch_point in pseudo_arclength_continuator(
        initial_system=hbm,
        solver=solver,
        stepsize=0.1,
        stepsize_range=(0.001, 0.1),
        continuation_parameter="omega",
        initial_direction=1,
        verbose=True,
        num_steps=4000,
    ):
        frc.append(branch_point)

        # break if omega exceeds maximum
        if branch_point.omega > 4:
            break

    # --- Analyze the branch. ---
    freqs = np.array([np.squeeze(point.omega) for point in frc])
    stable = np.array([point.stable for point in frc])

    # --- Determine maximum x_1 amplitude of every point on the branch. ---
    # the first equation of the BranchPoint equation system is the HBMequation, which provides x_time().
    amps = np.array([np.max(point.equations[0].x_time()[0, :]) for point in frc])

    # --- plot ---
    # plt.figure()

    freqs_stable = np.where(stable, freqs, np.nan)
    freqs_unstable = np.where(~stable, freqs, np.nan)

    plt.plot(freqs_stable, amps, "r", label="stable")
    plt.plot(freqs_unstable, amps, "b--", label="unstable")

    plt.xlabel("\\omega")
    plt.ylabel("x_1")
    # plt.legend()


if __name__ == "__main__":
    main()
    # tikzplotlib.save("duffing.tikz", axis_width="5cm", axis_height="5cm")
    plt.show()
