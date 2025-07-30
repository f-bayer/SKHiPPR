"""Template for getting started immediately on setting up a frc continuation problem.
This example uses a Duffing oscillator, but by modifying the content of system_function, one can immediately continue other nonautonomous systems.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Fourier configuration ---
from skhippr.Fourier import Fourier

# --- HBM equation system ---
from skhippr.cycles.hbm import HBMEquation

# --- Stability method ---
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

# --- Continuation ---
from skhippr.solvers.continuation import pseudo_arclength_continuator, BranchPoint


def main():
    """
    Demonstration for creating a simple frequency response curve with HBM.

    This function performs the following steps:

    #. Creation of a :py:class:`~skhippr.Fourier` object to collect FFT parameters.
    #. Creation of a :py:class:`~skhippr.stability.KoopmanHillProjection.KoopmanHillSubharmonic` object which defines the stability method.

    #. Setup and solution of the initial :py:class:`~skhippr.cycles.hbm.hbmProblem`.
    #. Continuation of the frequency response curve using :py:func:`~skhippr.cycles.continuation.pseudo_arclength_continuator`and collecting branch point properties
    #. plotting

    Returns
    -------

    None
    """

    # instantiation of the ODE

    # --- FFT, stability method configuration ---
    fourier = Fourier(N_HBM=25, L_DFT=300, n_dof=2, real_formulation=True)
    stability_method = KoopmanHillSubharmonic(
        fourier=fourier, tol=1e-4, autonomous=False
    )

    # --- HBM problem setup: Define keyword args for system function ---
    parameters = {"F": 0.5, "omega": 1}
    omega = parameters["omega"]

    # --- Define continuation parameter (here: could be 'omega' or 'F')
    key_param = "omega"

    # --- Initial guess in time and frequency domain ---
    ts = fourier.time_samples(omega)
    x0_samples = np.array([np.cos(omega * ts), -omega * np.sin(omega * ts)])
    X0 = fourier.DFT(x0_samples)

    initial_problem = HBMEquation(
        f=system_function,
        omega=omega,
        initial_guess=X0,
        fourier=fourier,
        stability_method=stability_method,
        period_k=1,
        parameters_f=parameters,
        verbose=True,
    )

    # --- Solve initial HBM problem---
    initial_problem.solve()
    print(initial_problem)
    assert initial_problem.converged

    # --- Avoid that the HBM problem prints a full report at each continuation step
    initial_problem.verbose = False

    # --- Force response continuation ---
    freqs = []
    amps = []
    branch = []
    stable = []
    unstable = []

    for branch_point in pseudo_arclength_continuator(
        initial_problem=initial_problem,
        stepsize=0.1,
        stepsize_range=(0.001, 0.1),
        key_param="omega",
        value_param=omega,
        verbose=True,
        num_steps=4000,
    ):
        branch.append(branch_point)
        stable.append(branch_point.stable)
        unstable.append(not branch_point.stable)
        freqs.append(branch_point.omega)

        # Maximum amplitude of first DOF over a period
        amps.append(np.max(branch_point.x_time()[1, :]))

        # break if omega exceeds maximum
        if branch_point.omega > 2.5:
            break

    # --- plot ---
    plt.figure()

    freqs_stable = np.array(freqs)
    freqs_stable[unstable] = np.nan

    freqs_unstable = np.array(freqs)
    freqs_unstable[stable] = np.nan

    plt.plot(freqs_stable, amps, color="blue")
    plt.plot(freqs_unstable, amps, color="red")

    plt.xlabel(key_param)
    plt.ylabel("|x_1|")


if __name__ == "__main__":
    main()
    plt.show()
