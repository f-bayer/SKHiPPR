import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np
from scipy.integrate import solve_ivp

from skhippr.Fourier import Fourier
from skhippr.odes.ltp import MathieuODE
from skhippr.cycles.hbm import HBMEquation

from skhippr.stability.AbstractStabilityHBM import AbstractStabilityHBM
from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillProjection,
    KoopmanHillSubharmonic,
)


from pseudospectrum import does_pseudospectrum_include, compute_pseudospectrum


def optimal_error_bound(hbm: HBMEquation, t: float, subharmonic: bool = False):
    """
    Computes the optimal error bound for the solution of a Mathieu-type equation using the Harmonic Balance Method (HBM).
    The error bound is calculated based on the norms of the Fourier coefficient matrices of the system, assuming that the Fourier coefficient matrices have finite support and only J_0 and J_1 are nonzero.
    The formulas for the error are derived in Bayer & Leine (2025).

    Parameters
    ----------

    hbm : HBMEquation
        An instance of the HBMEquation class representing the system, containing the ODE and Fourier parameters.
    t : float
        The time parameter at which the error bound is evaluated.

    Returns
    -------

    E : float
        The computed optimal error bound for the given system and time.

    Notes
    -----

    - The function temporarily sets the parameter `b` of the ODE to zero to compute the norm of the matrix J_0.
    - The error bound formula depends on the relationship between `beta` (norm of J_0), `gamma` (norm of J_1),
      the number of harmonics `N`, and the time `t`.
    """

    # Determine norms of Fourier coeff matrices J_0, J_{\pm 1}
    b = hbm.ode.b
    hbm.ode.b = 0
    J_0 = hbm.ode.closed_form_derivative("x")
    hbm.ode.b = b
    beta = np.linalg.norm(J_0, 2)

    J_1 = np.array([[0, 0], [-0.5 * b, 0]])
    gamma = np.linalg.norm(J_1)

    # Optimal error bound - see Bayer&Leine, 2025
    N = hbm.fourier.N_HBM
    if subharmonic:
        N = 2 * N

    if beta < N / (4 * t):
        E = (8 * gamma * t / N) ** N * (np.exp(N) - 1)
    else:
        E = (2 * gamma / beta) ** N * (np.exp(4 * beta * t) - 1)

    return E


def plot_FMs_with_guarantee(ode, N, subh, ax=None, color=None, **kwargs):
    fourier = Fourier(N_HBM=N, L_DFT=10 * N, n_dof=2)
    X = np.zeros(2 * (2 * N + 1))
    hbm = HBMEquation(ode, ode.omega, fourier, X)
    hbm.residual(update=True)

    if subh:
        stability_method = KoopmanHillSubharmonic(fourier=fourier)
    else:
        stability_method = KoopmanHillProjection(fourier=fourier)

    if ax is None:
        ax = plot_FM(hbm, stability_method)

    E = optimal_error_bound(hbm, hbm.T_solution, subh)

    Phi_T = stability_method.fundamental_matrix(t_over_period=1, hbm=hbm)
    FMs, _ = np.linalg.eig(Phi_T)

    if 1e-14 < E < 10:
        for k, FM in enumerate(FMs):
            z_pseudospectrum = compute_pseudospectrum(
                Phi_T, epsilon=E, z_init=FM, verbose=True, max_step=max(3e-3, E)
            )
            if k == 0:
                label = f"N = {N}"
            else:
                label = None
            ax.plot(
                np.real(z_pseudospectrum),
                np.imag(z_pseudospectrum),
                label=label,
                **kwargs,
            )

    return ax


def plot_FM(hbm: HBMEquation, method: AbstractStabilityHBM = None):
    """Plot Floquet multipliers and unit circle"""

    _, ax = plt.subplots(1, 1)
    phis = np.linspace(0, 2 * np.pi, 1000)
    ax.plot(np.cos(phis), np.sin(phis), color="gray")
    ax.set_aspect("equal")

    if method is None:
        method = hbm.stability_method

    FMs = method.determine_eigenvalues(hbm)
    ax.plot(np.real(FMs), np.imag(FMs), "rx", label="Floquet multipliers")

    return ax


def plot_near_PD():
    a_vals = (-0.3549, -0.35485)
    for a in a_vals:
        ode = MathieuODE(0, np.array([0, 0]), a, 2.4, omega=2)

        # Simulate ODE at this configuration over 10 periods
        T = 2 * np.pi / ode.omega
        sol = solve_ivp(
            ode.dynamics,
            (0, 300 * T),
            [0, 0.05],
            dense_output=True,
            rtol=1e-13,
            atol=1e-13,
        )
        plt.figure()
        plt.plot(sol.t / T, sol.y[0])
        plt.xlabel("t/T")
        plt.ylabel("y_0")
        plt.title(f"Evolution for a = {a}")
        tikzplotlib.save(f"examples/convergence/evolution_a_{a}_b_{ode.b}.tikz")

        ax = None
        styles = ("--", "-", "-.")
        for N, style in zip([44, 45, 46], styles):
            ax = plot_FMs_with_guarantee(
                ode, N, subh=True, ax=ax, linestyle=style, color="k"
            )
        ax.set_title(f"Floquet multipliers and confidence regions for a = {a}")
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.legend()
        ax.set_xlim([-1.1, -0.9])
        ax.set_ylim([-0.13, 0.13])
        tikzplotlib.save(f"examples/convergence/FMs_a_{a}_b_{ode.b}.tikz")


def plot_N_over_a(ode, a_values, E_des=1e-6):
    pass


if __name__ == "__main__":
    plot_near_PD()
    plt.show()
