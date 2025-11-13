"""Mathieu equation examples for dissertation by Fabia Bayer, 2026"""

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


from skhippr.odes.ltp import MathieuODE, TruncatedMeissner
from skhippr.solvers.newton import NewtonSolver
from skhippr.Fourier import Fourier
from skhippr.cycles.hbm import HBMEquation
from skhippr.stability.ClassicalHill import ClassicalHill
from skhippr.equations.PseudoSpectrumEquation import compute_pseudospectrum

solver = NewtonSolver()


def plot_all_hill_EVs(ode, N_HBM=10, ax_FE=None, ax_FM=None, save=False, **plot_args):
    fourier = Fourier(N_HBM=N_HBM, L_DFT=1024, n_dof=2)
    # ode = TruncatedMeissner(
    #     t=0,
    #     x=np.array([0.0, 0.0]),
    #     a=delta,
    #     b=epsilon,
    #     omega=omega,
    #     damping=damping,
    #     N_harmonics=20,
    # )
    omega = ode.omega
    hbm = HBMEquation(
        ode,
        omega=omega,
        fourier=fourier,
        initial_guess=np.zeros((2 * N_HBM + 1) * 2),
        stability_method=ClassicalHill(fourier, "imaginary"),
    )
    solver = NewtonSolver(verbose=True)
    solver.solve_equation(hbm, "X")

    hill_matrix = hbm.hill_matrix(real_formulation=False)
    print(hill_matrix.shape)

    eigenvalues, eigenvectors = np.linalg.eig(hill_matrix)

    if ax_FE is None:
        _, ax_FE = plt.subplots(1, 1)

    ax_FE.plot(np.real(eigenvalues), np.imag(eigenvalues), "x", **plot_args)
    if save:
        tikzplotlib.save("examples/dissertation_bayer/plots/mathieu_spurious.tikz")

    if ax_FM is not None:
        FMs = np.exp(eigenvalues * 2 * np.pi / omega)
        ax_FM.plot(np.real(FMs), np.imag(FMs), "x", **plot_args)
        FMs_choice = hbm.eigenvalues
        ax_FM.plot(
            np.real(FMs_choice), np.imag(FMs_choice), "o", mfc="none", **plot_args
        )
        ax_FM.plot()
    return hill_matrix


def plot_mathieu_pseudospectrum(ode, N_HBM=10):
    # Plot sorted eigenvalues of Hill matrix
    hill_mat = plot_all_hill_EVs(ode, N_HBM=10)
    FE_all, _ = np.linalg.eig(hill_mat)

    # Consider LTI variant
    lti = ...  # TO DO
    A = lti.derivative(variable="x", update=True)
    hill_mat_A = np.kron(np.eye(2 * N_HBM + 1), A) + np.kron(
        np.diag(np.arange(-N_HBM, N_HBM + 1)), -1j * ode.omega * np.eye(2)
    )
    E = np.linalg.norm(hill_mat_A - hill_mat, ord=2)
    print(f"E for pseudospectrum: {E}")

    plt.figure()
    plt.plot(np.real(FE_all), np.imag(FE_all), "kx")
    alphas, _ = np.linalg.eig(a=A)
    for alpha in alphas:
        # for k in range(-N_HBM, N_HBM + 1):
        for k in [0]:
            bdry = compute_pseudospectrum(
                hill_mat_A,
                E,
                z_init=alpha + 1j * k * ode.omega,
                verbose=True,
                tolerance=1e-11,
                max_step=0.001,
            )
            print(f"Pseudospectrum N = {k} computed.")
            plt.plot(np.real(alpha), np.imag(alpha) + k * ode.omega, "r.")
            plt.plot(np.real(bdry), np.imag(bdry), "b")


if __name__ == "__main__":
    # Plot first demo case
    epsilon = 3
    delta = 2
    ode = MathieuODE(t=0, x=np.array([0, 0]), a=delta, b=epsilon, damping=0.1, omega=1)
    plot_all_hill_EVs(ode=ode, N_HBM=10, save=True, ax_FE=None)
    # plot_mathieu_pseudospectrum(epsilon=3, delta=2)

    # Plot varying epsilon case
    fig, ax = plt.subplots(2, 1)
    phis = np.linspace(0, 2 * np.pi, 200)
    ax[1].plot(np.cos(phis), np.sin(phis), "k")
    ax[1].set_aspect("equal")
    ax[1].set_xlim([-1.5, 1.5])
    ax[1].set_ylim([-1.5, 1.5])

    delta_max = 10
    epsilon_max = 20
    for factor in np.linspace(0, 1, 50):
        ode = MathieuODE(
            t=0,
            x=np.array([0, 0]),
            b=factor * epsilon_max,
            a=delta_max,
            damping=0.001,
            omega=1,
        )
        plot_all_hill_EVs(
            ode=ode,
            N_HBM=3,
            ax_FE=ax[0],
            ax_FM=ax[1],
            color=(factor, factor, 0),
            save=False,
        )

    plt.show()
