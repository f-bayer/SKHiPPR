"""Mathieu equation examples for dissertation by Fabia Bayer, 2026"""

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


from skhippr.odes.ltp import MathieuODE
from skhippr.solvers.newton import NewtonSolver
from skhippr.Fourier import Fourier
from skhippr.cycles.hbm import HBMEquation
from skhippr.stability.ClassicalHill import ClassicalHill
from skhippr.equations.PseudoSpectrumEquation import compute_pseudospectrum

solver = NewtonSolver()


def plot_all_hill_EVs(epsilon, delta, N_HBM=10, omega=1, damping=0.1):
    fourier = Fourier(N_HBM=N_HBM, L_DFT=1024, n_dof=2)
    ode = MathieuODE(
        t=0, x=np.array([0.0, 0.0]), a=epsilon, b=delta, omega=omega, damping=damping
    )
    hbm = HBMEquation(
        ode,
        omega=omega,
        fourier=fourier,
        initial_guess=np.zeros((2 * N_HBM + 1) * 2),
        stability_method=ClassicalHill(fourier, "imaginary"),
    )
    solver = NewtonSolver(verbose=True)
    solver.solve_equation(hbm, "X")
    print(hbm.hill_matrix().shape)

    eigenvalues_correct = hbm.stability_method.hill_EVP(hbm, visualize=True)
    tikzplotlib.save("examples/dissertation_bayer/plots/mathieu_spurious.tikz")
    return hbm.hill_matrix(real_formulation=False)


def plot_mathieu_pseudospectrum(epsilon, delta, N_HBM=10, omega=1, damping=0.1):
    # Plot sorted eigenvalues of Hill matrix
    hill_mat = plot_all_hill_EVs(epsilon, delta, N_HBM=10, omega=1, damping=0.1)
    FE_all, _ = np.linalg.eig(hill_mat)

    # Consider LTI variant
    lti = MathieuODE(
        t=0, x=np.array([0.0, 0.0]), a=0, b=delta, omega=omega, damping=damping
    )
    A = lti.derivative(variable="x", update=True)
    hill_mat_A = np.kron(np.eye(2 * N_HBM + 1), A) + np.kron(
        np.diag(np.arange(-N_HBM, N_HBM + 1)), -1j * omega * np.eye(2)
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
                z_init=alpha + 1j * k * omega,
                verbose=True,
                tolerance=1e-11,
                max_step=0.001,
            )
            print(f"Pseudospectrum N = {k} computed.")
            plt.plot(np.real(alpha), np.imag(alpha) + k * omega, "r.")
            plt.plot(np.real(bdry), np.imag(bdry), "b")


if __name__ == "__main__":
    plot_all_hill_EVs(epsilon=2, delta=3)
    plot_mathieu_pseudospectrum(epsilon=2, delta=3)
    plt.show()
