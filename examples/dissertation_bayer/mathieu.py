"""Mathieu equation examples for dissertation by Fabia Bayer, 2026"""

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


from skhippr.odes.ltp import MathieuODE
from skhippr.solvers.newton import NewtonSolver
from skhippr.Fourier import Fourier
from skhippr.cycles.hbm import HBMEquation
from skhippr.stability.ClassicalHill import ClassicalHill

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

    # # Plot all eigenvalues
    # eigenvalues_all, _ = np.linalg.eig(hbm.hill_matrix())
    # plt.figure()
    # plt.plot(np.real(eigenvalues_all), np.imag(eigenvalues_all), ".")

    # circle "correct" eigenvalues
    eigenvalues_correct = hbm.stability_method.hill_EVP(hbm, visualize=True)
    tikzplotlib.save("examples/dissertation_bayer/plots/mathieu_spurious.tikz")


if __name__ == "__main__":
    plot_all_hill_EVs(epsilon=0.1, delta=0.2)
    plt.show()
