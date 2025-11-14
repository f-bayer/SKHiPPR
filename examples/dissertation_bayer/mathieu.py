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
        stability_method=ClassicalHill(fourier, "symmetry"),
    )
    solver = NewtonSolver(verbose=True)
    solver.solve_equation(hbm, "X")

    hill_matrix = hbm.hill_matrix(real_formulation=False)
    print(hill_matrix.shape)

    eigenvalues, eigenvectors = np.linalg.eig(hill_matrix)

    if ax_FE is None:
        _, ax_FE = plt.subplots(1, 1)

    ax_FE.plot(np.real(eigenvalues), np.imag(eigenvalues), ".", **plot_args)
    if save:
        tikzplotlib.save("examples/dissertation_bayer/plots/mathieu_spurious.tikz")

    if ax_FM is not None:
        FMs = np.exp(eigenvalues * 2 * np.pi / omega)
        ax_FM.plot(np.real(FMs), np.imag(FMs), ".", **plot_args)
        FMs_choice = hbm.eigenvalues
        ax_FM.plot(
            np.real(FMs_choice), np.imag(FMs_choice), "x", mfc="none", **plot_args
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


def continuity_sort(eigenvalues_old, eigenvalues_new, omega):

    eigenvalues_sorted = []
    idx_all = np.arange(len(eigenvalues_new))

    if len(eigenvalues_old) < len(eigenvalues_new):
        N_HBM = int(0.5 * (len(eigenvalues_new) / len(eigenvalues_old) - 1))
        for k in zigzag(N_HBM):
            for eigenvalue_old in eigenvalues_old:
                FE = eigenvalue_old + k * omega * 1j
                idx_candidates = idx_all[
                    np.abs(np.imag(eigenvalues_new) - np.imag(eigenvalue_old))
                    < 0.5 * omega
                ]
                if len(idx_candidates) == 0:
                    idx_candidates = idx_all
                idx_new = np.argmin(np.abs(FE - eigenvalues_new[idx_candidates]))
                idx_new = idx_candidates[idx_new]
                eigenvalues_sorted.append(eigenvalues_new[idx_new])
                eigenvalues_new[idx_new] = np.inf

    else:
        for FE in eigenvalues_old:
            idx_candidates = idx_all[
                np.abs(np.imag(eigenvalues_new) - np.imag(FE)) < 0.5 * omega
            ]
            if len(idx_candidates) == 0:
                idx_candidates = idx_all
            idx_new = np.argmin(np.abs(FE - eigenvalues_new[idx_candidates]))
            idx_new = idx_candidates[idx_new]
            eigenvalues_sorted.append(eigenvalues_new[idx_new])
            eigenvalues_new[idx_new] = np.inf

    eigenvalues_sorted = np.array(eigenvalues_sorted)
    eigenvalues_new[:] = eigenvalues_sorted
    return eigenvalues_sorted


def zigzag(N):
    yield 0
    for i in range(1, N + 1):
        yield i
        yield -i


def plot_eigenvalues_continuously(
    hbm, name_param, vals_param, ax_cont=None, ax_FE=None, ax_FM=None, solver=None
):

    if solver is None:
        solver = NewtonSolver(verbose=False)

    if ax_cont is None:
        _, ax_cont = plt.subplots(1, 1)

    eigenvalues_old = []
    factors = (vals_param - np.min(vals_param)) / (
        np.max(vals_param) - np.min(vals_param)
    )

    eigenvalues_all = np.zeros((len(hbm.X), len(vals_param)), dtype=complex)

    for k, (param, factor) in enumerate(zip(vals_param, factors)):
        print(f"{name_param} = {param}")
        setattr(hbm, name_param, param)
        solver.solve_equation(hbm, "X")

        if len(eigenvalues_old) == 0:
            # old eigenvalues derived from constant case
            J_coeffs = hbm.ode_coeffs()
            if hbm.fourier.real_formulation:
                A = J_coeffs[:, :, 0]
            else:
                A = J_coeffs[:, :, hbm.fourier.N_HBM]
            eigenvalues_old, _ = np.linalg.eig(A)

        eigenvalues, _ = np.linalg.eig(hbm.hill_matrix(update=True))
        FEs, _ = hbm.stability_method.hill_EVP(hbm, visualize=False)

        color = (factor, 0, factor)

        if ax_FE is not None:
            ax_FE.plot(np.real(eigenvalues), np.imag(eigenvalues), ".", color=color)
            ax_FE.plot(np.real(FEs), np.imag(FEs), "o", color=color, mfc="none")

        if ax_FM is not None:
            FMs_all = np.exp(eigenvalues * 2 * np.pi / hbm.omega)
            FMs = np.exp(FEs * 2 * np.pi / hbm.omega)
            ax_FM.plot(np.real(FMs_all), np.imag(FMs_all), ".", color=color)
            ax_FM.plot(np.real(FMs), np.imag(FMs), "o", color=color, mfc="none")

        eigenvalues = continuity_sort(eigenvalues_old, eigenvalues, hbm.omega)
        eigenvalues_all[:, k] = eigenvalues
        eigenvalues_old = eigenvalues

        ax_cont.cla()
        for ii in range(len(eigenvalues_all[:, 0])):
            ax_cont.plot(
                np.real(eigenvalues_all[ii, : (k + 1)]),
                np.imag(eigenvalues_all[ii, : (k + 1)]),
                ".",
            )
        pass


if __name__ == "__main__":
    # Plot first demo case
    epsilon = 3
    delta = 2
    ode = MathieuODE(t=0, x=np.array([0, 0]), a=delta, b=epsilon, damping=0.1, omega=1)
    # plot_all_hill_EVs(ode=ode, N_HBM=10, save=True, ax_FE=None)
    # plot_mathieu_pseudospectrum(epsilon=3, delta=2)

    # Plot varying epsilon case
    fig, ax = plt.subplots(2, 2)
    phis = np.linspace(0, 2 * np.pi, 200)
    ax[0, 1].plot(np.cos(phis), np.sin(phis), "k")
    ax[0, 1].set_aspect("equal")
    ax[0, 1].set_xlim([-1.5, 1.5])
    ax[0, 1].set_ylim([-1.5, 1.5])
    ax[0, 0].set_xlim([-1, 1])

    delta = 12
    epsilons = np.linspace(0, 12, 200)
    fourier = Fourier(N_HBM=3, real_formulation=False, L_DFT=1024, n_dof=2)

    ode = MathieuODE(t=0, x=np.array([0.0, 0.0]), b=epsilons[0], a=delta)
    hbm = HBMEquation(
        ode=ode,
        omega=ode.omega,
        fourier=fourier,
        initial_guess=np.zeros(ode.n_dof * (2 * fourier.N_HBM + 1)),
        stability_method=ClassicalHill(fourier, "symmetry"),
    )

    plot_eigenvalues_continuously(
        hbm, "b", epsilons, ax[1, 0], ax_FE=ax[0, 0], ax_FM=ax[0, 1]
    )

    # delta_max = 10
    # epsilon_max = 20
    # for factor in np.linspace(0, 1, 10):
    #     ode = MathieuODE(
    #         t=0,
    #         x=np.array([0, 0]),
    #         b=factor * epsilon_max,
    #         a=delta_max,
    #         damping=0.001,
    #         omega=1,
    #     )
    #     plot_all_hill_EVs(
    #         ode=ode,
    #         N_HBM=3,
    #         ax_FE=ax[0],
    #         ax_FM=ax[1],
    #         color=(factor, 0, factor),
    #         save=False,
    #     )
    #     # TODO plot eigenvalues continuously

    plt.show()
