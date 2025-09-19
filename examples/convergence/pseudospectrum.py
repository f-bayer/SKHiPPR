"""This file illustrates the relationship between eigenvalues and errors in Phi for the Mathieu equation."""

import numpy as np
import matplotlib.pyplot as plt

from skhippr.odes.ltp import MathieuODE

from skhippr.Fourier import Fourier
from skhippr.equations.AbstractEquation import AbstractEquation
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.cycles.hbm import HBMEquation
from skhippr.solvers.newton import NewtonSolver
from skhippr.solvers.continuation import pseudo_arclength_continuator


class PseudoSpectrumEquation(AbstractEquation):
    def __init__(self, A, epsilon, z=None):
        super().__init__(stability_method=None)
        self.A = A
        self.epsilon = epsilon
        self.eye = np.eye(A.shape[0])

        if z is None:
            eigenvalues, _ = np.linalg.eig(A)
            z = eigenvalues[0]

        self.re_z = np.real(z)
        self.im_z = np.imag(z)

    def residual_function(self):
        z = self.re_z + 1j * self.im_z
        R = self.A - z * self.eye
        _, s, _ = np.linalg.svd(R)
        sigma_min = np.min(s)
        return np.atleast_1d(sigma_min - self.epsilon)

    def closed_form_derivative(self, variable):
        if variable == "epsilon":
            return np.atleast_2d(-1)

        return super().closed_form_derivative(variable)


def compute_pseudospectrum(A=None, epsilon=0.3, z_init=None):

    eq = PseudoSpectrumEquation(A=A, epsilon=epsilon, z=z_init)
    sys = EquationSystem(equations=[eq], unknowns=["re_z", "im_z"])
    solver = NewtonSolver(verbose=False)

    pseudo_spectrum = []

    for bp in pseudo_arclength_continuator(
        initial_system=sys,
        solver=solver,
        verbose=True,
        num_steps=500,
    ):
        z = bp.re_z + 1j * bp.im_z
        pseudo_spectrum.append(z)
        if len(pseudo_spectrum) > 1 and np.imag(z) > np.imag(
            pseudo_spectrum[0]
        ) > np.imag(pseudo_spectrum[-2]):
            break

    return pseudo_spectrum


def plot_pseudospectrum(A, epsilon=0.3, ax=None):

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    eigs, _ = np.linalg.eig(A)
    ax.plot(np.real(eigs), np.imag(eigs), "rx")

    for z in eigs:
        p_spectrum = compute_pseudospectrum(A, epsilon, z)
        ax.plot(np.real(p_spectrum), np.imag(p_spectrum), ".")

    ax.set_aspect("equal")


def examples_pseudospectrum():
    # normal matrix: pseudospectrum is a circle
    eigs = [1 + 1j, 2]
    A = np.diag(eigs)
    plot_pseudospectrum(A)

    # non-normal matrix
    A = np.array([[1, 50], [0, 0.4]])
    plot_pseudospectrum(A, epsilon=0.002)


if __name__ == "__main__":

    examples_pseudospectrum()
    plt.show()
