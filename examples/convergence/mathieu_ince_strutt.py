"""This example file creates some mathieu-equation-related figures for the Bayer2025 paper:

* Ince-Strutt diagram
* Special case on the stability boundary with coinciding eigenvalues

"""

from collections.abc import Callable, Iterable
from typing import override
import numpy as np
from copy import copy

from skhippr.Fourier import Fourier
from skhippr.equations.AbstractEquation import AbstractEquation
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.cycles.hbm import HBMEquation
from skhippr.odes.ltp import MathieuODE
from skhippr.solvers.newton import NewtonSolver
from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillProjection,
    KoopmanHillSubharmonic,
)
import matplotlib.pyplot as plt


class FixedHarmonic(AbstractEquation):
    """This equation fixes the cosine component of the ``harmo``-th harmonic of the ``dof``-th degree of  freedom to ``value``.
    ``fourier`` is used during initialization to know the correct number of degrees of freedom and the formulation.
    """

    def __init__(self, X, fourier, harmo=1, dof=0, value=1.0):
        super().__init__(None)
        self.X = X
        self.value = value
        self.anchor = np.zeros(fourier.n_dof * (2 * fourier.N_HBM + 1))
        if fourier.real_formulation:
            self.anchor[harmo * fourier.n_dof + dof] = 1
        else:
            self.anchor[
                [
                    (self.fourier.N_HBM + harmo) * self.fourier.n_dof + dof,
                    (self.fourier.N_HBM - harmo) * self.fourier.n_dof + dof,
                ]
            ] = 0.5

    def residual_function(self):
        return np.atleast_1d(np.inner(self.anchor, self.X) - self.value)

    def closed_form_derivative(self, variable):
        match variable:
            case "X":
                return np.atleast_2d(self.anchor)
            case "value":
                return np.atleast_2d(-1)
            case _:
                return np.atleast_2d(0)


class InceStruttSystem(EquationSystem):
    """
    This is a subclass of :py:class:`~skhippr.problems.HBM.HBMProblem` for finding periodic solutions as stability boundaries of an Ince-Strutt-Type chart.

    An additional equation is appended to the HBMEquation used to fix a selected harmonic and degree of freedom to 1,
    identifying one (nontrivial) periodic solution out of the dense set of periodic solutions at a stability boundary.

    The following attributes are added/modified compared to the parent :py:class:`~skhippr.problems.HBM.HBMProblem`:

    """

    def __init__(
        self,
        hbm: HBMEquation,
        harmo_anchor=1,
        dof_anchor=0,
        varying_parameter="b",
    ):
        anchor = FixedHarmonic(hbm.X, hbm.fourier, harmo_anchor, dof_anchor)

        super().__init__(
            equations=[hbm, anchor],
            unknowns=["X", varying_parameter],
            equation_determining_stability=hbm,
        )


def main():
    a_grid = np.linspace(-0.5, 3, 101)
    b_grid = np.linspace(0, 5, num=101)

    fourier = Fourier(N_HBM=10, L_DFT=30, n_dof=2, real_formulation=True)
    X = np.zeros((2 * fourier.N_HBM + 1) * fourier.n_dof)
    ode = MathieuODE(t=0, x=np.array([0.0, 0.0]), a=1, b=1, omega=1, damping=0)
    hbm = HBMEquation(
        ode=ode,
        omega=ode.omega,
        fourier=fourier,
        initial_guess=X,
        stability_method=KoopmanHillProjection(fourier=fourier),
    )
    # _, magnitudes_fm = ince_strutt_rastered(a_grid, b_grid, hbm)
    # ax = plot_magnitude(magnitudes_fm, logscale=True, a_grid=a_grid, b_grid=b_grid)

    # Continuation along stability boundary
    ode.a = -0.001
    ode.b = 0
    sys_IS = InceStruttSystem(hbm)
    solver = NewtonSolver()
    solver.solve(sys_IS)
    assert sys_IS.solved


def ince_strutt_rastered(a_grid: Iterable, b_grid: Iterable, hbm: HBMEquation):
    """Return a len(a_grid)*len(b_grid)*n_dof array of solved HBM equations corresponding to the (a, b) grid"""

    solver = NewtonSolver()
    ince_strutt = []
    magnitude_fm = np.zeros((len(a_grid), len(b_grid)))

    for k, b in enumerate(b_grid):
        list_b = []
        print(f"b = {b:.3f} ({k:2d}/{len(b_grid):2d}) ", end="\n")

        for l, a in enumerate(a_grid):
            hbm = copy(hbm)
            hbm.a = a
            hbm.b = b
            solver.solve_equation(equation=hbm, unknown="X")  # solved at initial guess
            # Update the derivative as it was not used when solved at initial guess
            hbm.derivative("X", update=True)
            stable = hbm.determine_stability(update=True)
            list_b.append(hbm)
            magnitude_fm[k, l] = np.max(np.abs(hbm.eigenvalues))

        ince_strutt.append(list_b)

    return ince_strutt, magnitude_fm


def plot_magnitude(magnitude_fm, logscale=True, a_grid=None, b_grid=None):

    # if logscale:
    #     magnitude_fm = np.log10(magnitude_fm)
    # else:
    #     # ensure that stability cutoff is at magnitude = 0
    #     magnitude_fm = magnitude_fm - 1

    max_val = np.max(magnitude_fm)
    shade = np.minimum(magnitude_fm, 10)  # np.maximum(0, magnitude_fm)

    if logscale:
        norm = "log"
    else:
        norm = "linear"

    # Show image
    fig, ax = plt.subplots()
    ax.pcolormesh(a_grid, b_grid, shade, cmap="gray_r", norm=norm)
    cbar = plt.colorbar(label="largest FM (magnitude)")
    ax.xlabel("$a$")
    ax.ylabel("$b$")

    return ax


def test_plotting_magnitudes():
    magnitudes = 10 ** (3 * np.random.rand(150, 150))
    a_grid = -3 + np.arange(magnitudes.shape[1])
    b_grid = 5 + np.arange(magnitudes.shape[0])

    for logscale in (True, False):
        plt.figure()
        plot_magnitude(magnitudes, logscale=logscale, a_grid=a_grid, b_grid=b_grid)

    plt.show()


if __name__ == "__main__":
    # test_plotting_magnitudes()
    main()
    plt.show()
