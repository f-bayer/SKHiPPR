"""This example file creates some mathieu-equation-related figures for the Bayer2025 paper:
* Ince-Strutt diagram
* Special case on the stability boundary with coinciding eigenvalues"""

from collections.abc import Callable, Iterable
from encodings.punycode import selective_find
from typing import override
import numpy as np
from copy import copy

from skhippr.Fourier import Fourier
from skhippr.problems import HBM
from skhippr.problems.HBM import HBMProblem
from skhippr.systems.ltp import mathieu
from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillProjection,
    KoopmanHillSubharmonic,
)
import matplotlib.pyplot as plt


class InceStruttProblem(HBMProblem):
    """
    This is a subclass of :py:class:`~skhippr.problems.HBM.HBMProblem` for finding periodic solutions as stability boundaries of an Ince-Strutt-Type chart.

    An additional anchor equation is used to fix a selected harmonic and degree of freedom to 1,
    identifying one (nontrivial) periodic solution out of the dense set of periodic solutions at a stability boundary.

    The following attributes are added/modified compared to the parent :py:class:`~skhippr.problems.HBM.HBMProblem`:

    * :py:attr:`~skhippr.problems.HBM.HBMProblem_autonomous.x` has one entry more: The last entry is the y coordinate of the Ince-Strutt diagram (usually 'b').
    * :py:attr:`~skhippr.problems.HBM.HBMProblem_autonomous.<varying_parameter>` returns the last entry of :py:attr:`~skhippr.problems.HBM.HBMProblem_autonomous.x`, which can change during Newton updates.
    * :py:func:`~skhippr.problems.HBM.HBMProblem_autonomous.residual_function` now has an appended equation for the anchor, which selects exactly one out of the dense periodic solutions at a stability boundary.
    * :py:attr:`~skhippr.problems.HBM.HBMProblem_autonomous.idx_anchor` is determined by the arguments ``harmo_anchor`` and ``dof_anchor`` and fixes the harmonic whose value is prescribed by the anchor.

    """

    def __init__(
        self,
        f: Callable[[np.ndarray], tuple[np.ndarray, dict[str, np.ndarray]]],
        initial_guess: np.ndarray,
        omega: float,
        fourier: Fourier,
        variable: str = "x",
        stability_method=None,
        tolerance: float = 1e-8,
        max_iterations: int = 20,
        verbose: bool = False,
        period_k=1,
        harmo_anchor: int = 1,
        dof_anchor: int = 0,
        parameters_f: dict[str, float] = None,
        varying_parameter="b",
    ):
        super().__init__(
            f=f,
            initial_guess=initial_guess,
            omega=omega,
            fourier=fourier,
            variable=variable,
            stability_method=stability_method,
            tolerance=tolerance,
            max_iterations=max_iterations,
            verbose=verbose,
            period_k=period_k,
            parameters_f=parameters_f,
        )
        self.idx_anchor = self._determine_anchor(harmo_anchor, dof_anchor)
        self.label = "Ince-Strutt HBM"

    def __getattr__(self, name):
        """
        Provides dynamic attribute access for the instance.
        If the requested attribute is varying_parameter, then the last entry of x is returned.
        """
        # Wrap the _problem object to provide direct access to its attributes and methods.
        # self.__getattr__(name) is ONLY called if self.name throws an AttributeError.
        if "varying_parameter" in self.__dict__ and name == self.varying_parameter:
            return self.x[(2 * self.fourier.N_HBM + 1) * self.fourier.n_dof]

        raise AttributeError(f"InceStruttProblem has no attribute '{name}'.")

    def __setattr__(self, name, value):
        """
        Custom attribute setter that delegates most attribute assignment of the parameter.
        """
        # Defer almost all parameters to the _problem
        if name in ("_problem", "anchor", "tangent", "variable", "x"):
            super().__setattr__(name, value)
        elif name in ("_list_params", "f"):
            pass
        else:
            setattr(self._problem, name, value)

            # TODOOOO find a better strategy tomorrow

    @property
    def omega(self):
        return self.x[-1]

    @omega.setter
    def omega(self, value):
        if len(self.x) == (2 * self.fourier.N_HBM + 1) * self.fourier.n_dof:
            # during initialization we set omega as the last element of x
            self.x = np.append(self.x, value)
        else:
            raise AttributeError(
                "property 'omega' of 'HBMProblem_autonomous' is read-only after initialization"
            )

    @override
    def determine_stability(self):
        super().determine_stability()
        if self.stability_method is not None:
            floquet_multipliers = self.eigenvalues
            idx_freedom_of_phase = np.argmin(abs(floquet_multipliers - 1))
            floquet_multipliers = np.delete(floquet_multipliers, idx_freedom_of_phase)
            self.stable = np.all(
                np.abs(floquet_multipliers) < 1 + self.stability_method.tol
            )

    @override
    def residual_function(self, x=None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Computes the residual vector and its derivatives for the current problem,
        including an anchor equation to constrain the phase.

        Parameters
        ----------
        x : np.ndarray, optional
            The input variable vector. If None, uses ``self.x``.

        Returns
        -------

        R : np.ndarray
            The residual vector. HBM residual with an additional 0 appended.
        derivatives : dict of str to np.ndarray
            Dictionary mapping variable names to their derivatives with respect to ``x``.
            The derivatives for the anchor equation are included, and the ``"omega"`` key is removed.
        """

        if x is None:
            x = self.x

        R, derivatives = super().residual_function(x[:-1])

        X_ext = self.x
        # anchor equation (phase may not change):
        # delta X[anchor[0]] = X[anchor[0]]/X[anchor[1]] * delta X[anchor[1]]

        anchor = np.zeros((1, X_ext.size), dtype=X_ext.dtype)
        anchor[0, self.idx_anchor[0]] = -1
        anchor[0, self.idx_anchor[1]] = (
            X_ext[self.idx_anchor[0]] / X_ext[self.idx_anchor[1]]
        )
        dR_dX = np.vstack(
            (
                np.hstack(
                    (
                        derivatives[self.variable],
                        derivatives["omega"][:, np.newaxis],
                    )
                ),
                anchor,
            )
        )

        derivatives[self.variable] = dR_dX
        del derivatives["omega"]

        R = np.append(R, 0)
        for key in derivatives:
            if len(derivatives[key].shape) == 1:
                derivatives[key] = np.append(derivatives[key], 0)

        return R, derivatives

    def _determine_anchor(self, harmo: int = 1, dof: int = 0) -> np.ndarray:
        """Determine the index of the anchor equation.
        The anchor equation ensures that the phase of the  harmo-th harmonic
        and the dof-th degree of freedom does not change during HBM solution for autonomous systems.
        """
        if self.fourier.real_formulation:
            # -tan(phi) = c_k/s_k = const -->  delta c = c_k/s_k * delta s
            idx_anchor = [
                harmo * self.fourier.n_dof + dof,
                (harmo + self.fourier.N_HBM) * self.fourier.n_dof + dof,
            ]
        else:
            # exp(i*phi) = X+/X- = const -->  delta X+ = X+/X- * delta X-
            idx_anchor = [
                (self.fourier.N_HBM + harmo) * self.fourier.n_dof + dof,
                (self.fourier.N_HBM - harmo) * self.fourier.n_dof + dof,
            ]

        # Avoid large numbers and division by zero
        if abs(self.x[idx_anchor[1]]) < (1e-4 * abs(self.x[idx_anchor[0]])):
            idx_anchor.reverse()

        return np.array(idx_anchor)

    @override
    def residual_function(self, x=None):
        return super().residual_function(x)


def main():
    a_grid = np.linspace(-0.5, 3, 101)
    b_grid = np.linspace(0, 5, num=101)

    fourier = Fourier(N_HBM=10, L_DFT=30, n_dof=2, real_formulation=True)
    solution = np.zeros((2 * fourier.N_HBM + 1) * fourier.n_dof)
    problem = HBMProblem(
        mathieu,
        solution,
        omega=1,
        fourier=fourier,
        variable="y",
        stability_method=KoopmanHillProjection(fourier=fourier, autonomous=False),
        parameters_f={"a": 1, "b": 1, "omega": 1, "d": 0},
        verbose=False,
    )
    _, magnitudes_fm = ince_strutt_rastered(a_grid, b_grid, problem)
    plot_magnitude(magnitudes_fm, logscale=True, a_grid=a_grid, b_grid=b_grid)


def ince_strutt_rastered(a_grid: Iterable, b_grid: Iterable, problem: HBMProblem):
    """Return a len(a_grid)*len(b_grid)*n_dof array of solved Ince-Strutt problems corresponding to the (a, b) grid"""

    ince_strutt = []
    magnitude_fm = np.zeros((len(a_grid), len(b_grid)))

    for k, b in enumerate(b_grid):
        list_b = []
        print(f"b = {b:.3f} ({k:2d}/{len(b_grid):2d}) ", end="\n")

        for l, a in enumerate(a_grid):
            problem = copy(problem)
            problem.a = a
            problem.b = b
            problem.reset()
            problem.solve()
            if not problem.converged:
                raise RuntimeError(f"Ince-Strutt problem not converged at a={a}, b={b}")
            list_b.append(problem)
            magnitude_fm[k, l] = np.max(np.abs(problem.eigenvalues))

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
    plt.pcolormesh(a_grid, b_grid, shade, cmap="gray_r", norm=norm)
    cbar = plt.colorbar(label="largest FM (magnitude)")
    plt.xlabel("$a$")
    plt.ylabel("$b$")

    # # Manually rectify the ticks to remove the scaling/shifting:
    # ticks = cbar.get_ticks()
    # if logscale:
    #     cbar.set_ticks(ticks)
    #     cbar.set_ticklabels([f"10^{tick}" for tick in ticks])
    # else:
    #     cbar.set_ticks(ticks - 1)
    #     cbar.set_ticklabels([f"{tick}" for tick in ticks])


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
