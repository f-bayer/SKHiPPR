"""This example file creates some mathieu-equation-related figures for the Bayer2025 paper:
* Ince-Strutt diagram
* Special case on the stability boundary with coinciding eigenvalues"""

from collections.abc import Iterable
from re import M
import numpy as np
from copy import copy

from skhippr.Fourier import Fourier
from skhippr.problems.HBM import HBMProblem
from skhippr.systems.ltp import mathieu
from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillProjection,
    KoopmanHillSubharmonic,
)
import matplotlib.pyplot as plt


def main():
    a_grid = np.linspace(2.24, 2.34, 41)
    b_grid = np.linspace(0, 1, num=41)

    fourier = Fourier(N_HBM=30, L_DFT=500, n_dof=2, real_formulation=True)
    solution = np.zeros((2 * fourier.N_HBM + 1) * fourier.n_dof)
    problem = HBMProblem(
        mathieu,
        solution,
        omega=1,
        fourier=fourier,
        variable="y",
        stability_method=KoopmanHillSubharmonic(fourier=fourier, autonomous=False),
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
