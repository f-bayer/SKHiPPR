"""This example file creates some mathieu-equation-related figures for the Bayer2025 paper:
* Ince-Strutt diagram
* Special case on the stability boundary with coinciding eigenvalues"""

from collections.abc import Iterable
import numpy as np
from copy import copy
from skhippr import stability
from skhippr.problems.HBM import HBMProblem
import matplotlib.pyplot as plt


def ince_strutt_rastered(a_grid: Iterable, b_grid: Iterable, ISproblem: HBMProblem):
    """Return a len(a_grid)*len(b_grid)*n_dof array of solved Ince-Strutt problems corresponding to the (a, b) grid"""

    ince_strutt = []
    magnitude_fm = np.zeros((len(a_grid), len(b_grid)))

    for k, b in enumerate(b_grid):
        list_b = []

        for l, a in enumerate(a_grid):
            ISproblem = copy(ISproblem)
            ISproblem.a = a
            ISproblem.b = b
            ISproblem.converged = False
            ISproblem.solve()
            if not ISproblem.check_converged():
                raise RuntimeError(f"Ince-Strutt problem not converged at a={a}, b={b}")
            list_b.append(ISproblem)
            magnitude_fm[k, l] = np.max(np.abs(ISproblem.eigenvalues))

        ince_strutt.append(list_b)

    return ince_strutt, magnitude_fm


def plot_magnitude(magnitude_fm, logscale=True):

    if logscale:
        magnitude_fm = np.log10(magnitude_fm)
    else:
        # ensure that stability cutoff is at magnitude = 0
        magnitude_fm = magnitude_fm - 1

    max_val = np.max(magnitude_fm)
    shade = np.maximum(0, magnitude_fm)

    # Show image
    plt.imshow(shade, cmap="gray_r", origin="lower", aspect="auto")
    cbar = plt.colorbar(label="largest FM (magnitude)")
    plt.xlabel("$a$")
    plt.ylabel("$b$")

    # Manually rectify the ticks to remove the scaling/shifting:
    ticks = cbar.get_ticks()
    if logscale:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"10^{tick}" for tick in ticks])
    else:
        cbar.set_ticks(ticks - 1)
        cbar.set_ticklabels([f"{tick}" for tick in ticks])


def test_plotting_magnitudes():
    magnitudes = 10 ** (3 * np.random.rand(150, 150))

    for logscale in (True, False):
        plt.figure()
        plot_magnitude(magnitudes, logscale=logscale)

    plt.show()
