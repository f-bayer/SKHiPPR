"""Compare the stability assertions of all stability methods"""

import pytest
import numpy as np
import itertools
import matplotlib.pyplot as plt

from skhippr.Fourier import Fourier

from skhippr.problems.HBM import HBMProblem, HBMProblem_autonomous
from skhippr.problems.shooting import ShootingProblem

from skhippr.systems.nonautonomous import duffing
from skhippr.systems.autonomous import vanderpol

from skhippr.stability.KoopmanHillProjection import (
    KoopmanHillProjection,
    KoopmanHillSubharmonic,
)
from skhippr.stability.ClassicalHill import ClassicalHill
from skhippr.stability.SinglePass import SinglePassRK4, SinglePassRK38


@pytest.mark.parametrize(
    "autonomous,stability_method",
    [
        (False, KoopmanHillProjection),
        (False, KoopmanHillSubharmonic),
        (False, lambda fourier: ClassicalHill(fourier, sorting_method="imaginary")),
        # (False, lambda fourier: ClassicalHill(fourier, sorting_method="symmetry")),
        (False, SinglePassRK4),
        (False, lambda fourier: SinglePassRK38(fourier, stepsize=1e-3)),
        (True, KoopmanHillProjection),
        (True, KoopmanHillSubharmonic),
        (True, lambda fourier: ClassicalHill(fourier, sorting_method="imaginary")),
        # (True, lambda fourier: ClassicalHill(fourier, sorting_method="symmetry")),
        (True, SinglePassRK4),
        (True, lambda fourier: SinglePassRK38(fourier, stepsize=1e-3)),
    ],
)
def test_stability(
    params_duffing, autonomous, stability_method, fourier, visualize=False
):

    if autonomous:
        f = vanderpol
        params = {"nu": 0.5}
        omega = 1
        x0 = np.array([2.0, 0])

    else:
        f = duffing
        omega = 0.4
        params = params_duffing[1]
        x0 = np.array([1.0, 0.0])

    ode_kwargs = {"rtol": 1e-7, "atol": 1e-7}

    # Reference: Floquet multipliers obtained using Shooting
    sol_shooting = ShootingProblem(
        f=f,
        x0=x0,
        T=2 * np.pi / omega,
        autonomous=autonomous,
        kwargs_odesolver=ode_kwargs,
        parameters=params,
    )
    sol_shooting.solve()
    assert sol_shooting.converged
    floquet_multipliers_shooting = sol_shooting.eigenvalues

    # determine periodic solution using HBM
    method = stability_method(fourier)
    x_shooting = sol_shooting.x_time(fourier.time_samples(sol_shooting.omega))
    X0 = fourier.DFT(x_shooting)

    if autonomous:
        factory_HBM = HBMProblem_autonomous
    else:
        factory_HBM = HBMProblem

    sol_HBM = factory_HBM(
        f=f,
        initial_guess=X0,
        omega=omega,
        fourier=fourier,
        parameters_f=params,
        stability_method=method,
    )
    sol_HBM.solve()
    assert sol_HBM.converged
    floquet_multipliers = sol_HBM.eigenvalues

    # Account for the fact that the arrays of FMs might be sorted differently
    for k, idx in enumerate(itertools.permutations(range(len(floquet_multipliers)))):
        fm_reordered = floquet_multipliers[idx,]
        fm_equal = np.allclose(floquet_multipliers_shooting, fm_reordered, atol=1e-5)
        if fm_equal:
            break
    # assert fm_equal

    if visualize:
        print(method)
        print(floquet_multipliers_shooting)
        print(floquet_multipliers)
        if fm_equal:
            print(f"Same as shooting method (in {k}-th permutation).")
        else:
            print("Different from Shooting method!")

        plt.figure()
        phi = np.linspace(0, 2 * np.pi, 250)
        plt.plot(np.cos(phi), np.sin(phi), "k")

        plt.plot(
            np.real(floquet_multipliers_shooting),
            np.imag(floquet_multipliers_shooting),
            "x",
            label="shooting",
        )
        plt.title(str(method))
        plt.axis("equal")

        plt.plot(
            np.real(floquet_multipliers),
            np.imag(floquet_multipliers),
            ".",
            label=method.label,
        )

        plt.legend(loc="upper right")


if __name__ == "__main__":
    pytest.main([__file__])
