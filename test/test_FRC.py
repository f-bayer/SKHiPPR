"""Test FRC using the Duffing oscillator."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from skhippr.problems.HBM import HBMEquation
from skhippr.systems.nonautonomous import duffing
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

from skhippr.problems.continuation import pseudo_arclength_continuator


@pytest.fixture
def initial_problem(fourier, params_duffing):
    params = params_duffing[1]  # period-1 solution
    omega_0 = 5
    params["omega"] = omega_0
    ts = fourier.time_samples(omega_0)
    x0_samples = np.vstack((-np.cos(omega_0 * ts), omega_0 * np.sin(omega_0 * ts)))
    X0 = fourier.DFT(x0_samples)

    return HBMEquation(
        f=duffing,
        initial_guess=X0,
        omega=params["omega"],
        fourier=fourier,
        stability_method=None,
        verbose=False,
        max_iterations=20,
        parameters_f=params,
    )


def test_FRC_stability(initial_problem, params_duffing):
    # Continuation along the Duffing from back to front. verify that the last 2 stability changes are folds.
    initial_problem.stability_method = KoopmanHillSubharmonic(initial_problem.fourier)
    initial_problem.solve()
    assert initial_problem.converged
    assert initial_problem.stable

    # deque is a list that supports efficient appending and has a maximum length
    # https://docs.python.org/3/library/collections.html#collections.deque
    previous_bps = deque([initial_problem] * 5, maxlen=5)
    num_stabchanges = 0

    plt.figure()

    for branch_point in pseudo_arclength_continuator(
        initial_problem=initial_problem,
        stepsize=0.1,
        stepsize_range=(0.01, 0.15),
        key_param="omega",
        value_param=initial_problem.omega,
        verbose=False,
        num_steps=1000,
        initial_direction=-1,
    ):
        previous_bps.append(branch_point)

        plt.plot(branch_point.omega, np.linalg.norm(branch_point.x[:-1]), ".")
        if previous_bps[-2].stable != previous_bps[-3].stable:
            num_stabchanges += 1
            delta_om_pre = previous_bps[-3].omega - previous_bps[-4].omega
            delta_om_post = previous_bps[-1].omega - previous_bps[-2].omega
            assert delta_om_pre * delta_om_post < 0

        assert (
            np.abs(np.imag(branch_point.omega)) < 1e-14
        ), f"It is expected that complex formulation fails here as frequency becomes cplx"
        assert np.linalg.norm(branch_point.x[:-1]) >= np.linalg.norm(
            previous_bps[-2].x[:-1]
        )
        if num_stabchanges > 1:
            break

    assert num_stabchanges == 2


@pytest.mark.parametrize("initial_direction", [1, -1, 2])
def test_FRC_validity(initial_problem, params_duffing, initial_direction):
    params = params_duffing[1]
    initial_problem.solve()
    assert initial_problem.converged
    assert initial_problem.omega == params["omega"]

    num_steps = 2
    branch = []

    for branch_point in pseudo_arclength_continuator(
        initial_problem=initial_problem,
        stepsize=0.1,
        stepsize_range=(0.0001, 0.15),
        key_param="omega",
        value_param=initial_problem.omega,
        verbose=False,
        num_steps=num_steps,
        initial_direction=initial_direction,
    ):
        branch.append(branch_point)

    # Verify stopping criterion
    assert branch_point.converged
    assert len(branch) == num_steps + 1

    for k, bp in enumerate(branch):
        assert bp.converged

        # Verify that parameters are correct and going in correct direction
        if k == 0:
            assert bp.omega == params["omega"]
        else:
            assert initial_direction * (bp.omega - branch[k - 1].omega) > 0

        # Verify that the point on the branch is indeed a solution, i.e., dx/dt == f
        params["omega"] = bp.omega
        ts = bp.fourier.time_samples(bp.omega)
        xs = bp.x_time()
        fs, _ = duffing(ts, xs, **params)
        dx_dts = bp.fourier.differentiate(xs, omega=bp.omega)
        dx_dts_approx = (xs[:, 1:] - xs[:, :-1]) / (ts[1] - ts[0])

        assert np.allclose(fs, dx_dts, rtol=1e-6, atol=1e-6)
        assert np.allclose(
            0.5 * (fs[:, 1:] + fs[:, :-1]), dx_dts_approx, rtol=1e-1, atol=1e-1
        )


if __name__ == "__main__":
    pytest.main([__file__])
