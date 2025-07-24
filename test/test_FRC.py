"""Test FRC using the Duffing oscillator."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from copy import copy

from skhippr.problems.HBM import HBMSystem
from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic

from skhippr.problems.continuation import pseudo_arclength_continuator


@pytest.fixture
def initial_system(fourier, duffing_ode):
    omega_0 = 5
    ts = fourier.time_samples(omega_0)
    x0_samples = np.vstack((-np.cos(omega_0 * ts), omega_0 * np.sin(omega_0 * ts)))
    X0 = fourier.DFT(x0_samples)

    period_k, ode = duffing_ode
    ode.omega = omega_0

    return HBMSystem(
        ode=ode,
        omega=omega_0,
        fourier=fourier,
        initial_guess=X0,
        period_k=period_k,
        stability_method=None,
    )


def test_FRC_stability(initial_system, params_duffing):
    # Continuation along the Duffing from back to front. verify that the last 2 stability changes are folds.
    initial_system.stability_method = KoopmanHillSubharmonic(initial_system.fourier)
    initial_system.solve()
    assert initial_system.converged
    assert initial_system.stable

    # deque is a list that supports efficient appending and has a maximum length
    # https://docs.python.org/3/library/collections.html#collections.deque
    previous_bps = deque([initial_system] * 5, maxlen=5)
    num_stabchanges = 0

    plt.figure()

    for branch_point in pseudo_arclength_continuator(
        initial_problem=initial_system,
        stepsize=0.1,
        stepsize_range=(0.01, 0.15),
        key_param="omega",
        value_param=initial_system.omega,
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
def test_FRC_validity(solver, initial_system, initial_direction):
    omega = initial_system.equations[0].omega
    solver.solve(initial_system)
    assert initial_system.solved
    assert initial_system.equations[0].omega == omega

    num_steps = 3
    branch = []

    for branch_point in pseudo_arclength_continuator(
        initial_system=initial_system,
        solver=solver,
        stepsize=0.1,
        stepsize_range=(0.0001, 0.15),
        initial_direction=initial_direction,
        continuation_parameter="omega",
        verbose=True,
        num_steps=num_steps,
    ):
        branch.append(branch_point)

    # Verify stopping criterion
    assert branch_point.solved
    assert len(branch) == num_steps + 1

    for k, bp in enumerate(branch):
        assert bp.solved

        # Verify that parameters are correct and going in correct direction
        assert np.imag(bp.omega) < 1e-15
        if k == 0:
            assert bp.omega == omega
        else:
            assert np.real(initial_direction * (bp.omega - branch[k - 1].omega)) > 0

        # Verify that the point on the branch is indeed a solution, i.e., dx/dt == f

        # ode_copy = copy(initial_system.ode)
        # omega = bp.omega
        # ts = bp.fourier.time_samples(bp.omega)
        # xs = bp.x_time()
        # fs, _ = duffing(ts, xs, **params)
        # dx_dts = bp.fourier.differentiate(xs, omega=bp.omega)
        # dx_dts_approx = (xs[:, 1:] - xs[:, :-1]) / (ts[1] - ts[0])

        # assert np.allclose(fs, dx_dts, rtol=1e-6, atol=1e-6)
        # assert np.allclose(
        #     0.5 * (fs[:, 1:] + fs[:, :-1]), dx_dts_approx, rtol=1e-1, atol=1e-1
        # )


if __name__ == "__main__":
    pytest.main([__file__])
