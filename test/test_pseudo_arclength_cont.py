import pytest
import numpy as np
import matplotlib.pyplot as plt

from skhippr.problems.newton import NewtonProblem
from skhippr.problems.continuation import pseudo_arclength_continuator
from skhippr.systems.autonomous import truss
from skhippr.stability._StabilityMethod import StabilityEquilibrium


@pytest.fixture
def truss_params():
    params = {
        "a": 1,
        "l_0": 1.2,
        "m": 1,
        "k": 3,
        "c": 0.01,
        "F": -2,
    }
    return params


def circle(y: np.ndarray, radius=1) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # Residual must be returned as 1D numpy array
    residual = np.atleast_1d(y[0] ** 2 + y[1] ** 2 - radius**2)
    dr_dy = np.atleast_2d(np.array([2 * y[0], 2 * y[1]]))
    dr_dradius = np.atleast_1d(-2 * radius)  # optional

    # Derivatives are returned as a dictionary. The keys correspond to the function arguments.
    # Derivatives w.r.t everything else than y are optional and are only needed for explicit parameter continuation.
    derivatives = {"y": dr_dy, "radius": dr_dradius}
    return residual, derivatives


def test_cont_circle(visualize=False):
    radius = 1.5
    y0 = [0.9 * radius, 0]
    initial_guess = NewtonProblem(
        residual_function=circle,
        initial_guess=y0,
        variable="y",
        tolerance=1e-8,
        verbose=False,
        max_iterations=20,
        stability_method=StabilityEquilibrium(n_dof=2),
        radius=radius,
    )
    branch = []
    num_steps = 1000
    for branch_point in pseudo_arclength_continuator(
        initial_problem=initial_guess,
        stepsize=0.05,
        stepsize_range=(0.01, 0.1),
        initial_direction=1,
        key_param=None,
        verbose=visualize,
        num_steps=num_steps,
    ):
        branch.append(branch_point)
        assert np.allclose(np.linalg.norm(branch_point.x), radius)

        if len(branch) > 1:
            assert np.linalg.norm(branch_point.x - branch[-2].x) > 0.008

            # End condition
            if branch[-2].x[-1] < 0 and branch_point.x[-1] >= 0:
                # continue with different radius
                radius += 1
                branch_point.radius = radius

    # Check number of steps
    assert (
        len(branch) == num_steps + 1
    ), f"Expected {num_steps+1} branch points (including initial guess), but got {len(branch)}"

    if visualize:
        plt.figure()
        plt.plot(*np.array([branch_point.x for branch_point in branch]).T)


def test_cont_truss(truss_params, verbose=False):

    def sys(x, **params):
        return truss(0, x, **params)

    x0 = np.array([-1.0, 0.0])
    initial_guess = NewtonProblem(
        residual_function=sys,
        initial_guess=x0,
        variable="x",
        tolerance=1e-8,
        verbose=False,
        max_iterations=20,
        stability_method=StabilityEquilibrium(n_dof=2),
        **truss_params,
    )

    initial_guess.solve()
    assert initial_guess.converged
    if verbose:
        print(initial_guess)

    interval_F = [-2, 2]

    branch = []
    xs_pred = []
    xs = []
    Fs = []
    stable = []

    if verbose:
        plt.figure()

    for branch_point in pseudo_arclength_continuator(
        initial_problem=initial_guess,
        stepsize=0.03,
        stepsize_range=(0.001, 0.2),
        initial_direction=1,
        num_steps=300,
        key_param="F",
        value_param=interval_F[0],
        verbose=verbose,
    ):
        branch.append(branch_point)
        xs_pred.append(branch_point.x0)
        xs.append(branch_point.x)
        Fs.append(branch_point.F)
        stable.append(branch_point.stable)

        branch_point.determine_tangent()
        x_tng = branch_point.x + 0.2 * branch_point.tangent
        if verbose:
            plt.plot(
                (branch_point.x[-1], x_tng[-1]), (branch_point.x[0], x_tng[0]), "gray"
            )

        if branch_point.F < interval_F[0] or branch_point.F > interval_F[1]:
            break
    xs_pred = np.array(xs_pred).T
    xs = np.array(xs).T
    Fs = np.array(Fs)
    stable = np.array(stable)
    unstable = np.invert(stable)

    F_stbl = Fs.copy()
    F_stbl[unstable] = np.nan
    F_ustbl = Fs.copy()
    F_ustbl[stable] = np.nan

    if verbose:
        plt.plot(xs_pred[-1, :], xs_pred[0, :], "g1", label="predictor")
        plt.plot(F_stbl, xs[0, :], "rx", label="stable")
        plt.plot(F_ustbl, xs[0, :], "b+", label="unstable")
        plt.title("Pseudo-arclength results")
        plt.legend()

    """Verify stability assertions are correct at start, middle, end"""
    assert stable[0]
    assert stable[-1]
    idx_F_0 = np.argmin(np.abs(Fs))
    assert not stable[idx_F_0]

    """Verify that the stability changes are saddle nodes: F turns around near a stab. change"""
    idx_stabchange = [
        k for k in range(2, len(stable) - 1) if stable[k] != stable[k - 1]
    ]
    assert len(idx_stabchange) == 2
    for k in idx_stabchange:
        assert (Fs[k - 1] - Fs[k - 2]) * (Fs[k + 1] - Fs[k]) < 0

    """ Ensure that the custom getter and setter of BranchPoint do not lead to recursions"""
    val_a = branch[-1].a
    assert val_a == truss_params["a"]

    branch[-1].a = 100
    assert branch[-1].a == 100
    assert truss_params["a"] != 100
    assert branch[-2].a == truss_params["a"]

    branch_point.useless_attribute = 3
    val_attr = branch_point.useless_attribute
    assert val_attr == 3
    try:
        val_attr = branch[-2].useless_attribute
        assert False, "branch[-2].useless_attribute should raise an AttributeError"
    except AttributeError:
        # expected behavior
        pass


if __name__ == "__main__":
    params = {"a": 1, "l_0": 1.2, "m": 1, "k": 3, "c": 0.01, "F": -2}
    test_cont_truss(params, sparsity=False, verbose=True)
    test_cont_circle(visualize=True)
    plt.show()
