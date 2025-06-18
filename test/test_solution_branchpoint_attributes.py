import pytest
import numpy as np
from copy import copy
from skhippr.problems.newton import NewtonProblem
from skhippr.problems.continuation import BranchPoint


def circle_explicit(x, radius, theta, other):
    r = np.array([x[0] ** 2 + x[1] ** 2 - radius**2, x[1] - x[0] * np.tan(theta)])
    dr_dx = np.array([[2 * x[0], 2 * x[1]], [-np.tan(theta), 1]])
    dr_drad = np.array([-2 * radius, 0])
    dr_dtheta = np.array([0, 1 / (np.cos(theta) ** 2)])
    derivatives = {
        "x": dr_dx,
        "radius": dr_drad,
        "theta": dr_dtheta,
        "other": other,
        "other2": other[2],
    }

    return r, derivatives


@pytest.fixture
def prb():
    theta = 0
    radius = 1
    problem = NewtonProblem(
        residual_function=circle_explicit,
        initial_guess=np.array([1.1, 0.1]),
        verbose=True,
        radius=radius,
        theta=theta,
        other=[1, 2, 3],
    )
    return problem


@pytest.fixture
def bp(prb):
    return BranchPoint(problem=prb, key_param="theta", value_param=prb.theta)


@pytest.fixture(params=[{"use_problem": True}, {"use_problem": False}])
def setter(bp, request):
    if request.param["use_problem"]:
        return bp._problem
    else:
        return bp


"""Getter tests"""


def test_solution_getter(prb):
    radius = prb.radius
    x = prb.x


"""Setter tests"""


def test_solution_param_setter(prb):
    # Set new radius
    radius = prb.radius + 0.2
    prb.radius = radius
    assert prb.radius == radius

    # Assert that new radius is used in derivative
    assert not prb.converged
    prb.solve()
    assert prb.derivatives["radius"][0] == -2 * radius


def test_solution_param_mutable(prb):
    # Value to which the mutable parameter will be changed soon
    other2 = prb.other[2] + 2

    assert not prb.converged
    prb.solve()
    assert prb.derivatives["other"][2] != other2
    assert prb.derivatives["other2"] == prb.derivatives["other"][2]

    # Change 2nd entry of prb.other
    prb.other[2] = other2

    # derivatives['other'] is a reference to a mutable array --> it changes with prb.other
    assert prb.derivatives["other"][2] == other2

    # derivatives['other'] is an immutable scalar that changes only when solve() is called again
    assert prb.derivatives["other2"] != other2

    prb.reset()
    prb.solve()
    assert prb.derivatives["other2"] == other2


def test_solution_x_setter(prb):
    # Set new radius
    prb.solve()
    x = prb.x + 0.2
    prb.x = x
    assert (prb.x == x).all()


def test_solution_x_mutable(prb):
    # Set new radius
    prb.solve()
    x0 = prb.x[0] + 0.2
    prb.x[0] = x0
    assert prb.x[0] == x0


def test_branchpoint_other_param_setter(bp, setter):

    # Set new radius
    radius = bp.radius + 0.2
    setter.radius = radius
    assert bp.radius == radius
    assert bp._problem.radius == radius

    # Assert that new radius is used in function
    assert not bp.converged
    bp.solve()
    assert bp.derivatives["radius"][0] == -2 * radius


def test_branchpoint_contin_param_setter(bp, setter):

    # Set new theta
    theta = bp.theta + 0.2
    setter.theta = theta
    assert bp.theta == theta
    assert bp._problem.theta == theta

    # Assert that new theta is used in function
    assert not bp.converged
    bp.solve()
    assert bp.derivatives["x"][1, 0] == -np.tan(theta)


def test_branchpoint_x_param_setter(bp, setter):

    # Set new x
    x = setter.x + 0.2
    setter.x = x
    assert (bp.x[: len(x)] == x).all()
    assert (bp._problem.x == x[: len(bp.x) - 1]).all()


def test_branchpoint_other_param_mutable(bp, setter):
    # Value to which the mutable parameter will be changed soon
    other2 = bp.other[2] + 2
    other = np.copy(bp.other)
    assert other is not bp.other
    other[2] = other2

    assert not bp.converged
    bp.solve()
    assert bp.derivatives["other"][2] != other2
    assert bp.derivatives["other2"] == bp.derivatives["other"][2]

    # Change 2nd entry of setter.other
    setter.other[2] = other2
    assert (bp.other == other).all()
    assert (bp._problem.other == other).all()

    # derivatives['other'] is a reference to a mutable array --> it changes with prb.other
    assert bp.derivatives["other"][2] == other2

    # derivatives['other'] is an immutable scalar that changes only when solve() is called again
    assert bp.derivatives["other2"] != other2

    bp.reset()
    bp.solve()
    assert bp.derivatives["other2"] == other2


def test_branchpoint_x_mutable(bp, setter):
    # Value to which the mutable parameter will be changed soon
    bp.solve()
    x0 = bp.x[0] + 2
    assert bp.x[0] != x0

    # Change x
    setter.x[0] = x0

    if isinstance(setter, BranchPoint):
        # bp.x returns a copy every time to append continuation parameter! --> Mutating is NOT possible!
        assert bp.x[0] != x0
        assert bp._problem.x[0] != x0
    else:
        # derivatives['other'] is a reference to a mutable array --> it changes with prb.other
        assert bp.x[0] == x0
        assert bp._problem.x[0] == x0
