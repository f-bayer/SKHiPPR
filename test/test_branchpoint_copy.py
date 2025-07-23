import pytest
import numpy as np
from copy import copy
from skhippr.problems.newton import NewtonSolver
from skhippr.problems.continuation import BranchPoint
from skhippr.systems.AbstractSystems import AbstractEquation

""" Test parameter access in BranchPoint objects"""


class CircleExplicit(AbstractEquation):
    def __init__(self, x, radius, theta, other):
        super().__init__(stability_method=None)
        self.x = x
        self.radius = radius
        self.theta = theta
        self.other = other

    def residual_function(self):
        return np.array(
            [
                self.x[0] ** 2 + self.x[1] ** 2 - self.radius**2,
                self.x[1] * np.cos(self.theta) - self.x[0] * np.sin(self.theta),
            ]
        )

    def closed_form_derivative(self, variable):
        match variable:
            case "x":
                return np.array(
                    [
                        [2 * self.x[0], 2 * self.x[1]],
                        [-np.sin(self.theta), np.cos(self.theta)],
                    ]
                )
            case "radius":
                return np.array([[-2 * self.radius], [0]])
            case "theta":
                return np.array(
                    [
                        [0],
                        [
                            -self.x[1] * np.sin(self.theta)
                            - self.x[0] * np.cos(self.theta)
                        ],
                    ]
                )
            case "other":
                return self.other
            case "other2":
                return self.other[2]
            case _:
                raise NotImplementedError

@pytest.

@pytest.fixture
def circle(solver):
    theta = 0
    radius = 1
    circ = CircleExplicit(
        x=np.array([1.1, 0.1]), radius=radius, theta=theta, other=(100, 10, 0)
    )
    problem = NewtonSolver(
        residual_function=circle_explicit,
        initial_guess=np.array(),
        verbose=True,
        radius=radius,
        theta=theta,
        other=(100, 10, 0),
    )
    problem.solve()
    assert problem.converged
    return problem


def compare_attributes(sol1, sol2, attr, is_list=False):
    """Tests whether the sol1 and sol2 and its attributes are same (reference) and/or equal (value).
    Results are expected do differ for different ways of constructing sol1 and sol2 (see the tests).
    sol1, sol2 may be NewtonSolutions or BranchPoints."""

    same = sol1 is sol2
    same_attr = getattr(sol1, attr) is getattr(sol2, attr)
    equal_attr = np.all(getattr(sol1, attr) == getattr(sol2, attr))

    if is_list:
        getattr(sol1, attr)[0] = getattr(sol1, attr)[0] + 2
        attr_changed = getattr(sol1, attr)[0] == getattr(sol2, attr)[0]
    else:
        setattr(sol1, attr, getattr(sol1, attr) + 2)
        attr_changed = getattr(sol1, attr) == getattr(sol2, attr)

    same_attr_after_change = getattr(sol1, attr) is getattr(sol2, attr)

    return same, same_attr, equal_attr, attr_changed, same_attr_after_change


def test_solution_reference(prb):
    # Directly duplicate the reference - everything should always be the same
    sol2 = prb
    same, same_attr, equal_attr, attr_changed, same_attr_after_change = (
        compare_attributes(prb, sol2, "theta")
    )
    assert same
    assert same_attr
    assert equal_attr
    assert attr_changed
    assert same_attr_after_change


def test_solution_shallow_copy_theta(prb):
    # Shallow-copy the reference.
    # Changes to the (immutable) attribute directly overwrite.
    sol2 = copy(prb)
    same, same_attr, equal_attr, attr_changed, same_attr_after_change = (
        compare_attributes(prb, sol2, "theta", is_list=False)
    )
    assert not same
    assert same_attr
    assert equal_attr
    assert not attr_changed
    assert not same_attr_after_change


def test_solution_shallow_copy_x(prb):
    # Shallow-copy the reference.
    # Changes within mutable attributes persist.
    sol2 = copy(prb)
    same, same_attr, equal_attr, attr_changed, same_attr_after_change = (
        compare_attributes(prb, sol2, "x", is_list=True)
    )
    assert not same
    assert same_attr
    assert equal_attr
    assert attr_changed
    assert same_attr_after_change


def test_solution_shallow_copy_x_solve(prb):
    # Shallow-copy the reference.
    # The attribute x is overwritten during solve().

    prb.theta += 0.5
    prb.reset()
    sol2 = copy(prb)

    prb.solve()
    sol2.solve()

    same, same_attr, equal_attr, attr_changed, same_attr_after_change = (
        compare_attributes(prb, sol2, "x", is_list=True)
    )
    assert not same
    assert not same_attr
    assert equal_attr
    assert not attr_changed
    assert not same_attr_after_change


def test_branchpoint_reference(prb):
    # Directly duplicate the branch point - everything should always be the same
    bp = BranchPoint(problem=prb, key_param="theta", value_param=prb.theta)
    bp2 = bp
    same, same_attr, equal_attr, attr_changed, same_attr_after_change = (
        compare_attributes(bp, bp2, "theta", is_list=False)
    )
    assert same
    assert same_attr
    assert equal_attr
    assert attr_changed
    assert same_attr_after_change


def test_branchpoint_wrong_copy_theta(prb):
    # (Wrongly) generating a new branch point with reference to the same _problem
    # As attributes are deferred to problem, all changes are reflected in branch point.
    bp = BranchPoint(problem=prb, key_param="theta", value_param=prb.theta)
    bp2 = BranchPoint(problem=bp._problem, key_param="theta", value_param=prb.theta)
    same, same_attr, equal_attr, attr_changed, same_attr_after_change = (
        compare_attributes(bp, bp2, "theta", is_list=False)
    )
    assert not same
    assert same_attr
    assert equal_attr
    assert attr_changed
    assert same_attr_after_change


def test_branchpoint_shallow_copy_radius(prb):
    # Correctly generating a new branch point with reference to the copied _problem
    # Then, both BranchPoints behave like their _problems w.r.t non-continuation parameters.
    bp = BranchPoint(problem=prb, key_param="theta", value_param=prb.theta)
    bp2 = BranchPoint(
        problem=bp.copy_problem(), key_param="theta", value_param=prb.theta
    )
    same, same_attr, equal_attr, attr_changed, same_attr_after_change = (
        compare_attributes(bp, bp2, "radius", is_list=False)
    )
    assert not same
    assert same_attr
    assert equal_attr
    assert not attr_changed
    assert not same_attr_after_change


def test_branchpoint_shallow_copy_other(prb):
    # Correctly generating a new branch point with reference to the copied _problem
    # Then, both BranchPoints behave like their _problems w.r.t non-continuation parameters.
    prb.other = [1, 2, 3]
    bp = BranchPoint(problem=prb, key_param="theta", value_param=prb.theta)
    bp2 = BranchPoint(
        problem=bp.copy_problem(), key_param="theta", value_param=prb.theta
    )
    same, same_attr, equal_attr, attr_changed, same_attr_after_change = (
        compare_attributes(bp, bp2, "other", is_list=True)
    )
    assert not same
    assert same_attr
    assert equal_attr
    assert attr_changed
    assert same_attr_after_change


def test_branchpoint_shallow_copy_theta(prb):
    # Correctly generating a new branch point with reference to the copied _problem
    # x and parameter are overwritten (i.e., link is destroyed) during creation of BranchPoint (in NewtonSolution.__init__())
    bp = BranchPoint(problem=prb, key_param="theta", value_param=prb.theta)
    bp2 = BranchPoint(
        problem=bp.copy_problem(), key_param="theta", value_param=prb.theta
    )
    same, same_attr, equal_attr, attr_changed, same_attr_after_change = (
        compare_attributes(bp, bp2, "theta", is_list=False)
    )
    assert not same
    assert not same_attr
    assert equal_attr
    assert not attr_changed
    assert not same_attr_after_change


if __name__ == "__main__":
    pytest.main([__file__])
