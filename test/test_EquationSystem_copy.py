import pytest
import numpy as np
from copy import copy
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.solvers.continuation import BranchPoint
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
        return np.squeeze(
            np.array(
                [
                    self.x[0] ** 2 + self.x[1] ** 2 - self.radius**2,
                    self.x[1] * np.cos(self.theta) - self.x[0] * np.sin(self.theta),
                ]
            )
        )

    def closed_form_derivative(self, variable):
        match variable:
            case "x":
                return np.array(
                    [
                        [2 * self.x[0], 2 * self.x[1]],
                        [
                            -np.sin(np.squeeze(self.theta)),
                            np.cos(np.squeeze(self.theta)),
                        ],
                    ]
                )
            case "radius":
                radius = np.atleast_1d(self.radius)
                return np.array([-2 * radius, [0]])
            case "theta":
                theta = np.atleast_1d(self.theta)
                return np.array(
                    [
                        [0],
                        -self.x[1] * np.sin(self.theta)
                        - self.x[0] * np.cos(self.theta),
                    ]
                )
            case "other":
                return self.other
            case "other2":
                return self.other[2]
            case _:
                raise NotImplementedError


@pytest.fixture
def circle(solver):
    theta = 0
    radius = 1
    circ = CircleExplicit(
        x=np.array([1.1, 0.1]), radius=radius, theta=theta, other=(100, 10, 0)
    )
    circle_sys = EquationSystem([circ], ["radius", "theta"], None)

    solver.solve(circle_sys)
    assert circle_sys.solved
    return circle_sys


def compare_attributes(sys1, sys2, attr, mutable=False):
    """Tests whether the sol1 and sol2 and its attributes are same (reference) and/or equal (value).
    Results are expected do differ for different ways of constructing sol1 and sol2 (see the tests).
    sol1, sol2 may be NewtonSolutions or BranchPoints."""

    same = sys1 is sys2
    same_attr = getattr(sys1, attr) is getattr(sys2, attr)
    equal_attr = np.all(getattr(sys1, attr) == getattr(sys2, attr))

    if mutable:
        getattr(sys1, attr)[0] = getattr(sys1, attr)[0] + 2
        attr_changed = getattr(sys1, attr)[0] == getattr(sys2, attr)[0]
        attr_eq_changed = (
            getattr(sys1.equations[0], attr)[0] == getattr(sys2.equations[0], attr)[0]
        )
    else:
        setattr(sys1, attr, getattr(sys1, attr) + 2)
        attr_changed = getattr(sys1, attr) == getattr(sys2, attr)
        attr_eq_changed = getattr(sys1.equations[0], attr) == getattr(
            sys2.equations[0], attr
        )

    return same, same_attr, equal_attr, attr_changed, attr_eq_changed


def test_no_copy(circle):
    # Directly duplicate the reference - everything should always be the same
    circle2 = circle
    same, same_attr, equal_attr, attr_changed, attr_eq_changed = compare_attributes(
        circle, circle2, "theta", mutable=False
    )
    assert same
    assert same_attr
    assert equal_attr
    assert attr_changed
    assert attr_eq_changed


def test_copy(circle):
    # Shallow-copy the reference using copy.copy() (not recommended)
    # Changes to the (immutable) attribute directly overwrite.
    circle2 = copy(circle)
    same, same_attr, equal_attr, attr_changed, attr_eq_changed = compare_attributes(
        circle, circle2, "theta", mutable=False
    )
    assert not same
    assert same_attr
    assert equal_attr
    assert not attr_changed  # because changes in the equations do not reflect "upwards"
    assert attr_eq_changed


def test_duplicate(circle):
    # Duplicate the reference properly using its method, shallow-copying the equations.
    circle2 = circle.duplicate()
    same, same_attr, equal_attr, attr_changed, attr_eq_changed = compare_attributes(
        circle, circle2, "theta", mutable=False
    )
    assert not same
    assert same_attr
    assert equal_attr
    assert not attr_changed
    assert not attr_eq_changed


def test_duplicate_mutable(circle):
    # Duplicating retains references of the equations to the same mutable objects
    circle = EquationSystem(circle.equations, ["x"], None)
    circle2 = circle.duplicate()
    same, same_attr, equal_attr, attr_changed, attr_eq_changed = compare_attributes(
        circle, circle2, "x", mutable=True
    )
    assert not same
    assert same_attr
    assert equal_attr
    assert attr_changed
    assert attr_eq_changed


@pytest.mark.parametrize("mutable", [False, True])
def test_solve(solver, circle, mutable):
    # Shallow-copy the reference.
    # The attribute x is overwritten during solve().

    if mutable:
        circle = EquationSystem(circle.equations, ["x"], None)
        solver.solve(circle)

    assert circle.solved
    try:
        circle.theta += 0.5
        assert not circle.solved
    except AttributeError:
        # theta is not part of the unknowns
        circle.equations[0].theta += 0.5
        circle.solved = False

    circle2 = circle.duplicate()
    solver.solve(circle)

    assert not circle2.solved
    solver.solve(circle2)
    assert circle2.solved

    if mutable:
        attr = "x"
    else:
        attr = "theta"

    same, same_attr, equal_attr, attr_changed, same_attr_after_change = (
        compare_attributes(circle, circle2, attr, mutable=mutable)
    )
    assert not same
    assert not same_attr
    assert equal_attr
    assert not attr_changed
    assert not same_attr_after_change


if __name__ == "__main__":
    pytest.main([__file__])
