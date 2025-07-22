import pytest
import numpy as np

from skhippr.problems.newton import NewtonSolver, EquationSystem
from skhippr.systems.AbstractSystems import AbstractEquation
from skhippr.stability._StabilityMethod import StabilityEquilibrium


class TestEquationY(AbstractEquation):
    def __init__(self, y, a):
        super().__init__()
        self.y = y
        self.a = a

    def residual_function(self):
        return np.array([self.y[0] ** 2 - 2 * self.y[1] + 1, self.y[0] - self.y[1]])

    def closed_form_derivative(self, variable):
        match variable:
            case "y":
                return np.array([[2 * self.y[0], -2], [1, -1]])
            case "a":
                return self.a
            case "b":
                return np.array([[0], [0]])
            case _:
                raise NotImplementedError


class TestEquationB(AbstractEquation):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def residual_function(self):
        return np.atleast_1d(self.b - self.a)

    def closed_form_derivative(self, variable):
        match variable:
            case "b":
                return np.atleast_2d(1)
            case "a":
                return np.atleast_2d(-1)
            case "y":
                return np.array([[0, 0]])
            case _:
                raise NotImplementedError


@pytest.fixture
def solver():
    return NewtonSolver(tolerance=1e-8, max_iterations=20, verbose=True)


@pytest.fixture
def equation_system():
    a_ref = 3.0
    equ_y = TestEquationY(y=np.array([0.0, 0.0]), a=3.0)
    equ_b = TestEquationB(a=a_ref, b=4.0)

    eq_sys = EquationSystem(
        [equ_y, equ_b], ["y", "b"], equation_determining_stability=None
    )
    return eq_sys


@pytest.mark.parametrize("num_eqs", [1, 2])
def test_solve(num_eqs, solver, equation_system):

    if num_eqs == 1:
        equ_y = equation_system.equations[0]
        equation_system = EquationSystem([equ_y], ["y"])

    assert equation_system.well_posed

    assert not equation_system.solved
    solver.solve(equation_system)
    assert equation_system.solved

    residual_expected = np.zeros(1 + num_eqs)
    solution_expected = np.array([1.0, 1.0])
    if num_eqs == 2:
        solution_expected = np.append(solution_expected, 3.0)

    assert np.allclose(
        equation_system.residual_function(update=False),
        residual_expected,
        atol=solver.tolerance,
    )
    assert np.allclose(equation_system.vector_of_unknowns, solution_expected, atol=1e-4)


def test_badly_posed(equation_system):
    equ_y = equation_system.equations[0]
    equ_b = equation_system.equations[1]

    # Well posed
    assert equation_system.well_posed

    equation_system = EquationSystem([equ_y], ["y"])
    assert equation_system.well_posed

    equation_system = EquationSystem([equ_b], ["a"])
    assert equation_system.well_posed

    # Too many variables
    equation_system = EquationSystem([equ_y, equ_b], ["y", "b", "a"])
    assert not equation_system.well_posed

    equation_system = EquationSystem([equ_y], ["y", "a"])
    assert not equation_system.well_posed

    # Not enough variables
    equation_system = EquationSystem([equ_y], ["a"])
    assert not equation_system.well_posed

    # unknowns which are not variables
    with pytest.raises(ValueError):
        equation_system = EquationSystem([equ_y], ["c"])


def test_reset(setup):
    params, eq_y, eq_b = setup
    solver = NewtonSolver(
        [eq_y, eq_b],
        ["y", "b"],
        equation_determining_stability=None,
        tolerance=params["tolerance"],
        max_iterations=params["max_iter"],
        verbose=True,
    )
    solver.solve()
    assert solver.converged
    assert solver.num_iter > 0

    # Perform manual reset
    y_new = np.array([5.0, 0.0, 2.0])
    solver.reset(y_new)
    assert not solver.converged
    assert solver.num_iter == 0

    # Check all the properties
    assert np.array_equal(solver.y, y_new[:2])
    assert solver.b == y_new[2]

    for eq in solver.equations:
        assert np.array_equal(eq.y, y_new[:2])
        assert eq.b == y_new[2]

    solver.solve()
    assert solver.converged


def test_solver_getter(setup):
    # Ensure that the custom getter does not lead to a recursion loop
    params, eq_y, eq_b = setup
    solver = NewtonSolver(
        [eq_y, eq_b],
        ["y", "b"],
        equation_determining_stability=None,
        tolerance=params["tolerance"],
        max_iterations=params["max_iter"],
        verbose=True,
    )
    solver.solve()
    assert solver.converged
    assert solver.num_iter > 0

    b = solver.b
    y = solver.y
    label = solver.label

    with pytest.raises(AttributeError):
        val = solver.this_attribute_does_not_exist


def test_solution_setter(setup):
    params, eq_y, eq_b = setup
    solver = NewtonSolver(
        [eq_y, eq_b],
        ["y", "b"],
        equation_determining_stability=None,
        tolerance=params["tolerance"],
        max_iterations=params["max_iter"],
        verbose=True,
    )
    solver.solve()
    assert solver.converged
    assert solver.num_iter > 0

    solver.useless_attribute = 1
    # Check that arbitrary attributes are NOT transferred to the equations
    for eq in solver.equations:
        with pytest.raises(AttributeError):
            val = eq.useless_attribute

    # Check that attribute updates directly to the equations do not transfer (how should they)
    solver.equations[0].b = 3
    solver.equations[1].b = 4
    assert solver.b != solver.equations[1].b != solver.equations[0].b

    # Check that updates to solver unknown transfer to all equations
    solver.b = 5
    assert solver.b == solver.equations[1].b == solver.equations[0].b
    assert solver.vector_of_unknowns[-1] == solver.equations[0].b

    # Check that updates to solver for non-unknowns (which do exist) do not transfer to equations
    solver.a = 42
    assert solver.equations[0].a != solver.a != solver.equations[1].a

    # Check that updates to unknowns are reflected in all equations
    solver.vector_of_unknowns = np.append(solver.vector_of_unknowns[:-1], 1)
    assert 1 == solver.b == solver.equations[1].b == solver.equations[0].b


if __name__ == "__main__":
    pytest.main([__file__])
