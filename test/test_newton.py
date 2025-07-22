import pytest
import numpy as np

from skhippr.problems.newton import NewtonSolver
from skhippr.systems.AbstractSystems import AbstractEquationSystem
from skhippr.stability._StabilityMethod import StabilityEquilibrium


class TestEquationY(AbstractEquationSystem):
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


class TestEquationB(AbstractEquationSystem):
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
def setup():
    params = {
        "tolerance": 1e-8,
        "max_iter": 20,
        "a_ref": 3.0,
        "y_0": [0.0, 0.0],
        "y_sol": [1.0, 1.0],
        "b_0": 3,
    }

    equ_y = TestEquationY(y=params["y_0"], a=params["a_ref"])
    equ_b = TestEquationB(a=params["a_ref"], b=params["b_0"])
    return params, equ_y, equ_b


def test_solver_constructor(setup):
    params, equ_y, equ_b = setup
    solver = NewtonSolver(
        [equ_y, equ_b],
        ["y", "a"],
        stability_method=None,
        tolerance=params["tolerance"],
        max_iterations=params["max_iter"],
        verbose=True,
    )
    assert not solver.converged
    assert solver.a == params["a_ref"]


@pytest.mark.parametrize("num_eqs", [1, 2])
def test_solve_one_eq(setup, num_eqs):
    params, equ_y, equ_b = setup
    eqs = [equ_y]
    unknowns = ["y"]
    if num_eqs == 2:
        eqs.append(equ_b)
        unknowns.append("b")

    solver = NewtonSolver(
        eqs,
        unknowns,
        stability_method=None,
        tolerance=params["tolerance"],
        max_iterations=params["max_iter"],
        verbose=True,
    )
    assert not solver.converged
    solver.solve()
    assert solver.converged

    residual_expected = np.zeros(1 + num_eqs)
    solution_expected = np.array([1.0, 1.0])
    if num_eqs == 2:
        solution_expected = np.append(solution_expected, params["a_ref"])

    assert np.allclose(
        solver.residual_function(update=False),
        residual_expected,
        atol=params["tolerance"],
    )
    assert np.allclose(solver.vector_of_unknowns, solution_expected, atol=1e-4)
    # assert not prb.stable


def test_reset(setup):
    params, eq_y, eq_b = setup
    solver = NewtonSolver(
        [eq_y, eq_b],
        ["y", "b"],
        stability_method=None,
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
        stability_method=None,
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
        stability_method=None,
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
