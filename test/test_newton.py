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
                raise NotImplementedError
                return np.array([[2 * self.y[0], -2], [1, -1]])
            case "a":
                return self.a
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
                return np.atleast_1d(1)
            case "a":
                return np.atleast_1d(-1)
            case "y":
                return np.array([0, 0])
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


def test_solve_one_eq(setup):
    params, equ_y, _ = setup
    solver = NewtonSolver(
        [equ_y],
        ["y"],
        stability_method=None,
        tolerance=params["tolerance"],
        max_iterations=params["max_iter"],
        verbose=True,
    )
    assert not solver.converged
    solver.solve()
    assert solver.converged
    assert np.allclose(
        solver.residual_function(update=False),
        np.array([0, 0]),
        atol=params["tolerance"],
    )
    assert np.allclose(solver.vector_of_unknowns, np.array([1, 1]), atol=1e-4)
    # assert not prb.stable


def test_reset(prb):
    prb.solve()
    assert prb.converged
    assert prb.num_iter > 0

    # Perform manual reset
    y_new = np.array([5.0, 0.0])
    prb.reset(x0_new=y_new)
    assert not prb.converged
    assert prb.num_iter == 0
    prb.solve()
    assert prb.converged

    # Expect that reset is performed if component of unknowns is manually updated
    prb.y = np.array(y_new)
    assert not prb.converged
    assert prb.num_iter == 0
    prb.solve()
    assert prb.converged

    # Expect that reset is performed if parameter is manually updated
    prb.a = 3
    assert not prb.converged
    assert prb.num_iter == 0
    prb.solve()
    assert prb.converged


def test_solution_getter(prb):
    # Ensure that the custom getter does not lead to a recursion loop
    a = prb.a
    y = prb.y
    label = prb.label
    try:
        b = prb.this_attribute_does_not_exist
        assert False, f"Expected an AttributeError, but got {b}"
    except AttributeError:
        # expected behavior
        pass


def test_solution_setter(prb, params):
    # Ensure that the custom setter works
    prb.useless_attribute = 1

    a_update = 15
    assert a_update != params["a_ref"]

    prb.solve()
    prb.reset()
    assert not prb.converged
    assert prb.a == params["a_ref"]
    assert prb.jacobians_dict[(prb.f_with_params, "a")] == params["a_ref"]

    prb.a = a_update
    # Ensure that new parameter value is passed into function
    _, derivatives = prb.f_with_params(prb.unknowns)
    assert derivatives["a"] == a_update

    # Ensure that updated parameter is used during function calls of solve()
    prb.solve()
    assert prb.jacobians_dict[(prb.f_with_params, "a")] == a_update

    # Check if update of unknowns works as expected
    assert prb.converged
    y_new = np.array([15.0, 10.0])
    assert all(y_new != prb.y)
    prb.y = y_new
    assert all(prb.y == y_new)
    assert all(prb.unknowns == y_new)
    res = prb.residual_function(recompute=True)
    assert all(res == my_function(y=y_new, a=prb.a)[0])


if __name__ == "__main__":
    pytest.main([__file__])
