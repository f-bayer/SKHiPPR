from tkinter import Y
import pytest
import numpy as np

from skhippr.problems.newton import NewtonProblem
from skhippr.stability._StabilityMethod import StabilityEquilibrium


@pytest.fixture
def params():
    return {
        "tol": 1e-8,
        "max_iter": 20,
        "a_ref": 3.0,
    }


@pytest.fixture
def prb(params):
    tol = params["tol"]
    max_iter = params["max_iter"]
    a_ref = params["a_ref"]

    x0 = np.array([-3.0, -5.0])

    return NewtonProblem(
        residual_function=residual_function,
        initial_guess=x0,
        variable="y",
        stability_method=StabilityEquilibrium(n_dof=2),
        tolerance=tol,
        max_iterations=max_iter,
        verbose=False,
        a=a_ref,
    )


def residual_function(y, *, a):
    residual = np.array([y[0] ** 2 - 2 * y[1] + 1, y[0] - y[1]])
    jacobian = np.array([[2 * y[0], -2], [1, -1]])
    return residual, {"y": jacobian, "a": a}


def test_problem_constructor(prb, params):
    # This test must always run first because prb is modified / solved by other tests
    assert not prb.converged
    assert prb.a == params["a_ref"]


def test_solve(prb, params):
    prb.solve()
    assert prb.converged
    assert np.allclose(
        prb.residual_function(recompute=False), np.array([0, 0]), atol=params["tol"]
    )
    assert np.allclose(prb.unknowns, np.array([1, 1]), atol=1e-4)
    assert not prb.stable


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
    assert all(res == residual_function(y=y_new, a=prb.a)[0])


if __name__ == "__main__":
    pytest.main([__file__])
