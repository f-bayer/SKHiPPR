import numpy as np
import pytest
from skhippr.systems.nonautonomous import Duffing
from skhippr.systems.autonomous import Vanderpol, Truss
from skhippr.problems.newton import NewtonSolver


def test_solution(ode_setting):
    # All considered test cases except Duffing have an equilibrium at zero
    _, ode = ode_setting

    if isinstance(ode, Duffing):
        ode.F = 0  # remove forcing

    solver = NewtonSolver(
        equations=[ode],
        unknowns=["x"],
        equation_determining_stability=None,
        verbose=True,
    )

    solver.solve()
    assert solver.converged


def test_stability(ode_setting):
    _, ode = ode_setting
    solver = NewtonSolver(
        equations=[ode],
        unknowns=["x"],
        equation_determining_stability=ode,
        verbose=True,
    )
    solver.solve()

    print(solver.stable)

    if isinstance(ode, Vanderpol):
        assert not solver.stable

    if isinstance(ode, Truss):
        if np.max(np.abs(ode.x)) > 0.1:
            assert ode.stable
        else:
            assert not ode.stable
