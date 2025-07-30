"""Demonstration example for the pseudo-arclength continuation workflow with no explicit parameter, i.e., with one more equation than unknowns."""

import numpy as np
import matplotlib.pyplot as plt

from skhippr.solvers.newton import NewtonSolver
from skhippr.solvers.continuation import pseudo_arclength_continuator
from skhippr.equations.Circle import CircleEquation, AngleEquation
from skhippr.equations.EquationSystem import EquationSystem


def main():
    """
    Runs a demonstration of pseudo-arclength continuation for solving a nonlinear problem
    using Newton's method on the circle equation.
    The process includes:

    #. Instantiating a :py:class:`~skhippr.cycles.newton.NewtonProblem` with the initial guess and the :py:func:`circle` function.
    #. Using the :py:func:`~skhippr.cycles.continuation.pseudo_arclength_continuator` to iterate along the solution branch.
    #. Plotting predictors, corrected solutions, and tangents at each :py:class:`~skhippr.cycles.continuation.BranchPoint`.
    #. After one circle is completed, ``radius`` is dynamically updated.
    #. Finally, the resulting branches are displayed using ``matplotlib``.

    Returns
    -------
    None
    """
    print("Circle example.")

    # Instantiate solver
    solver = NewtonSolver()

    # Instantiate equation to be solved
    radius = 2
    equ = CircleEquation(y=np.array([0.9 * radius, 0]), radius=radius)

    solve_equation_for_radius(equ, solver)
    attempt_to_solve_equation_for_y(equ, solver)

    # Instantiate EquationSystem encoding the circle eq. and the unknown
    equ_sys = EquationSystem([equ], unknowns=["y"])

    # We need a slightly different syntax but still, we cannot solve this immediately...
    attempt_to_solve_system_for_y(equ_sys, solver)

    # But we can perform continuation to find multiple solutions of the equations system!
    continuation_and_plot(equ_sys=equ_sys, solver=solver, continuation_parameter=None)

    # Make the equation system well-defined by appending an equation to fix the angle
    ang = AngleEquation(y=equ.y, theta=0)
    equ_sys_with_angle = EquationSystem([equ, ang], unknowns=["y"])
    attempt_to_solve_system_for_y(equ_sys_with_angle, solver)

    # And again, we can do continuation - but now a parameter must act as (explicit) continuation parameter
    continuation_and_plot(
        equ_sys=equ_sys_with_angle,
        solver=solver,
        continuation_parameter="theta",
        param_max=2 * np.pi,
    )

    equ_sys_with_angle.equations[1].theta = np.pi / 4
    continuation_and_plot(
        equ_sys=equ_sys_with_angle,
        solver=solver,
        continuation_parameter="radius",
        param_max=8,
    )


def solve_equation_for_radius(equ, solver):

    radius_old = equ.radius
    # Verify that the provided initial conditions do not solve the equation
    print(
        f"Initial guess does not solve the equation: r = {equ.radius} != {np.linalg.norm(equ.y)} = ||y||"
    )

    # We can solve immediately for the radius corresponding to a fixed y
    solver.solve_equation(equ, "radius")
    print(f"Solved for radius: r = {equ.radius} == {np.linalg.norm(equ.y)} = ||y||")
    # Note that scalar unknowns are transformed into 1-D numpy arrays.

    # Revert to original setup
    equ.radius = radius_old


def attempt_to_solve_equation_for_y(equ, solver):

    # We can NOT solve immediately for y because the number of equations does not match the number of unknowns
    try:
        solver.solve_equation(equ, "y")
    except ValueError as ME:
        print(
            f"Trying to solve equation immediately for 'y' yielded \n ValueError: '{ME}' "
        )


def attempt_to_solve_system_for_y(equ_sys, solver):
    try:
        solver.solve(equ_sys)
        print(f"Solved the equation system successfully.")
    except ValueError as ME:
        print(
            f"Trying to solve equation system immediately for 'y' yielded ValueError: \n'{ME}' "
        )


def continuation_and_plot(equ_sys, solver, continuation_parameter, param_max=None):

    # Set up the plot
    plt.figure()
    plt.title(f"Continuation with parameter {continuation_parameter}")
    y1_prev = 0

    # Iterate through points on the circle
    for branch_point in pseudo_arclength_continuator(
        initial_system=equ_sys,  # must be of type EquationSystem
        solver=solver,
        initial_direction=1,  # direction of tangent in x[-1] direction
        verbose=False,  # prints after every continuation step
        num_steps=300,
        stepsize=0.1,  # initial stepsize
        stepsize_range=(0.05, 0.15),
        continuation_parameter=continuation_parameter,
    ):
        # Plot the point
        plt.plot(branch_point.y[0], branch_point.y[1], "k.")

        # Stopping criteria etc. can be user-defined in the loop
        if (
            param_max is not None
            and getattr(branch_point, continuation_parameter) > param_max
        ):
            break

        if param_max is None and y1_prev < 0 and branch_point.y[1] >= 0:
            # Modify constant parameter of the equation (first equation in the equation system) and keep going
            branch_point.equations[0].radius += 2

        y1_prev = branch_point.y[1]

    plt.axis("equal")


if __name__ == "__main__":
    main()
    plt.show()
