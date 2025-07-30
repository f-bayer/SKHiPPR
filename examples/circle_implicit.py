"""Demonstration example for the pseudo-arclength continuation workflow with no explicit parameter, i.e., with one more equation than unknowns."""

import numpy as np
import matplotlib.pyplot as plt

from skhippr.solvers.newton import NewtonSolver
from skhippr.solvers.continuation import pseudo_arclength_continuator
from skhippr.equations.Circle import CircleEquation
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
    print("Implicit circle example.")

    # Instantiate theequation to be solved
    radius = 2
    equ = CircleEquation(y=np.array([0.9 * radius, 0]), radius=radius)
    # Verify that the provided initial conditions do not solve the equation
    print(
        f"Initial guess does not solve the equation: r = {equ.radius} != {np.linalg.norm(equ.y)} = ||y||"
    )

    # Instantiate a Newton solver
    solver = NewtonSolver(verbose=False)

    # We can solve immediately for the radius corresponding to a fixed y
    solver.solve_equation(equ, "radius")
    print(f"Solved for radius: r = {equ.radius} == {np.linalg.norm(equ.y)} = ||y||")
    # Note that scalar unknowns are transformed into 1-D numpy arrays.

    # Revert to original setup
    equ.radius = radius

    # We can NOT solve immediately for y because the number of equations does not match the number of unknowns
    try:
        solver.solve_equation(equ, "y")
    except ValueError as ME:
        print(f"Trying to solve immediately for 'y' yielded \n ValueError: '{ME}' ")

    # To find all y with the desired radius, we can use continuation.
    # The continuator needs an EquationSystem (which encodes the equation(s) together with the unknowns).
    equ_sys = EquationSystem([equ], unknowns=["y"])

    # Set up the plot
    plt.figure()
    y1_prev = 0

    # Iterate through points on the circle

    for branch_point in pseudo_arclength_continuator(
        initial_system=equ_sys,  # must be of type EquationSystem
        solver=solver,
        initial_direction=1,  # direction of tangent in x[-1] direction
        verbose=True,  # prints after every continuation step
        num_steps=300,
        stepsize=0.1,  # initial stepsize
        stepsize_range=(0.05, 0.15),
    ):
        # Plot the point
        plt.plot(branch_point.y[0], branch_point.y[1], "k.")

        # Stopping criteria etc. can be user-defined in the loop
        if y1_prev < 0 and branch_point.y[1] >= 0:
            # Modify constant parameter of the equation (first equation in the equation system) and keep going
            branch_point.equations[0].radius += 2

        y1_prev = branch_point.y[1]

    plt.axis("equal")
    plt.title("Circles found using continuation")


if __name__ == "__main__":
    main()
    plt.show()
