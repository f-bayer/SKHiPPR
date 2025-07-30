"""Demonstration example for the pseudo-arclength continuation workflow with no explicit parameter, i.e., with one more equation than unknowns."""

import numpy as np
import matplotlib.pyplot as plt

from skhippr.solvers.newton import NewtonSolver
from skhippr.solvers.continuation import pseudo_arclength_continuator


def main():
    """
    Runs a demonstration of pseudo-arclength continuation for solving a nonlinear problem
    using Newton's method on the circle equation.
    The process includes:

    #. Instantiating a :py:class:`~skhippr.problems.newton.NewtonProblem` with the initial guess and the :py:func:`circle` function.
    #. Using the :py:func:`~skhippr.problems.continuation.pseudo_arclength_continuator` to iterate along the solution branch.
    #. Plotting predictors, corrected solutions, and tangents at each :py:class:`~skhippr.problems.continuation.BranchPoint`.
    #. After one circle is completed, ``radius`` is dynamically updated.
    #. Finally, the resulting branches are displayed using ``matplotlib``.

    Returns
    -------
    None
    """
    radius = 2

    # Instantiate the problem class to be solved.
    initial_problem = NewtonSolver(
        residual_function=circle,
        initial_guess=np.array([0.9 * radius, 0]),
        variable="y",  # must correspond to an argument of residual function
        stability_method=None,
        tolerance=1e-7,  # residual must be smaller than tolerance at solution
        max_iterations=15,
        verbose=False,  # if True, every Newton iteration is printed
        radius=radius,  # All keyword arguments for residual function can be passed
    )
    print(initial_problem)

    fig, ax = plt.subplots(ncols=1, nrows=1)
    x1_prev = 0

    # Iterate through points on the branch
    for branch_point in pseudo_arclength_continuator(
        initial_problem=initial_problem,  # must be of type NewtonProblem
        initial_direction=1,  # direction of tangent in x[-1] direction
        key_param=None,
        verbose=True,  # prints after every continuation step
        num_steps=300,
        stepsize=0.1,  # initial stepsize
        stepsize_range=(0.05, 0.15),
    ):
        # Predictor (before correction steps) is available at branch_point.x0
        ax.plot(branch_point.initial_guess[0], branch_point.initial_guess[1], "r+")

        # Solution is at branch_point.x (even when variable name is not x...)
        ax.plot(branch_point.x[0], branch_point.x[1], ".", color="blue")

        # Tangent is computed once when needed. Usually at prediction of next point.
        branch_point.determine_tangent()
        x_tng = branch_point.x + 0.08 * branch_point.tangent
        ax.plot(
            (branch_point.x[0], x_tng[0]), (branch_point.x[1], x_tng[1]), color="gray"
        )

        # Stopping criteria etc. can be user-defined in the loop
        if x1_prev < 0 and branch_point.x[1] >= 0:
            # Modify constant parameter and keep going
            branch_point.radius += 2

        x1_prev = branch_point.x[1]

    plt.axis("equal")
    plt.legend(("predictor", "solution", "tangent"))
    plt.title("Circles found using continuation")


def circle(y: np.ndarray, radius=1) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Computes the residual and derivatives for a circle equation in 2D.

    Given a point ``y`` in 2D space and a circle of given ``radius``, this function returns:

    * The residual of the circle equation: ``y[0]**2 + y[1]**2 - radius**2``
    * The derivatives of the residual with respect to ``y`` and ``radius``.

    Parameters
    ----------
    y : np.ndarray
        A 1D array of shape (2,) representing the coordinates (x, y) of a point in 2D space.
    radius : float, optional
        The radius of the circle (default is 1).

    Returns
    -------
    residual : np.ndarray
        A 1D numpy array containing the residual of the circle equation.
    derivatives : dict[str, np.ndarray]
        A dictionary containing the derivatives of the residual with respect to the inputs.
    """
    # Residual must be returned as 1D numpy array
    residual = np.atleast_1d(y[0] ** 2 + y[1] ** 2 - radius**2)
    dr_dy = np.atleast_2d(np.array([2 * y[0], 2 * y[1]]))
    dr_dradius = np.atleast_1d(-2 * radius)  # optional

    # Derivatives are returned as a dictionary. The keys correspond to the function arguments.
    # Derivatives w.r.t everything else than y are optional and are only needed for explicit parameter continuation.
    derivatives = {"y": dr_dy, "radius": dr_dradius}
    return residual, derivatives


if __name__ == "__main__":
    main()
    plt.show()
