import numpy as np
import matplotlib.pyplot as plt

from skhippr.equations.AbstractEquation import AbstractEquation
from skhippr.equations.EquationSystem import EquationSystem


class NewtonSolver:
    """
    Implements Newton's method for solving nonlinear equations. Also supports optional stability analysis after convergence.

    Definition of the underlying equation system:

    * :py:func:`~skhippr.cycles.newton.NewtonProblem.f_with_params`
    * :py:attr:`~skhippr.cycles.newton.NewtonProblem.variable`
    * :py:attr:`~skhippr.cycles.newton.NewtonProblem.x0` (initial guess)
    * ``<param>``, when passed to the constructor as optional keyword argument, becomes an attribute with the corresponding value and is passed as keyword argument to the residual function.

    Newton solver parameters:

    * :py:attr:`~skhippr.cycles.newton.NewtonProblem.tolerance`
    * :py:attr:`~skhippr.cycles.newton.NewtonProblem.max_iterations`
    * :py:attr:`~skhippr.cycles.newton.NewtonProblem.verbose`
    * :py:attr:`~skhippr.cycles.newton.NewtonProblem.stability_method`

    Attributes (updated during the solution process):

    * :py:attr:`~skhippr.cycles.newton.NewtonProblem.x`
    * :py:attr:`~skhippr.cycles.newton.NewtonProblem.derivatives`
    * :py:attr:`~skhippr.cycles.newton.NewtonProblem.converged`
    * :py:attr:`~skhippr.cycles.newton.NewtonProblem.stable`
    * :py:attr:`~skhippr.cycles.newton.NewtonProblem.eigenvalues`

    Important class methods:

    * :py:func:`~skhippr.cycles.newton.NewtonProblem.solve`
    * :py:func:`~skhippr.cycles.newton.NewtonProblem.reset`

    Attributes:
    -----------
    # TODO
    num_iter : int
        Number of iterations performed.
    stability_method : :py:class:`~skhippr.stability._StabilityMethod._StabilityMethod` or ``None``
        Object encoding stability method. Defaults to ``None`` (no stability analysis performed).
    stable : bool or None
        Indicates whether the solution is stable (if stability analysis is performed). ``None`` if no stability analysis was performed.
    eigenvalues : np.ndarray or None
        Eigenvalues computed during stability analysis. ``None`` if no stability analysis was performed.
    max_iterations : int
        Maximum number of allowed iterations.
    verbose : bool
        If True, prints progress information.
    tolerance : float
        Convergence tolerance for the residual norm.

    """

    def __init__(
        self,
        tolerance: float = 1e-8,
        max_iterations: int = 20,
        verbose: bool = False,
    ):

        self.converged = False
        self.num_iter: int = 0

        # Parameters
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.tolerance = tolerance

    # def __str__(self):
    #     if self.converged:
    #         text_converged = f"converged {self.label} solution"
    #     else:
    #         text_converged = f"non-converged {self.label} guess"
    #     if self.stable is None:
    #         text_stable = ""
    #     elif self.stable:
    #         text_stable = "(stable)"
    #     else:
    #         text_stable = "(unstable)"
    #     if len(self.x) < 5:
    #         text_x = f"at {self.variable} = {self.x} "
    #     else:
    #         text_x = ""
    #     return f"{text_converged} {text_x}after {self.num_iter}/{self.max_iterations} iterations {text_stable}"

    def reset(self) -> None:
        """
        Reset the state of the solver, optionally with a new initial guess.

        If a new initial guess ``x0_new`` is provided, updates the internal state accordingly.
        Resets convergence status, iteration count, and other diagnostic attributes.

        Parameters
        ----------

        x0_new : array-like, optional
            New initial guess for the solver. If provided, replaces the current initial guess
            and resets related state variables.

        """

        self.converged = False
        self.num_iter = 0

    def correction_step(self, equation_system) -> None:

        residual = self.check_converged(equation_system, update=True)
        if not equation_system.solved:
            delta_x = np.linalg.solve(equation_system.jacobian(update=True), -residual)
            equation_system.vector_of_unknowns = (
                equation_system.vector_of_unknowns + delta_x
            )

    def check_converged(self, equation_system, update=True) -> np.array:
        residual = equation_system.residual_function(update=update)
        if np.linalg.norm(residual) < self.tolerance:
            equation_system.solved = True
        return residual

    def solve_equation(self, equation: AbstractEquation, unknown: str):
        system = EquationSystem([equation], [unknown], equation)
        self.solve(system)
        if not system.solved:
            raise RuntimeError(
                f"Could not solve {equation} within {self.num_iter} iterations."
            )
        return system

    def solve(self, equation_system: EquationSystem):
        """
        Applies Newton's method to solve the system of nonlinear equations given by :py:func:`~skhippr.cycles.newton.NewtonProblem.residual_function`.

        Performs iterative correction steps starting from the current vector of unknowns until the residual norm is sufficiently small or the maximum number of iterations is reached. After convergence, performs a stability check if applicable.

        Notes
        -----
        * Solution is stored in :py:attr:`~skhippr.cycles.newton.NewtonProblem.unknowns`,constructed by the members of :py:attr:`~skhippr.cycles.newton.NewtonProblem.unknowns_dict`
        * Prints progress and convergence information if :py:attr:`~skhippr.cycles.newton.NewtonProblem.verbose` is ``True``.
        """
        if not equation_system.well_posed:
            raise ValueError(
                "Equation system is not well-posed: Number of unknowns and number of equations differ"
            )
        visualize = False
        self.reset()

        if self.verbose:
            print(f"Initial guess: x[-1]={equation_system.vector_of_unknowns[-1]:.3g}")

        while self.num_iter < self.max_iterations and not equation_system.solved:
            self.num_iter += 1
            if self.verbose:
                print(
                    f"Newton iteration {self.num_iter:2d}", end=""
                )  # , x = {equation_system.vector_of_unknowns}", end="")

            if visualize:
                print("visualizing")
                ax = plt.gca()
                ax.plot(
                    equation_system.vector_of_unknowns[-1],
                    equation_system.vector_of_unknowns[0],
                    equation_system.vector_of_unknowns[1],
                    ".",
                    color="blue",
                )
                pass

            self.correction_step(equation_system)

            if visualize:
                print("visualizing")
                ax = plt.gca()
                ax.plot(
                    equation_system.vector_of_unknowns[-1],
                    equation_system.vector_of_unknowns[0],
                    equation_system.vector_of_unknowns[1],
                    ".",
                    color="green",
                )
                pass

            if self.verbose:
                print(
                    f", |r| = {np.linalg.norm(equation_system.residual_function(update=False)):8.3g}, x[-1]={equation_system.vector_of_unknowns[-1]:.3g}"
                )

            if equation_system.solved and self.verbose:
                print(f" Converged", end="")
                if equation_system.length_unknowns["total"] < 5:
                    print(f" to {equation_system.parse_vector_of_unknowns()}")
                else:
                    print("")

        if equation_system.solved:
            equation_system.determine_stability(update=True)
        elif self.verbose:
            print(f" Did not converge after {self.num_iter} iterations")
