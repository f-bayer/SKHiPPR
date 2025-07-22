from typing import TYPE_CHECKING
from collections.abc import Callable, Iterable
import numpy as np

if TYPE_CHECKING:
    from skhippr.stability._StabilityMethod import _StabilityMethod

from skhippr.systems.AbstractSystems import AbstractEquationSystem


class NewtonSolver:
    """
    Implements Newton's method for solving nonlinear equations. Also supports optional stability analysis after convergence.

    Definition of the underlying equation system:

    * :py:func:`~skhippr.problems.newton.NewtonProblem.f_with_params`
    * :py:attr:`~skhippr.problems.newton.NewtonProblem.variable`
    * :py:attr:`~skhippr.problems.newton.NewtonProblem.x0` (initial guess)
    * ``<param>``, when passed to the constructor as optional keyword argument, becomes an attribute with the corresponding value and is passed as keyword argument to the residual function.

    Newton solver parameters:

    * :py:attr:`~skhippr.problems.newton.NewtonProblem.tolerance`
    * :py:attr:`~skhippr.problems.newton.NewtonProblem.max_iterations`
    * :py:attr:`~skhippr.problems.newton.NewtonProblem.verbose`
    * :py:attr:`~skhippr.problems.newton.NewtonProblem.stability_method`

    Attributes (updated during the solution process):

    * :py:attr:`~skhippr.problems.newton.NewtonProblem.x`
    * :py:attr:`~skhippr.problems.newton.NewtonProblem.derivatives`
    * :py:attr:`~skhippr.problems.newton.NewtonProblem.converged`
    * :py:attr:`~skhippr.problems.newton.NewtonProblem.stable`
    * :py:attr:`~skhippr.problems.newton.NewtonProblem.eigenvalues`

    Important class methods:

    * :py:func:`~skhippr.problems.newton.NewtonProblem.solve`
    * :py:func:`~skhippr.problems.newton.NewtonProblem.reset`

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
        equations: Iterable[AbstractEquationSystem],
        unknowns: Iterable[str],
        equation_determining_stability: AbstractEquationSystem = None,
        tolerance: float = 1e-8,
        max_iterations: int = 20,
        verbose: bool = False,
    ):

        self.label = "Newton"
        self.equations = equations
        self.unknowns = unknowns
        self.init_unknowns()
        self.initial_guess = unknowns

        self.converged = False
        self.num_iter: int = 0

        # Stability
        self.equation_determining_stability = equation_determining_stability

        # Parameters
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.tolerance = tolerance

    def init_unknowns(self):
        """Make sure that every Equation object (and self) has every unknown as attribute"""
        self.length_unknowns = {}
        for unk in self.unknowns:
            equ_with_value = next(equ for equ in self.equations if hasattr(equ, unk))
            value = np.atleast_1d(getattr(equ_with_value, unk))

            if value.ndim > 1:
                raise ValueError(
                    f"Attribute{equ_with_value}.{unk} has shape {value.shape}, not 1-D: not valid as unknown!"
                )

            # Check integrity of system of equations
            for equ_other in self.equations:
                if hasattr(equ_other, unk):
                    other_value = np.atleast_1d(getattr(equ_other, unk))
                    if not np.array_equal(value, other_value):
                        raise ValueError(
                            f"Error during Newton solver initialization: "
                            f"Equations {equ_with_value} and {equ_other} have the same parameter '{unk}' "
                            f"with conflicting initial values: {value} vs. {other_value}"
                        )

            self.length_unknowns[unk] = value.size
            # custom setter also sets the attribute in all equations
            setattr(self, unk, value)
        self.length_unknowns["total"] = sum(self.length_unknowns.values())

    @property
    def vector_of_unknowns(self):
        """
        Stack all individual unknowns into a single 1-D numpy array.
        For first evaluation:
            If multiple elements of self.equation have a parameter, the first one is taken.

        Returns
        -------
        numpy.ndarray
            A 1-D array containing all unknowns concatenated along axis 0.
        """
        return np.concatenate(
            [np.atleast_1d(getattr(self.equations[0], unk)) for unk in self.unknowns]
        )

    @vector_of_unknowns.setter
    def vector_of_unknowns(self, x: np.ndarray) -> None:
        """
        Separates a 1-D array into individual unknowns components and updates their values for every Equation.
        This method takes a 1-D NumPy array `x`, splits it into segments corresponding to the sizes of the unknown variables, and updates these variables accordingly.

        Parameters
        ----------

        x : np.ndarray
            A 1-D NumPy array containing the concatenated values of all unknown variables.

        Raises
        ------

        ValueError
            If `x` is not a 1-D array.
        """

        unknowns_parsed = self.parse_vector_of_unknowns(x)
        for unk, value in unknowns_parsed.items():
            # custom setter also sets the attribute in all equations
            setattr(self, unk, value)

    @property
    def stable(self):
        return self.equation_determining_stability.stable

    @property
    def eigenvalues(self):
        return self.equation_determining_stability.eigenvalues

    def parse_vector_of_unknowns(self, x=None):
        if x is None:
            x = self.vector_of_unknowns

        if x.ndim != 1:
            raise ValueError(
                f"unknowns must be a 1-D array but array is {len(x.shape)}-D"
            )

        idx = 0
        unknowns_parsed = {}
        for unk in self.unknowns:
            n = self.length_unknowns[unk]
            value = x[idx : idx + n]
            # custom setter also sets the attribute in all equations
            unknowns_parsed[unk] = value
            idx += n
        return unknowns_parsed

    def __getattr__(self, name):
        """Custom attribute extension that searches also in the unknowns"""
        if (
            "unknowns" in self.__dict__
            and "equations" in self.__dict__
            and name in self.unknowns
        ):
            return getattr(self.equations[0], name)
        else:
            raise AttributeError(f"Attribute '{name}' not available")

    def __setattr__(self, name, value) -> None:
        """Custom attribute setter. If the attribute is part of the unknowns, it is set also in all equations."""

        if (
            "unknowns" in self.__dict__
            and "equations" in self.__dict__
            and name in self.unknowns
        ):
            for equ in self.equations:
                setattr(equ, name, value)

        super().__setattr__(name, value)

    def __str__(self):
        if self.converged:
            text_converged = f"converged {self.label} solution"
        else:
            text_converged = f"non-converged {self.label} guess"
        if self.stable is None:
            text_stable = ""
        elif self.stable:
            text_stable = "(stable)"
        else:
            text_stable = "(unstable)"
        if len(self.x) < 5:
            text_x = f"at {self.variable} = {self.x} "
        else:
            text_x = ""
        return f"{text_converged} {text_x}after {self.num_iter}/{self.max_iterations} iterations {text_stable}"

    def residual_function(
        self, update=False
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Assembles one overall residual from the residuals of self.equations. Raises an error if there are not as many equations as unknowns.

        Returns
        -------

        np.ndarray
            The result of calling the individual residuals.
        """

        res = np.concatenate([equ.residual(update=update) for equ in self.equations])
        if res.size != self.length_unknowns["total"]:
            raise RuntimeError(
                f"Length {res.size} of residual is not number {self.length_unknowns['total']} of unknowns!"
            )
        return res

    def jacobian(self, update=False, h_fd=1e-4):
        """Assemble the derivative of the residual w.r.t the unknowns.
        NOTE: If update is True, the equations are updated in order. I.e., if one equation relies on Jacobians of the previous equation, that should work.

        """
        jac = np.vstack(
            [
                np.hstack([equ.derivative(unk, update, h_fd) for unk in self.unknowns])
                for equ in self.equations
            ]
        )

        if jac.shape != (self.length_unknowns["total"], self.length_unknowns["total"]):
            raise RuntimeError(f"Size mismatch in Jacobian, got {jac.shape}")

        return jac

    def reset(self, x0_new=None) -> None:
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

        if x0_new is not None:
            self.initial_guess = x0_new
            self.vector_of_unknowns = x0_new

        self.converged = False
        self.num_iter = 0

    def correction_step(self) -> None:

        residual = self.check_converged(update=True)
        if not self.converged:
            delta_x = np.linalg.solve(self.jacobian(update=True), -residual)
            self.vector_of_unknowns = self.vector_of_unknowns + delta_x

    def check_converged(self, update=True) -> None:
        residual = self.residual_function(update=update)
        if np.linalg.norm(residual) < self.tolerance:
            self.converged = True
        return residual

    def solve(self):
        """
        Applies Newton's method to solve the system of nonlinear equations given by :py:func:`~skhippr.problems.newton.NewtonProblem.residual_function`.

        Performs iterative correction steps starting from the current vector of unknowns until the residual norm is sufficiently small or the maximum number of iterations is reached. After convergence, performs a stability check if applicable.

        Notes
        -----
        * Solution is stored in :py:attr:`~skhippr.problems.newton.NewtonProblem.unknowns`,constructed by the members of :py:attr:`~skhippr.problems.newton.NewtonProblem.unknowns_dict`
        * Prints progress and convergence information if :py:attr:`~skhippr.problems.newton.NewtonProblem.verbose` is ``True``.
        """
        if self.verbose:
            print(f", Initial guess: x[-1]={self.vector_of_unknowns[-1]:.3g}")

        while self.num_iter < self.max_iterations and not self.converged:
            self.num_iter += 1
            if self.verbose:
                print(f"Newton iteration {self.num_iter:2d}", end="")  # , x = {x}")

            self.correction_step()
            if self.verbose:
                print(
                    f", |r| = {np.linalg.norm(self.residual_function(update=False)):8.3g}, x[-1]={self.vector_of_unknowns[-1]:.3g}"
                )

            if self.converged and self.verbose:
                print(f" Converged", end="")
                if self.length_unknowns["total"] < 5:
                    print(f" to {self.parse_vector_of_unknowns()}")
                else:
                    print("")

        if self.converged:
            self.determine_stability(update=True)
        elif self.verbose:
            print(f" Did not converge after {self.num_iter} iterations")

    def determine_stability(self, update=False):
        if self.equation_determining_stability is None:
            return None, None
        else:
            return self.equation_determining_stability.determine_stability(
                update=update
            )
