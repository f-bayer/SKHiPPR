from typing import TYPE_CHECKING
from collections.abc import Callable
import numpy as np

if TYPE_CHECKING:
    from skhippr.stability._StabilityMethod import _StabilityMethod


class NewtonProblem:
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
    f : Callable[..., tuple[np.ndarray, dict[str, np.ndarray]]]
        Residual function directly as provided by the user. Takes the parameters as keyword arguments. Must return a tuple whose first entry is the residual and whose second entry is a dictionary collecting the derivatives w.r.t. the arguments (argument name as key).
    variable : str
        Name of the variable being solved for. Defaults to ``"x"``.
    x0 : np.ndarray
        Initial guess for the solution.
    x : np.ndarray
        Current solution estimate.
    converged : bool
        Indicates whether the solver has converged.
    residual : ``np.ndarray`` or ``None``
        Current residual vector.
    derivatives : dict[str, np.ndarray] or ``None``
        Dictionary of derivatives (e.g., Jacobian) with respect to variables at :py:attr:`~skhippr.problems.newton.NewtonProblem.x`. Keys correspond to input arguments of the residual function and values to the corresponding Jacobian.
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
        residual_function: Callable[..., tuple[np.ndarray, dict[str, np.ndarray]]],
        initial_guess: np.ndarray,
        variable: str = "x",
        stability_method: "_StabilityMethod" = None,
        tolerance: float = 1e-8,
        max_iterations: int = 20,
        verbose: bool = False,
        **parameters,
    ):

        self.label = "Newton"
        for key, value in parameters.items():
            setattr(self, key, value)
        self._list_params = parameters.keys()
        self.f = residual_function
        self.variable = variable

        # Initial guess
        self.x0 = initial_guess
        self.x = self.x0  # is overwritten later during updates
        self.converged = False
        self.residual: np.ndarray = None  # type:ignore
        self.derivatives: dict[str, np.ndarray] = None  # type:ignore
        self.num_iter: int = 0

        # Stability
        self.stability_method = stability_method
        self.stable: bool = None  # type:ignore
        self.eigenvalues: np.ndarray = None  # type:ignore

        # Parameters
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.tolerance = tolerance

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

    def get_params(self, keys_to_exclude: tuple[str] = ()):
        """
        Retrieve a dictionary of the parameters that are passed to the residual function, optionally excluding some.

        Parameters
        ----------
        keys_to_exclude : tuple[str], optional
            A tuple of parameter names to exclude from the returned dictionary. Default is an empty tuple.

        Returns
        -------
        dict
            A dictionary containing the parameter names of the residual function and their corresponding values, excluding those specified in ``keys_to_exclude``.
        """
        return {
            key: getattr(self, key)
            for key in self._list_params
            if not key in keys_to_exclude
        }

    def f_with_params(
        self, *args, **kwargs
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Calls the residual function with the provided arguments and parameters. Parameters that are not explicitly set are filled by :py:func:`~skhippr.problems.newton.NewtonProblem.get_params`.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to the residual function. Must includes the variable itself.
        **kwargs : dict
            Keyword arguments to pass to the residual function.

        Returns
        -------

        tuple[np.ndarray, dict[str, np.ndarray]]
            The result of calling the residual with the combined arguments and parameters:

            * the residual as a ``np.ndarray``
            * the derivatives as a dictionary
        """
        return self.f(
            *args,
            **self.get_params(keys_to_exclude=kwargs.keys()),
            **kwargs,
        )

    def residual_function(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Residual function evaluated at the current guess :py:attr:`~skhippr.problems.newton.NewtonProblem.x` with the default parameters.

        Note
        ----
        This function is used in the Newton update and should be overridden by subclasses.

        Returns
        -------

        tuple[np.ndarray, dict[str, np.ndarray]]
            The result of calling the residual with the combined arguments and parameters:

            * the residual as a ``np.ndarray``
            * the derivatives as a dictionary

        """

        return self.f_with_params(self.x)

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
            self.x0 = x0_new
            # Shallow copy is needed for x to allow += during correction_step().
            self.x = x0_new.copy()
            self.derivatives = None  # type:ignore
            self.residual = None  # type:ignore
        self.converged = False
        self.num_iter = 0
        self.stable = None  # type:ignore
        self.eigenvalues = None  # type:ignore

    def correction_step(self) -> None:
        sol = self.residual_function()
        self.residual = sol[0]
        self.derivatives = sol[1]

        self.check_converged()
        if not self.converged:
            jacobian = self.derivatives[self.variable]

            delta_x = np.linalg.solve(jacobian, -1 * self.residual)
            # Here, self.x is overwritten. This removes the link between self.x and self.x0
            # '+=' would NOT remove the link and does not work with BranchPoint objects.
            self.x = self.x + delta_x

    def check_converged(self) -> None:
        if np.linalg.norm(self.residual) < self.tolerance:
            self.converged = True

    def determine_stability(self):
        if self.stability_method is not None:
            self.eigenvalues = self.stability_method.determine_eigenvalues(self)
            self.stable = self.stability_method.determine_stability(self.eigenvalues)

    def solve(self):
        """
        Applies Newton's method to solve the system of nonlinear equations given by :py:func:`~skhippr.problems.newton.NewtonProblem.residual_function`.

        Performs iterative correction steps starting from the initial guess :py:attr:`~skhippr.problems.newton.NewtonProblem.x0` until the residual norm is sufficiently small or the maximum number of iterations is reached. After convergence, performs a stability check if applicable.

        Notes
        -----
        * Solution is stored in :py:attr:`~skhippr.problems.newton.NewtonProblem.x` and final residual in :py:attr:`~skhippr.problems.newton.NewtonProblem.residual` with derivatives :py:attr:`~skhippr.problems.newton.NewtonProblem.derivatives`.

        * Prints progress and convergence information if :py:attr:`~skhippr.problems.newton.NewtonProblem.verbose` is ``True``.
        """
        if self.verbose:
            print(f", Initial guess: x[-1]={self.x[-1]:.3g}")

        while self.num_iter < self.max_iterations and not self.converged:
            self.num_iter += 1
            if self.verbose:
                print(f"Newton iteration {self.num_iter:2d}", end="")  # , x = {x}")

            self.correction_step()
            if self.verbose:
                print(
                    f", |r| = {np.linalg.norm(self.residual):8.3g}, x[-1]={self.x[-1]:.3g}"
                )

            self.check_converged()
            if self.converged and self.verbose:
                print(f" Converged", end="")
                if len(self.x) < 5:
                    print(f" to {self.variable} = {self.x}")
                else:
                    print("")

        if self.converged:
            self.determine_stability()
        elif self.verbose:
            print(f" Did not converge after {self.num_iter} iterations")
