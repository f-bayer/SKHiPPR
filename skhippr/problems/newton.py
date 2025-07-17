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
        self.f = residual_function

        # Initialize unknowns and residuals dictionaries.
        self.unknowns_dict = {variable: initial_guess}
        self.residuals_dict = {
            self.f_with_params: None
        }  # all sub-residual functions must accept **self.unknowns as keyword arguments.
        self.jacobians_dict: dict[tuple[Callable, str], np.ndarray] = {}
        self.initial_guess = self.unknowns

        # Initialize optional arguments into self.f
        self.f_kwargs = parameters

        self.converged = False
        self.num_iter: int = 0

        # Stability
        self.stability_method = stability_method
        self.stable: bool = None  # type:ignore
        self.eigenvalues: np.ndarray = None  # type:ignore

        # Parameters
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.tolerance = tolerance

    @property
    def unknowns(self):
        """
        Stack all individual unknowns into a single 1-D numpy array.

        Returns
        -------
        numpy.ndarray
            A 1-D array containing all unknowns concatenated along axis 0.
        """
        return np.concatenate(list(self.unknowns_dict.values()), axis=0)

    @unknowns.setter
    def unknowns(self, x: np.ndarray) -> None:
        """
        Separates a 1-D array into individual unknowns components and updates their values.
        This method takes a 1-D NumPy array `x`, splits it into segments corresponding to the sizes of the unknown variables stored in `self.unknowns_dict`, and updates these variables accordingly.
        It also updates the attributes and values in `self.f_kwargs` if applicable.

        Parameters
        ----------

        x : np.ndarray
            A 1-D NumPy array containing the concatenated values of all unknown variables.

        Raises
        ------

        ValueError
            If `x` is not a 1-D array.

        Notes
        -----
        - If the variable exists as an attribute of the object or as a key in `self.f_kwargs`, those are updated as well.
        """

        if len(x.shape) > 1:
            raise ValueError(
                f"unknowns must be a 1-D array but array is {len(x.shape)}-D"
            )

        idx = 0
        for variable, old_value in self.unknowns_dict.items():
            n = old_value.shape[0]
            self.unknowns_dict[variable] = x[idx:n]
            idx += n

            # Attempt to update the value in other places
            if hasattr(self, variable):
                setattr(self, variable, self.unknowns_dict[variable])

            if variable in self.f_kwargs:
                self.f_kwargs[variable] = self.unknowns_dict[variable]

    def __getattr__(self, name):
        """Custom attribute extension that searches also in the unknowns and problem inputs"""
        if "unknowns_dict" in self.__dict__ and name in self.unknowns_dict:
            return self.unknowns_dict[name]
        if "f_kwargs" in self.__dict__ and name in self.f_kwargs:
            return self.f_kwargs[name]
        raise AttributeError(f"Attribute '{name}' does not exist")

    def __setattr__(self, name, value) -> None:
        """Custom attribute setter also attempts to set the attribute in self.unknowns and/or self.f_kwargs. Success leads to a convergence reset."""

        if "unknowns_dict" in self.__dict__ and name in self.unknowns_dict:
            self.unknowns_dict[name] = value
            self.reset()
        if "f_kwargs" in self.__dict__ and name in self.f_kwargs:
            self.f_kwargs[name] = value
            self.reset()

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

    def get_params(self, keys_to_exclude: tuple[str] = ()):
        """
        Retrieve a subset of self.f_kwargs.

        Parameters
        ----------
        keys_to_exclude : tuple[str], optional
            A tuple of parameter names to exclude from the returned dictionary. Default is an empty tuple.

        Returns
        -------
        dict
            A dictionary containing the parameter names of the residual function (including the main unknown variable) and their corresponding values, excluding those specified in ``keys_to_exclude``.
        """

        return {
            key: value
            for key, value in self.f_kwargs.items()
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

    def residual_function(
        self, recompute=False
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Residual function evaluated at the current unknowns with the current parameters.
            Updates all sub-residuals and sub-Jacobians of the corresponding dictionaries.

        Note
        ----
        This function is used in the Newton update. Residuals and Jacobians are updated in order of appearance in self.residuals_dict().

        Returns
        -------

        tuple[np.ndarray, dict[str, np.ndarray]]
            The result of calling the residual with the combined arguments and parameters:

            * the residual as a ``np.ndarray``
            * the derivatives w.r.t all the unknowns as a numpy array

        """
        if recompute:
            for fun in self.residuals_dict:
                res, jacobians = fun(**self.unknowns_dict)

                # Populate the dictionaries
                self.residuals_dict[fun] = res
                for key in jacobians:
                    self.jacobians_dict[(fun, key)] = jacobians[key]

        # Construct the overall residual
        return np.concatenate(list(self.residuals_dict.values()))

    def jacobian(self):
        """Assemble the derivative of the residual w.r.t the unknowns row by row."""
        return np.vstack(
            [
                np.hstack(
                    [
                        self.jacobians_dict[(fun, unknown)]
                        for unknown in self.unknowns_dict
                    ]
                )
                for fun in self.residuals_dict
            ]
        )

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
            self.unknowns = x0_new

        self.converged = False
        self.num_iter = 0
        self.stable = None  # type:ignore
        self.eigenvalues = None  # type:ignore

    def correction_step(self) -> None:

        residual = self.check_converged(recompute_residual=True)
        if not self.converged:
            delta_x = np.linalg.solve(self.jacobian(), -residual)
            self.unknowns = self.unknowns + delta_x

    def check_converged(self, recompute_residual=True) -> None:
        residual = self.residual_function(recompute=recompute_residual)
        if np.linalg.norm(residual) < self.tolerance:
            self.converged = True
        return residual

    def determine_stability(self):
        if self.stability_method is not None:
            self.eigenvalues = self.stability_method.determine_eigenvalues(self)
            self.stable = self.stability_method.determine_stability(
                self.eigenvalues
            )  # TODO sollte das nicht die Klasse 'Problem' machen?!

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
            print(f", Initial guess: x[-1]={self.unknowns[-1]:.3g}")

        while self.num_iter < self.max_iterations and not self.converged:
            self.num_iter += 1
            if self.verbose:
                print(f"Newton iteration {self.num_iter:2d}", end="")  # , x = {x}")

            self.correction_step()
            if self.verbose:
                print(
                    f", |r| = {np.linalg.norm(self.residual_function(recompute=False)):8.3g}, x[-1]={self.unknowns[-1]:.3g}"
                )

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
