"""
Module for pseudo-arclength continuation of nonlinear problems.

This module provides tools for performing pseudo-arclength continuation using Newton's method.
It includes a generator function for continuation and a :py:class:`~skhippr.problems.continuation.BranchPoint` wrapper class that extends :py:class:`~skhippr.problems.newton.NewtonProblem`
with continuation-specific features.

Classes
-------

BranchPoint
    Represents a point on a continuation branch. Wraps a NewtonProblem instance and provides
    augmented residuals, tangent computation, and prediction methods. Most attributes and methods
    are delegated to the underlying NewtonProblem.

Functions
---------

pseudo_arclength_continuator(initial_problem, stepsize=None, stepsize_range=(1e-4, 1e-1), initial_direction=1, key_param=None, value_param=None, verbose=False, num_steps=1000)
    Performs pseudo-arclength continuation for a nonlinear problem. Yields a sequence of BranchPoint
    objects representing solutions at different parameter values along the branch.

Dependencies
------------

numpy
copy
typing.override
collections.abc.Iterable, Iterator
skhippr.problems.newton.NewtonProblem
"""

from collections.abc import Iterable, Iterator
from typing import override
from copy import copy
import numpy as np

from skhippr.problems.newton import NewtonSolver
from skhippr.systems.AbstractSystems import AbstractEquation


def pseudo_arclength_continuator(
    initial_problem: NewtonSolver,
    stepsize: float = None,
    stepsize_range: Iterable[float] = (1e-4, 1e-1),
    initial_direction=1,
    key_param=None,
    value_param=None,
    verbose=False,
    num_steps: int = 1000,
) -> Iterator["BranchPoint"]:
    """
    Perform pseudo-arclength continuation of the solution branch emerging from a nonlinear :py:class:`~skhippr.problems.newton.NewtonProblem`.

    This generator yields a sequence of :py:class:`~skhippr.problems.continuation.BranchPoint` objects, each representing an individual solution
    to the problem, by following the solution branch using the pseudo-arclength continuation method.

    Parameters
    ----------

    initial_problem : :py:class:`~skhippr.problems.newton.NewtonProblem`
        The initial :py:class:`~skhippr.problems.newton.NewtonProblem` instance to start continuation from. It need not be solved already.

        * Explicit case: If the problem has as many equations as unknowns, an explicit continuation parameter (``key_param``) is required.
        * Implicit case: Otherwise, the problem must have exactly one equation more than unknowns.

    stepsize : float, optional
        Initial step size for continuation. If None, uses the lower bound of ``stepsize_range``.
    stepsize_range : Iterable of float, optional
        Tuple specifying the minimum and maximum allowed step sizes. Default is (1e-4, 1e-1).
    initial_direction : int, optional
        Direction (+1 or -1) to start continuation along the branch. If positive, the continuation parameter (explicit case) or the last entry of the unknowns (implicit case) increases initially.
    key_param : str, optional
        Name of the parameter to track during continuation.
    value_param : Any, optional
        Value of the parameter at the initial point. Defaults to ``initial_problem.<key_param>``.
    verbose : bool, optional
        If ``True``, prints progress and diagnostic information. Default is ``False``.
    num_steps : int, optional
        Maximum number of continuation steps to perform. Default is 1000.

    Yields
    ------

    :py:class:`~skhippr.problems.continuation.BranchPoint`
        The next converged solution point along the continuation branch.

    Raises
    ------

    RuntimeError
        If the initial problem does not converge.

    Notes
    -----
    * If the minimal stepsize does not converge, the continuation stops.
    * The continuator does not check whether the branch point lies within any value bounds. Such checks should be made within the ``for`` loop over which the continuator is iterated.
    """
    if stepsize is None:
        stepsize = stepsize_range[0]

    initial_point = BranchPoint(
        problem=initial_problem,
        key_param=key_param,
        value_param=value_param,
        anchor=initial_direction,
    )

    initial_point.solve()
    if not initial_point.converged:
        raise RuntimeError("Initial guess did not converge!")
    elif verbose:
        print(f"Initial guess converged after {initial_point.num_iter} steps.")

    yield initial_point
    last_point = initial_point
    stop_msg = "All steps finished successfully."

    k = 0
    while k < num_steps:
        if verbose:
            print(f"Continuation step {k}, step size {stepsize:.3f}:", end=" ")

        next_point = last_point.predict(stepsize=stepsize)

        if verbose and key_param is not None:
            print(f"{key_param}_pred = {getattr(next_point, key_param):.2f}", end=" ")

        next_point.solve()

        if next_point.converged:
            if verbose:
                print(f"success after {next_point.num_iter} steps.")
            yield next_point
            k += 1

            last_point = next_point
            stepsize = min(1.2 * stepsize, stepsize_range[1])

        elif stepsize > stepsize_range[0]:
            if verbose:
                print("no success. Retry.")
            stepsize = max(0.5 * stepsize, stepsize_range[0])

        else:
            stop_msg = "Minimal step size did not converge."
            break

    if verbose:
        print(stop_msg)


class BranchPoint(NewtonSolver):
    """
    A :py:class:`~skhippr.problems.continuation.BranchPoint` represents a point on an implicit or explicit continuation branch.

    An object of this class wraps a :py:class:`~skhippr.problems.newton.NewtonProblem` instance (or subclasses) and provides additional functionality for continuation methods:

    * stores the tangent vector at the branch point
    * predicts the next point on the branch
    * If an explicit continuation parameter is passed, it is appended to the vector ``x`` of unknowns.

    Notes
    -----

    All attributes that are not explicitly set in the :py:class:`~skhippr.problems.continuation.BranchPoint` (and mentioned below) are delegated directly to the underlying :py:class:`~skhippr.problems.newton.NewtonProblem`.

    For example, if ``branch_point`` is a :py:class:`~skhippr.problems.continuation.BranchPoint` with an underlying :py:class:`~skhippr.problems.HBM.HBMProblem`, the frequency of the solution can be accessed immediately by ``branch_point.omega``.

    Attributes which are *not* delegated:
    -------------------------------------

    _problem : :py:class:`~skhippr.problems.newton.NewtonProblem`
        The underlying :py:class:`~skhippr.problems.newton.NewtonProblem` instance being wrapped.
    anchor : np.ndarray
        The anchor vector used in the augmented system for continuation. Newton updates are performed orthogonal to the anchor.
        If a scalar is passed during initialization, the last entry of ``x`` is kept constant.
    tangent : np.ndarray or None
        The tangent to the branch at the branch point, used for prediction of the next point.
        Is set after the Newton updates have converged.
    variable: str
        Has the value ``f"{problem.variable}_ext"`` to distinguish original problem variable and extended problem variable.
    key_param: str or None
        Name of the continuation parameter.

        * If ``None``, the problem is assumed to be implicit, i.e., the underlying :py:class:`~skhippr.problems.newton.NewtonProblem` has one equation less than unknowns.
        * If not ``None``, the problem is assumed to be explicit. The underlying :py:class:`~skhippr.problems.newton.NewtonProblem` must have as many equations as unknowns and ``<key_param>`` must be a keyword argument to the system function. The ``derivatives`` dictionary returned by the system function must have an entry for ``key_param``.

        .. caution::

            ``key_param`` is set as attribute of the underlying :py:class:`~skhippr.problems.newton.NewtonProblem`, but can (like all other attributes) be immediately accessed by ``branch_point.key_param`` .


    Parameters
    ----------
    problem : :py:class:`~skhippr.problems.newton.NewtonProblem`
        The underlying problem to be solved along the branch. May have one equation less than unknowns (implicit case) or as many
    x0: np.ndarray, optional
        Initial guess. If ``None``, defaults to the current ``x`` value fo the underlying problem.
    key_param: str or None, optional
        Name of the continuation parameter.

        * If ``None``, the problem is assumed to be implicit, i.e., the underlying :py:class:`~skhippr.problems.newton.NewtonProblem` has one equation less than unknowns.
        * If not ``None``, the problem is assumed to be explicit. The underlying :py:class:`~skhippr.problems.newton.NewtonProblem` must have as many equations as unknowns and ``<key_param>`` must be a keyword argument to the system function. The ``derivatives`` dictionary returned by the system function must have an entry for ``key_param``.

    value_param: float or None, optional
        Initial value of the continuation parameter. Must be passed if ``key_param`` is not None.
    anchor : np.ndarray or float, optional
        Anchor vector (np.ndarray, Newton updates are orthogonal) or initial direction (scalar, ``1`` or ``-1``). Defaults to ``1``.

    """

    def __init__(
        self,
        problem: NewtonSolver,
        x0: np.ndarray = None,
        key_param: str = None,
        value_param: float = None,
        anchor: np.ndarray | float = 1,
    ):
        variable = problem.variable

        if x0 is None:
            x0 = problem.x
            if key_param:
                if value_param:
                    x0 = np.append(x0, value_param)
                else:
                    x0 = np.append(x0, getattr(problem, key_param))

        # if x0 is NOT None, we expect that it is already of extended length

        if key_param is not None:
            variable = f"{variable}_extended"

        self._problem = problem
        self._problem.reset()
        self._problem.key_param = key_param

        super().__init__(
            residual_function=None,
            initial_guess=x0,
            variable=variable,
            stability_method=problem.stability_method,
            tolerance=problem.tolerance,
            max_iterations=problem.max_iterations,
            verbose=problem.verbose,
        )

        self.label = f"{self._problem.label} branch point"

        if np.isscalar(anchor):
            self.anchor = np.zeros_like(self.x)
            self.anchor[-1] = anchor
        else:
            self.anchor = anchor
        self.tangent = None

    def __getattr__(self, name):
        """
        Provides dynamic attribute access for the instance.
        If the requested attribute `name` is not found on the current instance,
        this method attempts to retrieve it from the wrapped `_problem` object.
        If `_problem` does not have the attribute either, an AttributeError is raised.
        """
        # Wrap the _problem object to provide direct access to its attributes and methods.
        # self.__getattr__(name) is ONLY called if self.name throws an AttributeError.
        if "_problem" in self.__dict__:
            try:
                value = getattr(self._problem, name)
                return value
            except AttributeError:
                pass

        raise AttributeError(
            f"Neither the branch point nor its reference problem have an attribute '{name}'."
        )

    def __setattr__(self, name, value):
        """
        Custom attribute setter that delegates most attribute assignments to the wrapped '_problem' object.
        - If the attribute name is one of ("_problem", "anchor", "tangent", "variable", "x"), it is set directly on the current instance.
        - If the attribute name is one of ("_list_params", "f"), the assignment is ignored. This is relevant for the constructor.
        - For all other attribute names, the assignment is delegated.
        """
        # Defer almost all parameters to the _problem
        if name in ("_problem", "anchor", "tangent", "variable", "x"):
            super().__setattr__(name, value)
        elif name in ("_list_params", "f"):
            pass
        else:
            setattr(self._problem, name, value)

    @property
    def x(self):
        if self.key_param:
            return np.append(self._problem.x, getattr(self._problem, self.key_param))
        else:
            return self._problem.x

    @x.setter
    def x(self, value):
        if self.key_param:
            self._problem.x = value[:-1]
            setattr(self._problem, self.key_param, value[-1])
        else:
            self._problem.x = value

    def copy_problem(self) -> "BranchPoint":
        """Returns a shallow copy of ``self._problem`` with disconnected parameters."""
        return copy(self._problem)

    @override
    def residual_function(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Returns the residual and its derivatives for the augmented problem, including the anchor equation.

        Appends a 0 to the residual of the underlying problem and adds the anchor equation to the Jacobian.
        If applicable (explicit continuation), also adds the derivative w.r.t the continuation parameter to the Jacobian.

        """
        residual, derivatives = self._problem.residual_function()
        dr_dx = derivatives[self._problem.variable]

        # append anchor equation
        residual = np.append(residual, 0)

        # add derivative wrt parameter to Jacobian
        if self.key_param is not None:
            dr_dx = np.hstack((dr_dx, derivatives[self.key_param][:, np.newaxis]))

        # add anchor to Jacobian
        derivatives[self.variable] = np.vstack((dr_dx, self.anchor))
        return residual, derivatives

    def determine_tangent(self):
        """
        Compute and normalize the tangent vector at the branch point.

        This method checks if the current solution has converged. If not, it raises a ``RuntimeError``.
        It then constructs the tangent vector by solving a linear system. The resulting normalized tangent vector forms an acute angle with the previous tantent vector. It is stored in the ``self.tangent`` attribute.

        Raises
        -------

        RuntimeError
            If the solution has not converged and the tangent would thus be meaningless.

        """
        if not self.converged:
            raise RuntimeError("Cannot determine tangent: solution not converged.")

        dr_dx_full = self.derivatives[self.variable]
        residual_tangent = np.zeros_like(self.x)
        residual_tangent[-1] = 1

        # Solve for the tangent vector
        tangent = np.linalg.solve(dr_dx_full, residual_tangent)
        self.tangent = tangent / np.linalg.norm(tangent)

    def predict(self, stepsize: float) -> "BranchPoint":
        """
        Predict the next branch point in the continuation process.

        Parameters
        ----------

        stepsize : float
            The step size to advance along the tangent direction.

        Returns
        -------

        BranchPoint
            A new :py:class:`~skhippr.problems.continuation.BranchPoint` instance (not converged) representing the predicted next point along the continuation path.

        Notes
        -----
        The solution is shallow-copied to avoid modifying the state of the current point during future Newton iterations of the next point.
        """
        if self.tangent is None:
            self.determine_tangent()

        branch_point_next = BranchPoint(
            problem=self.copy_problem(),
            x0=self.x + stepsize * self.tangent,
            key_param=self.key_param,
            value_param=None,
            anchor=self.tangent,
        )

        return branch_point_next


class ContinuationAnchor(AbstractEquation):
    def __init__(self, equations):
        super().__init__(None)
