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
import numpy as np

from skhippr.solvers.newton import NewtonSolver, EquationSystem
from skhippr.systems.AbstractSystems import AbstractEquation


def pseudo_arclength_continuator(
    initial_system: EquationSystem,
    solver: NewtonSolver,
    stepsize: float = None,
    stepsize_range: Iterable[float] = (1e-4, 1e-1),
    initial_direction=1,
    continuation_parameter=None,
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
        underlying_system=initial_system,
        continuation_parameter=continuation_parameter,
        initial_direction=initial_direction,
    )

    solver.solve(initial_point)
    if not initial_point.solved:
        raise RuntimeError("Initial guess did not converge!")
    elif verbose:
        print(f"Initial guess converged after {solver.num_iter} steps.")

    yield initial_point
    last_point = initial_point
    stop_msg = "All steps finished successfully."

    k = 0
    while k < num_steps:
        if verbose:
            print(f"Continuation step {k}, step size {stepsize:.3f}:", end=" ")

        next_point = last_point.predict(stepsize=stepsize)

        if verbose and continuation_parameter is not None:
            print(
                f"{continuation_parameter}_pred = {np.squeeze(getattr(next_point, continuation_parameter)):.2f}",
                end=" ",
            )

        solver.solve(next_point)

        if next_point.solved:
            if verbose:
                print(f"success after {solver.num_iter} steps.")
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


class BranchPoint(EquationSystem):
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
        underlying_system: EquationSystem,
        continuation_parameter: str = None,
        initial_direction=1,
    ):

        anchor_equation = ContinuationAnchor(
            underlying_system,
            continuation_parameter,
            initial_direction=initial_direction,
        )

        unknowns = underlying_system.unknowns
        if continuation_parameter is not None:
            unknowns = unknowns + [continuation_parameter]

        super().__init__(
            equations=underlying_system.equations + [anchor_equation],
            unknowns=unknowns,
            equation_determining_stability=underlying_system.equation_determining_stability,
        )
        self.tangent = None

    def determine_tangent(self):
        """
        Compute and normalize the tangent vector at the branch point.

        This method checks if the current solution has converged. If not, it raises a ``RuntimeError``.
        It then constructs the tangent vector by solving a linear system. The resulting normalized tangent vector forms an acute angle with the previous tangent vector. It is stored in the ``self.tangent`` attribute.

        Raises
        -------

        RuntimeError
            If the solution has not converged and the tangent would thus be meaningless.

        """
        if not self.solved:
            raise RuntimeError(
                "Cannot determine tangent at branch point: Branch point not reached (system not solved)."
            )

        jac = self.jacobian(update=False)
        rhs = np.zeros(jac.shape[0])
        rhs[-1] = 1
        tangent = np.linalg.solve(jac, rhs)
        self.tangent = tangent / np.linalg.norm(tangent)
        # adding the tangent does not remove solution integrity
        self.solved = True
        return self.tangent

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
        The system is shallow-copied to avoid modifying the state of the current point during future Newton iterations of the next point.
        """
        if self.tangent is None:
            self.determine_tangent()

        branch_point_next = self.duplicate()
        branch_point_next.equations[-1].initialize_anchor(previous_tangent=self.tangent)
        branch_point_next.vector_of_unknowns = (
            self.vector_of_unknowns + stepsize * self.tangent
        )
        branch_point_next.tangent = None

        return branch_point_next


class ContinuationAnchor(AbstractEquation):
    def __init__(
        self,
        equation_system,
        continuation_parameter=None,
        previous_tangent=None,
        initial_direction=1,
    ):
        super().__init__(None)
        self.equation_system = equation_system
        self.continuation_parameter = continuation_parameter

        if previous_tangent is not None:
            self.initialize_anchor(previous_tangent)
        elif initial_direction is not None:
            self.initialize_anchor_with_direction(initial_direction)
        # otherwise (during prediction), the initialization must be called manually

    def initialize_anchor(self, previous_tangent):
        if self.continuation_parameter is None:
            self.anchor = self.equation_system.parse_vector_of_unknowns(
                previous_tangent
            )
        else:
            self.anchor = self.equation_system.parse_vector_of_unknowns(
                previous_tangent[:-1].flatten()
            )
            self.anchor[self.continuation_parameter] = np.atleast_1d(
                previous_tangent[-1]
            )

    def initialize_anchor_with_direction(self, initial_direction):

        anchor = np.zeros(self.equation_system.length_unknowns["total"])
        if self.continuation_parameter is None:
            anchor[-1] = initial_direction
        else:
            anchor = np.append(anchor, initial_direction)

        self.initialize_anchor(anchor)

    def residual_function(self):
        return np.atleast_1d(0)

    def closed_form_derivative(self, variable):
        try:
            return self.anchor[variable][np.newaxis, :]
        except KeyError:
            raise NotImplementedError
