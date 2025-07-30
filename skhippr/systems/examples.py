"""This module collects function stubs that illustrate the required syntax of the residual functions."""

import numpy as np


def trivial_newton_residual(x: np.ndarray, parameter_1: float, **parameters):
    """
    Illustrates the syntax of a residual function for an exemplary trivial Newton problem.
    The :py:class:`~skhippr.problems.newton.NewtonProblem` attempts to find a value for the first argument such that the ``residual`` is returned as (numerically) zero.

    A residual function must return the residual and the derivative w.r.t. the first argument. Derivatives w.r.t other arguments are optional (are used in, e.g., continuation).

    Parameters
    ----------

    x : np.ndarray
        The first argument must refer to the variable(s) for which the residual is computed. Can (optionally) be vectorized, where each ``x[:, k]`` denotes an individual vector of unknowns.
    parameter_1 : float
        An exemplary scalar parameter.
    **parameters
        Additional parameters.

    Returns
    -------

    residual : np.ndarray
        The residual array, with the same shape as ``x``.
    derivatives : dict of str to np.ndarray
        Dictionary containing derivatives:
            - ``"x"``: Jacobian of the residual with respect to the first argument ``x``; ``np.ndarray`` with shape ``(x.shape[0], x.shape[0], *x.shape[1:])``.
            - (e.g.) ``"parameter_1"``: Derivative of the residual with respect to the input argument ``parameter_1``, same shape as ``x``.

    Notes
    -----
    * The first argument, the unknown variable, may have an arbitrary name other than ``x``. In that case, the corresponding key in ``derivatives`` must reflect the correct name of the variable and that name must be passed to the :py:class:`~skhippr.problems.newton.NewtonProblem` upon initialization.
    * Implementation of the vectorized formulation is optional: All methods fall back to evaluating multiple unknowns one-by-one if necessary.
    """

    # Residual must have the same shape as x.
    # If x has more than 1 dimension, then every x[:, k] denotes an individual vector of unknowns.
    residual = np.zeros_like(x)

    # Jacobian w.r.t x is required for NewtonProblem.
    # dr_dx[:, :, k] is the Jacobian evaluated at x[:, k].
    dr_dx = np.zeros((x.shape[0], x.shape[0], *x.shape[1:]), dtype=x.dtype)

    # Derivatives w.r.t the other arguments are optional.
    # Parameters with derivative must be floats, so derivative has the same shape as x
    dr_dparam_1 = np.zeros_like(x)

    # Derivatives are returned as a dictionary with parameter names as keys
    derivatives = {"x": dr_dx, "parameter_1": dr_dparam_1}

    return residual, derivatives


def trivial_hbm_system(
    t: float | np.ndarray, x: np.ndarray, parameter_1: float, **parameters
):
    """
    Illustrates the syntax of the ODE system function for an exemplary trivial HBM problem.

    he system function must return the right-hand side of the ODE depending on the time ``t``, the state variable as second argument, and optional parameters. The encoding of the residual and the derivatives wotks as in :py:func:`~skhippr.systems.examples.trivial_newton_residual`.

    Parameters
    ----------

    t : float or np.ndarray
        Time. This argument **must** be the first and must be called ``t``. In the vectorized formulation, ``t[k]`` must correspond to ``x[:, k]``.
    x : np.ndarray
        The second argument refers to the state variable of the system. Can (optionally) be vectorized, where each ``x[:, k]`` denotes an individual state corresponding to time ``t[k]``.
    parameter_1 : float
        An exemplary scalar parameter.
    **parameters
        Additional parameters.

    Returns
    -------

    residual : np.ndarray
        The right-hand side of the ODE, with the same shape as ``x``.
    derivatives : dict of str to np.ndarray
        Dictionary containing derivatives:
            - ``"x"``: Jacobian of the ODE with respect to the first argument ``x``, shape ``(x.shape[0], x.shape[0], *x.shape[1:])``.
            - (e.g.) ``"parameter_1"``: Derivative of the ODE with respect to the input argument ``parameter_1``, same shape as `x`.

    Notes
    -----
    The second argument, the state, may have an arbitrary name other than ``x``. In that case, the corresponding key in ``derivatives`` must reflect the correct name of the variable and that name must be passed to the :py:class:`~skhippr.problems.hbm.hbmProblem` upon initialization.

    Caution
    -------
    The system function **must** accept time ``t`` as first argument, even in the autonomous case where this time is not used!
    """

    # Residual must have the same shape as x.
    # If x has more than 1 dimension, then every x[:, k] denotes an individual vector of unknowns.
    residual = np.zeros_like(x)

    # Jacobian w.r.t x is required for NewtonProblem.
    # dr_dx[:, :, k] is the Jacobian evaluated at x[:, k].
    dr_dx = np.zeros((x.shape[0], x.shape[0], *x.shape[1:]), dtype=x.dtype)

    # Derivatives w.r.t the other arguments are optional.
    # Parameters with derivative must be floats, so derivative has the same shape as x
    dr_dparam_1 = np.zeros_like(x)

    # Derivatives are returned as a dictionary with parameter names as keys
    derivatives = {"x": dr_dx, "parameter_1": dr_dparam_1}

    return residual, derivatives
