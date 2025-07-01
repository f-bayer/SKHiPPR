"""Dynamics and bifurcation diagram of a stick-slip oscillator at relatively high belt speeds.
Reference:
----------
Veraszto and Stepan, "Nonlinear Dynamics of hardware-in-the-loop experiments on stick-slip phenomena", IJNLM, 2017, https://doi.org/10.1016/j.ijnonlinmec.2017.01.006
"""

import numpy as np
import matplotlib.pyplot as plt

from skhippr.problems.newton import NewtonProblem
from skhippr.stability._StabilityMethod import StabilityEquilibrium
from skhippr.problems.continuation import pseudo_arclength_continuator


def main(parameters=None):
    if parameters is None:
        parameters = stick_slip_parameters(v=0.1)

    initial_guess_eq = np.array([0.0, 0.0])

    # find and continue equilibrium
    eq_problem = NewtonProblem(
        residual_function=stick_slip_equilibrium,
        initial_guess=initial_guess_eq,
        variable="y",
        stability_method=StabilityEquilibrium(n_dof=2),
        verbose=False,
        **parameters
    )
    eq_problem.solve()
    branch_eq = []
    for branch_point_eq in pseudo_arclength_continuator(
        eq_problem, 0.05, (0.001, 0.05), 1, "v", parameters["v"], True, 10000
    ):
        if len(branch_eq) > 0 and branch_eq[-1].stable != branch_point_eq.stable:
            point_bif = branch_point_eq

        branch_eq.append(branch_point_eq)

        if branch_point_eq.v > 1.3 * parameters["v_star"]:
            break

    y_eq = np.array([point.x[0] for point in branch_eq])
    v = np.array([point.v for point in branch_eq])
    y_exp = np.array([y_equilibrium(**point.get_params()) for point in branch_eq])
    plt.figure()
    plt.plot(v, y_eq, label="equilibrium (cont)")
    plt.plot(v, y_exp, "--", label="equilibrium (pred)")
    try:
        plt.plot(
            point_bif.v, point_bif.x[0], "x", label="Hopf bif point (cont, approx.)"
        )
        parameters["v"] = parameters["v_hopf"]
        plt.plot(
            parameters["v_hopf"],
            y_equilibrium(**parameters),
            ".",
            label="Hopf bif. point (pred.)",
        )
    except UnboundLocalError:
        print("No bifurcation found")
    plt.legend()


def stick_slip_parameters(
    m=0.538, b_0=5.746, k=6957, v=0.22, C=2, C_0=6, c_v=10, b_s=0.2
) -> dict[str, float]:
    """
    Return all Stribeck parameters in a dictionary.
    Default parameter values taken from Tables 1 and 2 of Veraszto2017.

    Parameters
    ----------

    m : float, optional
        Mass of the block.
    b_0 : float, optional
        Viscous damping coefficient of the oscillator.
    k : float, optional
        Stiffness  of the oscillator.
    v : float, optional
        Belt velocity.
    C : float
        Stribeck kinetic friction force.
    C_0 : float
        Stribeck static friction force.
    c_v : float
        Exponential decay parameter for convergence to kinetic friction force.
    b_s : float
        Stribeck viscous damping factor.

    Returns
    -------

    dict of str to float
        Dictionary containing all computed Stribeck parameters:
            * "m": Mass of the oscillator
            * "b_0": Viscous damping coefficient of the oscillator
            * "k": Stiffness coefficient of the oscillator
            * "v": Belt velocity
            * "C": Stribeck kinetic friction force
            * "C_0": Stribeck static friction force
            * "c_v": Exponential decay parameter for convergence to kinetic friction force
            * "b_s": Stribeck viscous damping factor
            * "v_hopf": Hopf bifurcation velocity (Eq. 11)
            * "y_equilibrium": Equilibrium displacement (Eq. 3)
            * "zeta_0": Damping ratio for oscillator (Eq. 5)
            * "zeta_s": Damping ratio for contact force (Eq. 5)
            * "zeta_v": Total damping ratio
    """

    omega = np.sqrt(k / m)  # Eq. (4)

    # Eq. (5)
    zeta_0 = b_0 / (2 * m * omega)
    zeta_s = b_s / (2 * m * omega)
    zeta_v = zeta_0 + zeta_s

    v_hopf = np.log(c_v * (C_0 - C) / (2 * zeta_v * omega * m)) / c_v  # Eq. (11)
    v_star = 4 / c_v * (1 - np.sqrt(1 - c_v * v_hopf / 2))

    return {
        "m": m,
        "b_0": b_0,
        "k": k,
        "v": v,
        "C": C,
        "C_0": C_0,
        "c_v": c_v,
        "b_s": b_s,
        "v_hopf": v_hopf,
        "v_star": v_star,
        "y_equilibrium": y_equilibrium(k, v, C, C_0, c_v, b_s),
        "zeta_0": zeta_0,
        "zeta_s": zeta_s,
        "zeta_v": zeta_v,
    }


def y_equilibrium(k=6957, v=0.22, C=2, C_0=6, c_v=10, b_s=0.2, **_):
    return (C + (C_0 - C) * np.exp(-c_v * v) + b_s * v) / k  # Eq. (3)


def force_stribeck(
    y_dot: float | np.ndarray,
    v: float,
    C: float,
    C_0: float,
    c_v: float,
    b_s: float,
    **_
) -> np.ndarray:
    """
    Compute the Stribeck friction force as described in Veraszto2017 (eq. 2).

    Parameters
    ----------
    y_dot : float or array_like
        Velocity of the block.
    v : float
        belt velocity.
    C : float
        kinetic friction force.
    C_0 : float
        static friction force.
    c_v : float
        Exponential decay parameter for convergence to kinetic friction force.
    b_s : float
        viscous damping factor.
    **_ : dict, optional
        Additional keyword arguments (ignored). Declared to allow passing the parameters dictionary.

    Returns
    -------
    float or ndarray
        The computed Stribeck friction force.

    dict
        The derivatives w.r.t y_dot and v


    """

    force = np.atleast_1d(
        (C + (C_0 - C) * np.exp(-c_v * np.abs(v - y_dot))) * np.sign(v - y_dot)
        + b_s * (v - y_dot)
    )

    dforce_dv = -c_v * (C_0 - C) * np.exp(-c_v * np.abs(v - y_dot)) + b_s

    # Sticking case
    try:
        force[y_dot == v] = np.nan
        dforce_dv[y_dot == v] = np.nan
    except TypeError:
        # force is not array_like
        if y_dot == v:
            force = np.nan
            dforce_dv = np.nan

    dforce_dydot = -dforce_dv

    return force, {"y_dot": dforce_dydot, "v": dforce_dv}


def stick_slip_equilibrium(y, **parameters):
    return stick_slip_dynamics(0, y, **parameters)


def stick_slip_dynamics(
    t, y: np.array, **parameters
) -> tuple[np.array, dict[str, np.array]]:
    """
    Compute the dynamics and Jacobians for the stick-slip system.
    The friction force is modeled using the Stribeck effect.

    Parameters
    ----------

    t : float
        Current time (not used in the computation but included for compatibility with ODE solvers).
    y : ndarray
        State vector of shape (2, ...) where `y[0, ...]` is position and `y[1, ...]` is velocity.
    **parameters :
        Additional keyword arguments (parameters) required for the dynamics, must include:
            - "b_0": float, viscous damping coefficient.
            - "k": float, spring constant.
            - Any other parameters required by `force_stribeck`.

    Returns
    -------

    f : ndarray
        Time derivative of the state vector, same shape as `y`.
    derivatives : dict of ndarray
        Dictionary containing the Jacobians:
            - "y": Jacobian of `f` with respect to `y`, shape (2, 2, ...).
            - "v": Jacobian of `f` with respect to `v`, same shape as `y`.
    """
    f_stribeck, derivatives_stribeck = force_stribeck(y_dot=y[1, ...], **parameters)

    f = np.zeros_like(y)
    f[0, ...] = y[1, ...]
    f[1, ...] = (
        -parameters["b_0"] * y[1, ...] - parameters["k"] * y[0, ...] + f_stribeck
    )

    df_dy = np.zeros((y.shape[0], y.shape[0], *y.shape[1:]))
    df_dy[0, 1, ...] = 1
    df_dy[1, 0, ...] = -parameters["k"]
    df_dy[1, 1, ...] = -parameters["b_0"] + derivatives_stribeck["y_dot"]

    df_dv = np.zeros_like(y)
    df_dv[1, ...] = derivatives_stribeck["v"]

    return f, {"y": df_dy, "v": df_dv}


def plot_stribeck(v: float, C: float, C_0: float, c_v: float, b_s: float, **_):

    # |y_dot - v| > delta_v_max implies that exponential decay is converged up to 1e-3
    delta_v_max = (3 + np.log(C_0 - C)) / c_v
    x_dot = -np.linspace(0, delta_v_max, 270)
    delta_speeds = v - x_dot
    forces, _ = force_stribeck(y_dot=x_dot, v=v, C=C, C_0=C_0, c_v=c_v, b_s=b_s)
    forces[0] = v

    kinetic = C + delta_speeds * b_s

    plt.figure()
    plt.plot(delta_speeds, forces, "k")
    plt.plot(delta_speeds, kinetic, "--g")
    plt.plot(-delta_speeds, -forces, "k")
    plt.plot(-delta_speeds, -kinetic, "--g")
    plt.show()


if __name__ == "__main__":
    main()
    plot_stribeck(**stick_slip_parameters(m=1, C_0=2, C=1, b_s=0.5, v=0, c_v=2))
