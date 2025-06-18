import numpy as np


def vanderpol(
    t: float | np.ndarray, x: np.ndarray, nu: float
) -> tuple[np.ndarray, dict[str, np.array]]:
    """
    Computes the dynamics and derivatives of the Van der Pol oscillator:
    ddot{x} + nu*(x**2 - 1)*dot{x} + x = 0
    Parameters
    ----------
    t : float or np.ndarray
        Current time or array of time values (not used in computation, included for API compatibility).
    x : np.ndarray
        State vector(s) of the system. Should have shape (2, ...) where the first row is position and the second is velocity.
    nu : float
        Nonlinearity/damping parameter of the Van der Pol oscillator.
    Returns
    -------
    f : np.ndarray
        Time derivative(s) of the state vector(s), same shape as `x`.
    derivatives : dict[str, np.ndarray]
        Dictionary containing partial derivatives:
            - "x": Jacobian of `f` with respect to `x`, shape (2, 2, ...).
            - "nu": Partial derivative of `f` with respect to `nu`, shape as `x`.
            - "T": Placeholder for time derivative (zeros), shape as `x`.
    """

    f = np.zeros_like(x)
    f[0, ...] = x[1, ...]
    f[1, ...] = nu * (1 - x[0, ...] ** 2) * x[1, ...] - x[0, ...]

    df_dx = np.zeros((2, 2, *x.shape[1:]), dtype=x.dtype)
    df_dx[0, 1, ...] = 1
    df_dx[1, 0, ...] = -1 - 2 * nu * x[0, ...] * x[1, ...]
    df_dx[1, 1, ...] = nu * (1 - x[0, ...] ** 2)

    df_dnu = np.zeros_like(x)
    df_dtau = np.zeros_like(x)
    df_dnu[1, ...] = (1 - x[0, ...] ** 2) * x[1, ...]

    derivatives = dict()
    derivatives["x"] = df_dx
    derivatives["nu"] = df_dnu
    derivatives["T"] = df_dtau

    return f, derivatives


def truss(
    t: float | np.ndarray,
    x: np.ndarray,
    k: float,
    c: float,
    F: float,
    a: float,
    l_0: float,
    m: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Dynamics  of a truss system.
    The truss system is given by a block (mass 'm') that can move in horizontal direction under viscous damping ('c').
    It is attached to a spring (unforced length 'l_0', spring constant 'k') whose other end is fixed at vertical height 'a'.
    The block is forced by 'F' in horizontal direction.
    This system can have up to three equilibria, depending on 'F'.
    Parameters
    ----------
    t : float or np.ndarray
        Current time or array of time values.
    x : np.ndarray
        State vector of the system, where x[0, ...] is position (q) and x[1, ...] is velocity (q_dot).
    k : float
        Spring constant.
    c : float
        Damping coefficient.
    F : float
        External force applied to the system.
    a : float
        Height of the spring.
    l_0 : float
        Rest length of the spring.
    m : float
        Mass of the system.
    Returns
    -------
    f : np.ndarray
        Time derivative of the state vector (i.e., [q_dot, q_ddot]).
    derivatives : dict[str, np.ndarray]
        Dictionary containing partial derivatives of the system dynamics with respect to parameters:
        - "k": Derivative with respect to spring constant.
        - "F": Derivative with respect to external force.
        - "c": Derivative with respect to damping coefficient.
        - "x": Jacobian of the system with respect to the state vector.
    """
    q = x[0, ...]
    q_dot = x[1, ...]

    derivatives = dict()

    f = np.zeros_like(x)
    f[1, ...] = -k / m * q
    f[1, ...] += k / m * q * l_0 / np.sqrt(a**2 + q**2)
    derivatives["k"] = f / k

    f[1, ...] += F / m - c / m * q_dot
    derivatives["F"] = np.zeros_like(x)
    derivatives["F"][1, ...] = 1 / m
    derivatives["c"] = np.zeros_like(x)
    derivatives["c"][1, ...] = x[1, ...] / m

    derivatives["x"] = np.zeros((f.shape[0], f.shape[0], *f.shape[1:]))
    derivatives["x"][0, 1, ...] = 1
    derivatives["x"][1, 1, ...] = -c / m
    derivatives["x"][1, 0, ...] = -k / m
    derivatives["x"][1, 0, ...] += k / m * l_0 / np.sqrt(a**2 + x[0, ...] ** 2)
    derivatives["x"][1, 0, ...] -= k / m * l_0 * q**2 / (np.sqrt(a**2 + q**2) ** 3)

    derivatives["x"] = derivatives["x"]
    derivatives["F"] = derivatives["F"]
    derivatives["c"] = derivatives["c"]
    derivatives["k"] = derivatives["k"]

    return f, derivatives


def blockonbelt(
    t: float | np.ndarray,
    x: np.ndarray,
    epsilon: float,
    k: float,
    m: float,
    Fs: float,
    vdr: float,
    delta: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Dynamics of a block-on-belt system with stick-slip motion and regularized Coulomb friction.
    Parameters
    ----------
    t : float or np.ndarray
        Current time or array of time points.
    x : np.ndarray
        State vector(s) of the system. The first row represents position(s), the second row velocity(ies).
    epsilon : float
        Regularization parameter for the friction law.
    k : float
        Spring constant.
    m : float
        Mass of the block.
    Fs : float
        Normal force.
    vdr : float
        Driving velocity of the belt.
    delta : float
        Slope parameter for the regularization of friction.
    Returns
    -------
    f : np.ndarray
        Time derivative(s) of the state vector(s).
    derivatives : dict[str, np.ndarray]
        Dictionary containing the Jacobian of the system with respect to the state vector, under the key "x".
    """

    gamma_T = x[1, ...] - vdr
    F_T = -Fs / (1 + delta * np.abs(gamma_T)) * 2 / np.pi * np.arctan(epsilon * gamma_T)

    f = np.zeros_like(x)
    f[0, ...] = x[1, ...]
    f[1, ...] = -k / m * x[0, ...] + F_T / m

    df_dx = np.zeros((2, 2, *x.shape[1:]), dtype=x.dtype)
    df_dx[0, 1, ...] = 1
    df_dx[1, 0, ...] = -k / m
    df_dx[1, 1, ...] = (Fs / m) * (
        (np.arctan(epsilon * gamma_T) / (1 + delta * np.abs(gamma_T)) ** 2)
        * (delta * np.sign(gamma_T) * 2 / np.pi)
        - (1 / (1 + delta * np.abs(gamma_T)))
        * (2 / np.pi)
        / (1 + (epsilon * gamma_T) ** 2)
        * epsilon
    )

    derivatives = dict()
    derivatives["x"] = df_dx
    return f.squeeze(), derivatives
