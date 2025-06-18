import numpy as np
from scipy.linalg import expm

"""Hill-type equations"""


def _hill_eq(
    y: np.ndarray,
    g: np.ndarray,
    g_deriv: dict[str, np.ndarray],
    d: float = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Compute the time derivative and parameter derivatives for a Hill-type linear ODE of the form:
        ddot{x} + d*dot{x} + g(t)*x = 0
    expressed in first-order form y = [x, dot{x}].
    Parameters
    ----------
    y : np.ndarray
        State vector, where y[0] is x and y[1] is dot{x}.
    g : np.ndarray
        Time-dependent function fpr parametric excitation.
    g_deriv : dict[str, np.ndarray]
        Dictionary of derivatives of g with respect to parameters.
    d : float, optional
        Damping coefficient (default is 0).
    Returns
    -------
    dy_dt : np.ndarray
        Time derivative of the state vector y.
    g_deriv : dict[str, np.ndarray]
        Updated dictionary including derivatives of the system with respect to parameters,
        the Jacobian with respect to y ("y"), and the derivative with respect to d ("d").
    """

    dy_dt = np.zeros_like(y)
    dy_dt[0, ...] = y[1, ...]
    dy_dt[1, ...] = -d * y[1, ...] - g * y[0, ...]

    J = np.zeros((2, 2, *y.shape[1:]), dtype=y.dtype)
    J[0, 1, ...] = 1
    J[1, 0, ...] = -g
    J[1, 1, ...] = -d

    for key in g_deriv:
        df_dkey = np.zeros_like(y)
        df_dkey[1, ...] = -g_deriv[key]
        g_deriv[key] = df_dkey * y[0, ...]

    g_deriv["y"] = J
    g_deriv["d"] = np.zeros_like(y)
    g_deriv["d"][1, ...] = -y[1, ...]

    return dy_dt, g_deriv


def lti_g(t, gamma_sq: float) -> tuple[np.ndarray, dict[str, int]]:
    """
    Compute the trivial function g(t) = gamma_sq (constant) and its derivative with respect to gamma_sq.

    Parameters
    ----------
    t : array_like
        Input array representing time or another independent variable.
    gamma_sq : float
        The constant value to scale the output.

    Returns
    -------
    g : np.ndarray
        Array of the same shape as `t`, where each element is equal to `gamma_sq`.
    derivatives : dict[str, int]
        Dictionary containing the derivative of g with respect to gamma_sq, with key "gamma_sq:".
    """
    dg_dgammasq = np.ones_like(t)
    g = gamma_sq * dg_dgammasq
    return g, {"gamma_sq:": dg_dgammasq}


def lti(t, y, gamma_sq, d=0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    System function for a second-order linear time-invariant (LTI) system of the form:
        ddot{x} + d*dot{x} + gamma_sq*x = 0
    as special case (independent of t) of Hill's equation.

    Parameters:
        t (float): The current time.
        y (array-like): The current state vector [y, dot{y}].
        gamma_sq (float): The squared natural frequency parameter.
        d (float, optional): The damping coefficient. Default is 0.

    Returns:
        dydt (array-like): The derivatives [dot{x}, ddot{x}] at time t.
        derivatives: The derivatives of dydt with respect to the arguments.

    Notes:
        This function uses lti_g to compute the system's coefficients and delegates
        the computation to _hill_eq for the state derivatives.
    """
    g, g_deriv = lti_g(t, gamma_sq)
    return _hill_eq(y=y, g=g, g_deriv=g_deriv, d=d)


def lti_fundamental_matrix(t, t_0, gamma_sq, d=0) -> np.ndarray:
    """
    Compute the fundamental matrix of a linear time-invariant (LTI) system using the matrix exponential.

    Parameters:
        t (float): The final time at which to evaluate the fundamental matrix.
        t_0 (float): The initial time.
        gamma_sq (float or array-like): System parameter(s) used in the LTI system.
        d (float, optional): Additional system parameter, default is 0.

    Returns:
        numpy.ndarray: The fundamental matrix evaluated at (t - t_0).

    Notes:
        The matrix exponential is computed using `scipy.linalg.expm`.
    """
    _, derivs = lti(0, np.array([0.0, 0.0]), gamma_sq, d=d)
    return expm(derivs["y"] * (t - t_0))


def mathieu_g(t, a, b, omega) -> tuple[str, dict[str, np.ndarray]]:
    """
    System function for the Mathieq equation:
        ddot{x} + d*dot{x} + (a + b*cos(omega*t))x = 0
    as special case (independent of t) of Hill's equation.

    Parameters:
        t (float): The current time.
        y (array-like): The current state vector [y, dot{y}].
        gamma_sq (float): The squared natural frequency parameter.
        d (float, optional): The damping coefficient. Default is 0.

    Returns:
        dydt (array-like): The derivatives [dot{y}, ddot{y}] at time t.

    Notes:
        This function uses lti_g to compute the system's coefficients and delegates
        the computation to _hill_eq for the state derivatives.
    """
    dg_db = np.cos(omega * t)
    g = a + b * dg_db
    dg_da = 1
    dg_dom = -omega * np.sin(omega * t)

    return g, {"a": dg_da, "b": dg_db, "omega": dg_dom}


def mathieu(t, y, a, b, d, omega) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Solves Hill's equation with a periodic coefficient of the form g(t) = a + b*cos(omega*t).

    Parameters:
        t (float or np.ndarray): time
        y (np.ndarray): state
        a (float): The constant term in the periodic coefficient.
        b (float): The amplitude of the cosine term in the periodic coefficient.
        d (float): The damping coefficient.
        omega (float): The angular frequency of the cosine term.

    Returns:
        A tuple containing the derivative of the state vector and a dictionary with additional computed arrays.
    """
    g, g_deriv = mathieu_g(t, a, b, omega)
    return _hill_eq(y, g, g_deriv, d)


def meissner_g(
    t: np.ndarray, a: float, b: float, omega: float, smoothing: float = 0
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Compute function g(t) for the Meissner equation and its derivatives with respect to parameters.
    The function g(t) is defined as:
        g(t) = a + b * sign(cos(omega*t))
    where sign(0) is 0.
    If `smoothing` is nonzero, the jumps are smoothed using a continuous approximation:
    sign(cos(x)) \approx cos(x)/sqrt(cos(x)^2 + smoothing * sin(x)^2)
    For `smoothing == 1`, the function coincides with the Mathieu function.
    Parameters
    ----------
    t : np.ndarray
        Array of time points at which to evaluate the function.
    a : float
        Constant offset parameter.
    b : float
        Amplitude parameter.
    omega : float
        Frequency parameter.
    smoothing : float, optional
        Smoothing parameter between 0 and 1 (default is 0, i.e., no smoothing).
    Returns
    -------
    g : np.ndarray
        The computed values of the Meissner function at each time point.
    derivatives : dict[str, np.ndarray]
        Dictionary containing the derivatives of g with respect to 'a', 'b', and 'omega'
    """
    cos = np.cos(omega * t)
    if smoothing == 0:
        dg_db = np.sign(cos)
    else:
        dg_db = cos / np.sqrt(cos**2 + smoothing * np.sin(omega * t) ** 2)
    g = a + b * dg_db
    dg_da = 1
    dg_dom = 0  # undefined at jump points

    return g, {"a": dg_da, "b": dg_db, "omega": dg_dom}


def meissner_fourier_g(
    t, a, b, omega, N_HBM=30
) -> tuple[np.array, dict[str, np.array]]:
    """
    Compute the approximation of the function g(t) for the Meissner equation
    and its derivatives with respect to parameters a and b
    using a truncated Fourier series representation (with N_HBM harmonics).

    Parameters
    ----------
    t : array_like
        Time values at which to evaluate the function.
    a : float or array_like
        Offset parameter for the function.
    b : float or array_like
        Amplitude parameter for the function.
    omega : float
        Angular frequency of the Fourier series.
    N_HBM : int, optional
        Number of harmonics to include in the Fourier series (default is 30).

    Returns
    -------
    g : ndarray
        The computed values of g(t).
    derivatives : dict of ndarray
        Dictionary containing the derivatives of g(t) with respect to 'a' and 'b'
    """
    dg_db = np.zeros_like(t)
    factor = 4 / np.pi
    for k in range(1, N_HBM + 1, 2):
        coeff = factor / k * (-1) ** (0.5 * (k - 1))
        dg_db += coeff * np.cos(k * omega * t)
    g = b * dg_db + a
    return g, {"a": np.ones_like(t), "b": dg_db}


def meissner(t, y, a, b, d, omega, smoothing=0) -> tuple[str, dict[str, np.ndarray]]:
    g, g_deriv = meissner_g(t, a, b, omega, smoothing)
    return _hill_eq(y, g, g_deriv, d)


def meissner_fourier(t, y, a, b, d, omega, N_HBM) -> tuple[str, dict[str, np.ndarray]]:
    g, g_deriv = meissner_fourier_g(t, a, b, omega, N_HBM)
    return _hill_eq(y, g, g_deriv, d)


def meissner_fundamental_matrix(t_end, t_0, a, b, d, omega):
    # Meissner fundamental matrix pieced together by LTI fundamental matrices
    T = 2 * np.pi / omega
    gamma_sq_1 = a + b  # when t in (-T/4, T/4)
    gamma_sq_2 = a - b  # when t in (T/4, 3*T/4)

    Phi_t = np.eye(N=2)

    # Determine current point on peri. sol.
    k_switch = np.floor(t_0 / (0.25 * T))
    t_switch = 0.25 * T * k_switch

    k_switch = np.mod(k_switch, 4)

    while t_end > t_0:
        if k_switch == 0 or k_switch == 3:
            gamma_sq = gamma_sq_1
        else:
            gamma_sq = gamma_sq_2

        t_switch_next = t_switch + 0.25 * T

        Phi_t = (
            lti_fundamental_matrix(min(t_switch_next, t_end), t_0, gamma_sq, d=d)
            @ Phi_t
        )

        t_switch = t_switch_next
        t_0 = t_switch
        k_switch = np.mod(k_switch + 1, 4)

    return Phi_t


def shirley(
    t, y, E_alpha, E_beta, b, omega
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # cf. Shirley1965: 10.1103/physrev.138.b979
    J = np.zeros(2, 2, *y.shape[1:], dtype=complex)
    J[0, 0, ...] = -1j * E_alpha
    J[0, 1, ...] = -1j * 2 * b * np.cos(omega * t)
    J[1, 0, ...] = J[0, 1, ...]
    J[1, 1, ...] = -1j * E_beta

    dy_dt = np.zeros_like(y, dtype=complex)
    if len(y.shape) == 2:
        for k in range(y.shape[1]):
            dy_dt[:, k] = J[:, :, k] @ y[:, k]

    else:
        dy_dt = J @ y

    return dy_dt, {"y": J}
