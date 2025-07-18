import numpy as np
from skhippr.systems.AbstractSystems import FirstOrderODE


class Vanderpol(FirstOrderODE):
    """
    Van der Pol oscillator as a first-order autonomous ODE system.

    Parameters
    ----------

    x : np.ndarray
        Initial state vector of shape (2, ...), where the first dimension corresponds to the two degrees of freedom.
    nu : float
        Nonlinearity/damping parameter of the Van der Pol oscillator.

    Attributes
    ----------

    nu : float
        Nonlinearity/damping parameter.
    x : np.ndarray
        State vector.

    Methods
    -------

    parse_kwargs(**kwargs)
        Parses keyword arguments for state and parameter values.
    dynamics(t=None, **kwargs)
        Computes the time derivative of the state vector according to the Van der Pol equations.
    derivative(variable, **kwargs)
        Computes the derivative of the dynamics with respect to a given variable ('x' or 'nu').

    Raises
    ------

    ValueError
        If the state vector `x` does not have the correct shape.
    AttributeError
        If the requested derivative variable is not recognized.
    """

    def __init__(self, x: np.ndarray, nu: float, **kwargs):
        super().__init__(autonomous=True, n_dof=2)
        self.nu = nu
        self.x = x
        self.t = kwargs.get("t", None)

    def parse_kwargs(self, **kwargs):
        x = kwargs.get("x", self.x)
        nu = kwargs.get("nu", self.nu)

        if x.shape[0] != self.n_dof:
            raise ValueError("first dimension of x must have length n_dof=2")

        return x, nu

    def dynamics(self, t=None, **kwargs):
        """
        Calculates the dynamics of the system at a given time.

        Parameters
        ----------

        t : float, optional
            The current time. Default is None.
        **kwargs : dict
            Additional keyword arguments. Can contain 'x' and 'nu', which are otherwise onbtained from self.x and self.nu.

        Returns
        -------

        f : ndarray
            The computed derivatives of the state variables.

        Notes
        -----

        This function expects `kwargs` to contain the state `x` and input `nu`, which are parsed using `self.parse_kwargs`.
        The dynamics are defined as:
            f[0] = x[1]
            f[1] = nu * (1 - x[0]**2) * x[1] - x[0]
        """

        x, nu = self.parse_kwargs(**kwargs)
        f = np.zeros_like(x)
        f[0, ...] = x[1, ...]
        f[1, ...] = nu * (1 - x[0, ...] ** 2) * x[1, ...] - x[0, ...]
        return f

    def derivative(self, variable, **kwargs):
        """
        Compute the derivative of the Van der Pol system with respect to a given variable.

        Parameters
        ----------

        variable : str
            The variable with respect to which the derivative is computed.
            Supported values are "x" (state variable) and "nu" (system parameter).
        **kwargs
            Additional keyword arguments. May contain:
                x : np.ndarray
                    State variable array of shape (2, ...).
                nu : float or np.ndarray
                    System parameter(s).
            By default, these values are set by self.x and self.nu.

        Returns
        -------

        np.ndarray
            The derivative of the system with respect to the specified variable:
            - If `variable` is "x", returns the Jacobian matrix with respect to `x` of shape (2, 2, ...).
            - If `variable` is "nu", returns the derivative with respect to `nu` of shape (2, ...).

        Raises
        ------

        AttributeError
            If the specified variable is not "x" or "nu".
        """

        x, nu = self.parse_kwargs(**kwargs)
        match variable:
            case "x":
                df_dx = np.zeros((2, 2, *x.shape[1:]), dtype=x.dtype)
                df_dx[0, 1, ...] = 1
                df_dx[1, 0, ...] = -1 - 2 * nu * x[0, ...] * x[1, ...]
                df_dx[1, 1, ...] = nu * (1 - x[0, ...] ** 2)
                return df_dx
            case "nu":
                df_dnu = np.zeros_like(x)
                df_dnu[1, ...] = (1 - x[0, ...] ** 2) * x[1, ...]
                return df_dnu
            case _:
                raise AttributeError(
                    f"{x} is not a parameter of vanderpol, derivative thus undefined"
                )


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
