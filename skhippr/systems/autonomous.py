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


class Truss(FirstOrderODE):

    def __init__(
        self,
        x: np.ndarray,
        k: float,
        c: float,
        F: float,
        a: float,
        l_0: float,
        m: float,
        **kwargs,
    ):
        super().__init__(autonomous=True, n_dof=2)
        self.t = kwargs.get("t", None)
        self.x = x
        self.k = k
        self.c = c
        self.F = F
        self.a = a
        self.l_0 = l_0
        self.m = m

    def parse_kwargs(self, **kwargs):
        x = np.atleast_1d(kwargs.get("x", self.x))
        k = kwargs.get("k", self.k)
        c = kwargs.get("c", self.c)
        F = kwargs.get("F", self.F)
        a = kwargs.get("a", self.a)
        l_0 = kwargs.get("l_0", self.l_0)
        m = kwargs.get("m", self.m)

        if x.shape[0] != self.n_dof:
            raise ValueError("first dimension of x must have length n_dof=2")

        return x, k, c, F, a, l_0, m

    def dynamics(self, **kwargs):
        x, k, c, F, a, l_0, m = self.parse_kwargs(**kwargs)
        q = x[0, ...]
        q_dot = x[1, ...]

        f = np.zeros_like(x)
        f[1, ...] = -k / m * q
        f[1, ...] += k / m * q * l_0 / np.sqrt(a**2 + q**2)
        f[1, ...] += F / m - c / m * q_dot
        return f

    def df_dF(self, **kwargs):
        x, _, _, _, _, _, m = self.parse_kwargs(**kwargs)
        df_dF = np.zeros_like(x)
        df_dF[1, ...] = 1 / m
        return df_dF

    def df_dc(self, **kwargs):
        x, _, _, _, _, _, _, m = self.parse_kwargs(**kwargs)
        df_dc = np.zeros_like(x)
        df_dc[1, ...] = x[1, ...] / m
        return df_dc

    def df_dk(self, **kwargs):
        x, _, _, _, a, l_0, m = self.parse_kwargs(**kwargs)
        q = x[0, ...]
        df_dk = np.zeros_like(x)
        df_dk[1, ...] = -1 / m * q
        df_dk[1, ...] += 1 / m * q * l_0 / np.sqrt(a**2 + q**2)
        return df_dk

    def df_dx(self, **kwargs):
        x, k, c, _, a, l_0, m = self.parse_kwargs(**kwargs)
        q = x[0, ...]

        df_dx = np.zeros((x.shape[0], x.shape[0], *x.shape[1:]))
        df_dx[0, 1, ...] = 1
        df_dx[1, 1, ...] = -c / m
        df_dx[1, 0, ...] = -k / m
        df_dx[1, 0, ...] += k / m * l_0 / np.sqrt(a**2 + x[0, ...] ** 2)
        df_dx[1, 0, ...] -= k / m * l_0 * q**2 / (np.sqrt(a**2 + q**2) ** 3)
        return df_dx


class BlockOnBelt(FirstOrderODE):

    def __init__(
        self,
        x: np.ndarray,
        epsilon: float,
        k: float,
        m: float,
        Fs: float,
        vdr: float,
        delta: float,
    ):
        super().__init__(True, 2)
        self.x = x
        self.epsilon = epsilon
        self.k = k
        self.m = m
        self.Fs = Fs
        self.vdr = vdr
        self.delta = delta

    def parse_kwargs(self, **kwargs):
        x = np.atleast_1d(kwargs.get("x", self.x))
        epsilon = kwargs.get("epsilon", self.epsilon)
        k = kwargs.get("k", self.k)
        m = kwargs.get("m", self.m)
        Fs = kwargs.get("Fs", self.Fs)
        vdr = kwargs.get("vdr", self.vdr)
        delta = kwargs.get("delta", self.delta)

        if x.shape[0] != self.n_dof:
            raise ValueError("first dimension of x must have length n_dof=2")
        return x, epsilon, k, m, Fs, vdr, delta

    def dynamics(self, **kwargs):
        x, epsilon, k, m, Fs, vdr, delta = self.parse_kwargs(**kwargs)

        gamma_T = x[1, ...] - vdr
        F_T = (
            -Fs
            / (1 + delta * np.abs(gamma_T))
            * 2
            / np.pi
            * np.arctan(epsilon * gamma_T)
        )

        f = np.zeros_like(x)
        f[0, ...] = x[1, ...]
        f[1, ...] = -k / m * x[0, ...] + F_T / m
        return f

    def derivative(self, variable, **kwargs):
        if variable == "x":
            x, epsilon, k, m, Fs, vdr, delta = self.parse_kwargs(**kwargs)
            gamma_T = x[1, ...] - vdr
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
            return df_dx
        else:
            return super().derivative(variable, **kwargs)
