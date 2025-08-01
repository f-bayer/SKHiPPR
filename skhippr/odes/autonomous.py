import numpy as np
from typing import override
from skhippr.odes.AbstractODE import AbstractODE


class Vanderpol(AbstractODE):
    """
    Autonomous van der Pol oscillator as a subclass of :py:class:`~skhippr.odes.AbstractODE.AbstractODE`. ::

        dx[0]/dt = x[1]
        dx[1]/dt = nu * (1 - x[0]**2) * x[1] - x[0]

    """

    def __init__(self, x: np.ndarray, nu: float, t=0):
        super().__init__(autonomous=True, n_dof=2)
        self.nu = nu
        self.x = x
        self.t = t

    @override
    def dynamics(self, t=None, x=None):

        if x is None:
            x = self.x

        self.check_dimensions(x=x)

        f = np.zeros_like(x)
        f[0, ...] = x[1, ...]
        f[1, ...] = self.nu * (1 - x[0, ...] ** 2) * x[1, ...] - x[0, ...]
        return f

    @override
    def closed_form_derivative(self, variable, t=None, x=None):
        if x is None:
            x = self.x

        self.check_dimensions(t=t, x=x)

        match variable:
            case "x":
                df_dx = np.zeros((2, *x.shape), dtype=x.dtype)
                df_dx[0, 1, ...] = 1
                df_dx[1, 0, ...] = -1 - 2 * self.nu * x[0, ...] * x[1, ...]
                df_dx[1, 1, ...] = self.nu * (1 - x[0, ...] ** 2)
                return df_dx
            case "nu":
                df_dnu = np.zeros_like(x)
                df_dnu[1, ...] = (1 - x[0, ...] ** 2) * x[1, ...]
                return df_dnu[:, np.newaxis, ...]
            case _:
                raise NotImplementedError(
                    f"Derivative w.r.t {variable} not implemented in closed form."
                )


class Truss(AbstractODE):
    """
    Autonomous truss system as a subclass of :py:class:`~skhippr.odes.AbstractODE.AbstractODE`. A mass ``m`` can move horizontally with viscous damping (``c``). It is attached to a linear spring (``k``, ``l_0``) mounted at the point ``(0, a)``. A constant force ``F`` acts on the mass. For small values of ``F``, there are three coexisting equilibria. The equations of motion are ::

        dx[0]/dt = x[1]
        dx[1]/dt = -k/m * x[0] + k/m * x[0] * l_0 / sqrt(a**2 + x[0]**2) + F/m - c/m * x[1]

    """

    def __init__(
        self,
        x: np.ndarray,
        k: float,
        c: float,
        F: float,
        a: float,
        l_0: float,
        m: float,
    ):
        super().__init__(autonomous=True, n_dof=2)
        self.x = x
        self.k = k
        self.c = c
        self.F = F
        self.a = a
        self.l_0 = l_0
        self.m = m

    def dynamics(self, t=None, x=None):
        if x is None:
            x = self.x

        self.check_dimensions(t=t, x=x)

        q = x[0, ...]
        q_dot = x[1, ...]

        f = np.zeros_like(x)
        f[0, ...] = q_dot
        f[1, ...] = -self.k / self.m * q
        f[1, ...] += self.k / self.m * q * self.l_0 / np.sqrt(self.a**2 + q**2)
        f[1, ...] = f[1, ...] + self.F / self.m - self.c / self.m * q_dot

        return f

    def closed_form_derivative(self, variable, t=None, x=None):

        if x is None:
            x = self.x

        self.check_dimensions(t=t, x=x)

        match variable:
            case "x":
                return self.df_dx(x)
            case "F":
                return self.df_dF(x)
            case "k":
                return self.df_dk(x)
            case "c":
                return self.df_dc(x)
            case _:
                raise NotImplementedError(
                    f"Derivative w.r.t {variable} not implemented in closed form."
                )

    def df_dF(self, x=None):
        if x is None:
            x = self.x
        df_dF = np.zeros_like(x)
        df_dF[1, ...] = 1 / self.m
        return df_dF[:, np.newaxis, ...]

    def df_dc(self, x=None):
        if x is None:
            x = self.x
        df_dc = np.zeros_like(x)
        df_dc[1, ...] = -x[1, ...] / self.m
        return df_dc[:, np.newaxis, ...]

    def df_dk(self, x=None):
        if x is None:
            x = self.x
        q = x[0, ...]
        df_dk = np.zeros_like(x)
        df_dk[1, ...] = -1 / self.m * q
        df_dk[1, ...] += 1 / self.m * q * self.l_0 / np.sqrt(self.a**2 + q**2)
        return df_dk[:, np.newaxis, ...]

    def df_dx(self, x=None):
        if x is None:
            x = self.x

        q = x[0, ...]

        df_dx = np.zeros((x.shape[0], x.shape[0], *x.shape[1:]))
        df_dx[0, 1, ...] = 1
        df_dx[1, 1, ...] = -self.c / self.m
        df_dx[1, 0, ...] = -self.k / self.m
        df_dx[1, 0, ...] += self.k / self.m * self.l_0 / np.sqrt(self.a**2 + q**2)
        df_dx[1, 0, ...] -= (
            self.k / self.m * self.l_0 * q**2 / (np.sqrt(self.a**2 + q**2) ** 3)
        )
        return df_dx


class BlockOnBelt(AbstractODE):
    """
    Smoothed Block-on-belt system as a subclass of :py:class:`~skhippr.odes.AbstractODE.AbstractODE`. A block with mass ``m`` is placed on a belt with constant velocity ``vdr``. The block is subject to a spring force ``Fs`` and smoothed coulomb friction with the belt. The equations of motion are ::

    dx[0]/dt = x[1]
    dx[1]/dt = -k/m * x[0] + Fs/m * 2/pi * arctan(epsilon * (x[1] - vdr)) / (1 + delta * abs(x[1] - vdr))

    """

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

        self.check_dimensions(t=None, x=x)

    def dynamics(self, t=None, x=None):
        if x is None:
            x = self.x

        self.check_dimensions(t=t, x=x)

        gamma_T = x[1, ...] - self.vdr
        F_T = (
            -self.Fs
            / (1 + self.delta * np.abs(gamma_T))
            * 2
            / np.pi
            * np.arctan(self.epsilon * gamma_T)
        )

        f = np.zeros_like(x)
        f[0, ...] = x[1, ...]
        f[1, ...] = -self.k / self.m * x[0, ...] + F_T / self.m
        return f

    def closed_form_derivative(self, variable, t=None, x=None):
        if x is None:
            x = self.x

        self.check_dimensions(t=t, x=x)

        if variable == "x":
            gamma_T = x[1, ...] - self.vdr
            df_dx = np.zeros((2, 2, *x.shape[1:]), dtype=x.dtype)
            df_dx[0, 1, ...] = 1
            df_dx[1, 0, ...] = -self.k / self.m
            df_dx[1, 1, ...] = (self.Fs / self.m) * (
                (
                    np.arctan(self.epsilon * gamma_T)
                    / (1 + self.delta * np.abs(gamma_T)) ** 2
                )
                * (self.delta * np.sign(gamma_T) * 2 / np.pi)
                - (1 / (1 + self.delta * np.abs(gamma_T)))
                * (2 / np.pi)
                / (1 + (self.epsilon * gamma_T) ** 2)
                * self.epsilon
            )
            return df_dx
        else:
            raise NotImplementedError(
                f"Derivative w.r.t {variable} not implemented in closed form."
            )
