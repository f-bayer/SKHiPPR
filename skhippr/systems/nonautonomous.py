import numpy as np
from typing import override
from skhippr.systems.AbstractSystems import FirstOrderODE


class Duffing(FirstOrderODE):
    def __init__(
        self,
        t: float,
        x: np.ndarray,
        omega: float,
        alpha: float,
        beta: float,
        F: float,
        delta: float,
    ):
        super().__init__(autonomous=False, n_dof=2)
        self.t = t
        self.x = x
        self.alpha = alpha
        self.beta = beta
        self.F = F
        self.omega = omega
        self.delta = delta

    @override
    def dynamics(self, t=None, x=None) -> np.ndarray:

        if t is None:
            t = self.t
        if x is None:
            x = self.x
        self.check_dimensions(t, x)

        f = np.zeros_like(x)
        f[0, ...] = x[1, ...]
        f[1, ...] = (
            -self.alpha * x[0, ...]
            - self.delta * x[1, ...]
            - self.beta * x[0, ...] ** 3
            + self.F * np.cos(self.omega * t)
        )

        return f

    @override
    def closed_form_derivative(self, variable, t=None, x=None):

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        self.check_dimensions(t, x)

        match variable:
            case "x":
                return self.df_dx(t, x)
            case "omega":
                return self.df_dom(t, x)
            case "F":
                return self.df_dF(t, x)
            case "alpha":
                return self.df_dalpha(t, x)
            case "beta":
                return self.df_dbeta(t, x)
            case "delta":
                return self.df_ddelta(t, x)
            case _:
                raise NotImplementedError

    def df_dx(self, t=None, x=None):

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        df_dx = np.zeros((2, *x.shape), dtype=x.dtype)
        df_dx[0, 1, ...] = 1
        df_dx[1, 0, ...] = -self.alpha - 3 * self.beta * x[0, ...] ** 2
        df_dx[1, 1, ...] = -self.delta

        return df_dx

    def df_dalpha(self, t=None, x=None):

        if x is None:
            x = self.x

        df_dal = np.zeros_like(x)
        df_dal[1, ...] = -x[0, ...]

        return df_dal

    def df_dbeta(self, t=None, x=None):

        if x is None:
            x = self.x

        df_dbe = np.zeros_like(x)
        df_dbe[1, ...] = -x[0, ...] ** 3
        return df_dbe

    def df_ddelta(self, t=None, x=None):

        if x is None:
            x = self.x
        df_ddel = np.zeros_like(x)
        df_ddel[1, ...] = -x[1, ...]

        return df_ddel

    def df_dF(self, t=None, x=None):

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        df_dF = np.zeros_like(x)
        df_dF[1, ...] = np.cos(self.omega * t)
        return df_dF

    def df_dom(self, t=None, x=None):

        if t is None:
            t = self.t
        if x is None:
            x = self.x

        df_dom = np.zeros_like(x)
        df_dom[1, ...] = -self.F * t * np.sin(self.omega * t)
        return df_dom
