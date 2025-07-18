import numpy as np
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

    def dynamics(self, **kwargs) -> tuple[np.ndarray, dict[str, np.ndarray]]:

        t = kwargs.get("t", self.t)
        x = kwargs.get("x", self.x)
        alpha = kwargs.get("alpha", self.alpha)
        beta = kwargs.get("beta", self.beta)
        omega = kwargs.get("omega", self.omega)
        F = kwargs.get("F", self.F)
        delta = kwargs.get("delta", self.delta)

        if x.shape[0] != self.n_dof:
            raise ValueError("first dimension of x must have length n_dof=2")

        f = np.zeros_like(x)
        f[0, ...] = x[1, ...]
        f[1, ...] = (
            -alpha * x[0, ...]
            - delta * x[1, ...]
            - beta * x[0, ...] ** 3
            + F * np.cos(omega * t)
        )

        return f

    def derivative(self, variable, **kwargs):
        match variable:
            case "x":
                return self.df_dx(**kwargs)
            case "omega":
                return self.df_dom(**kwargs)
            case "F":
                return self.df_dF(**kwargs)
            case "alpha":
                return self.df_dalpha(**kwargs)
            case "beta":
                return self.df_dbeta(**kwargs)
            case "delta":
                return self.df_ddelta(**kwargs)
            case _:
                return super().derivative(variable, **kwargs)

    def df_dx(self, **kwargs):

        x = kwargs.get("x", self.x)
        alpha = kwargs.get("alpha", self.alpha)
        beta = kwargs.get("beta", self.beta)
        delta = kwargs.get("delta", self.delta)

        df_dx = np.zeros((2, 2, *x.shape[1:]), dtype=x.dtype)
        df_dx[0, 1, ...] = 1
        df_dx[1, 0, ...] = -alpha - 3 * beta * x[0, ...] ** 2
        df_dx[1, 1, ...] = -delta

        return df_dx

    def df_dalpha(self, **kwargs):
        x = kwargs.get("x", self.x)

        df_dal = np.zeros_like(x)
        df_dal[1, ...] = -x[0, ...]

        return df_dal

    def df_dbeta(self, **kwargs):
        x = kwargs.get("x", self.x)

        df_dbe = np.zeros_like(x)
        df_dbe[1, ...] = -x[0, ...] ** 3
        return df_dbe

    def df_ddelta(self, **kwargs):
        x = kwargs.get("x", self.x)

        df_ddel = np.zeros_like(x)
        df_ddel[1, ...] = -x[1, ...]

        return df_ddel

    def df_dF(self, **kwargs):
        t = kwargs.get("t", self.t)
        x = kwargs.get("x", self.x)
        omega = kwargs.get("omega", self.omega)

        df_dF = np.zeros_like(x)
        df_dF[1, ...] = np.cos(omega * t)
        return df_dF

    def df_dom(self, **kwargs):

        t = kwargs.get("t", self.t)
        x = kwargs.get("x", self.x)
        omega = kwargs.get("omega", self.omega)
        F = kwargs.get("F", self.F)

        df_dom = np.zeros_like(x)
        df_dom[1, ...] = -F * t * np.sin(omega * t)
        return df_dom
