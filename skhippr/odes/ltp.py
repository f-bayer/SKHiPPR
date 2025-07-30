import numpy as np
from scipy.linalg import expm

from skhippr.odes.AbstractODE import AbstractODE

"""Hill-type equations"""


class HillODE(AbstractODE):
    def __init__(self, t, x, g_fcn, a=0, b=1, omega=1, damping=0):
        """Encodes damped Hill ODEs of the form x_ddot + d*x_dot + (a + b*g(omega*t))*x"""
        super().__init__(False, 2)

        self.g_fcn = g_fcn

        self.t = t
        self.x = x
        self.a = a
        self.b = b
        self.omega = omega
        self.damping = damping

    def dynamics(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t

        J = self.closed_form_derivative("x", x=x)

        if len(x.shape) > 1:
            f = np.zeros_like(x)
            for k in range(x.shape[1]):
                f[:, k] = J[:, :, k, ...] @ x[:, k]

        else:
            f = J @ x

        return np.squeeze(f)

    def closed_form_derivative(self, variable, t=None, x=None):
        if t is None:
            t = self.t
        if x is None:
            x = self.x
        self.check_dimensions(t, x)

        match variable:
            case "x":
                J = np.zeros((x.shape[0], *x.shape))
                J[0, 1, ...] = 1
                J[1, 1, ...] = self.damping
                J[1, 0, ...] = self.a

                tau = self.omega * t
                J[1, 0, ...] += self.b * self.g_fcn(tau)
                return J

            case _:
                raise NotImplementedError(
                    f"Derivative w.r.t {variable} not implemented in closed form."
                )


class HillLTI(HillODE):
    def __init__(self, t, x, a=0, b=1, damping=0, omega=1):
        super().__init__(t, x, lambda t: 1, a, b, omega, damping)

    def fundamental_matrix(self, t, t_0=None) -> np.ndarray:
        """
        Compute the fundamental matrix of the linear time-invariant (LTI) Hill system using the matrix exponential.

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
        if t_0 is None:
            t_0 = self.t

        A = self.derivative("x")
        return expm(A * (t - t_0))


class SmoothedMeissner(HillODE):
    def __init__(self, t, x, smoothing=0, a=0, b=1, omega=1, damping=0):
        if not (0 <= smoothing <= 1):
            raise ValueError(
                f"Smoothing must lie between 0 (Meissner eq) and 1 (Mathieu eq) but is {smoothing}"
            )
        super().__init__(t, x, self.smoothed_rectangular, a, b, omega, damping)
        self.smoothing = smoothing

    def smoothed_rectangular(self, t):
        if self.smoothing == 0:
            return np.sign(np.cos(t))
        else:
            cos = np.cos(t)
            return cos / np.sqrt(cos**2 + self.smoothing * np.sin(t) ** 2)

    def fundamental_matrix(self, t_end: float, t_0: float = None) -> np.ndarray:
        if self.smoothing != 0:
            raise ValueError(
                "Fundamental matrix in closed form only available for un-smoothed Meissner equation!"
            )

        if t_0 is None:
            t_0 = self.t

        if t_end < t_0:
            raise ValueError(
                f"End time t_end ({t_end}) cannot be smaller than initial time t_0 {(t_0)}"
            )

        # Meissner fundamental matrix pieced together by LTI fundamental matrices
        T = 2 * np.pi / self.omega
        gamma_sq_1 = self.a + self.b  # when t in (-T/4, T/4)
        gamma_sq_2 = self.a - self.b  # when t in (T/4, 3*T/4)

        A = np.array([[0, 1], [gamma_sq_1, self.damping]])

        Phi_t = np.eye(N=2)

        # Determine current point in time.
        k_switch = np.floor(t_0 / (0.25 * T))
        t_switch = 0.25 * T * k_switch

        k_switch = np.mod(k_switch, 4)

        while t_end > t_0:
            if k_switch == 0 or k_switch == 3:
                A[1, 0] = gamma_sq_1
            else:
                A[1, 0] = gamma_sq_2

            t_switch_next = t_switch + 0.25 * T

            Phi_t = expm(A * (min(t_switch_next, t_end) - t_0)) @ Phi_t

            t_switch = t_switch_next
            t_0 = t_switch
            k_switch = np.mod(k_switch + 1, 4)

        return Phi_t


class MathieuODE(SmoothedMeissner):
    def __init__(self, t, x, a=0, b=1, omega=1, damping=0):
        super().__init__(t=t, x=x, smoothing=1, a=a, b=b, omega=omega, damping=damping)


class TruncatedMeissner(HillODE):
    def __init__(self, t, x, N_harmonics, a=0, b=1, omega=1, damping=0):

        self.N_harmonics = N_harmonics

        factor = 4 / np.pi
        self.harmonics = np.arange(start=1, stop=N_harmonics + 1, step=2)
        self.fourier_coeffs = (
            factor / self.harmonics * (-1) ** (0.5 * (self.harmonics - 1))
        )

        super().__init__(t, x, self.truncated_rectangular, a, b, omega, damping)

    def truncated_rectangular(self, t):
        t = np.atleast_1d(t)
        cos = np.cos(np.tensordot(self.harmonics, t, axes=0))
        # harmonic is now in 0-th dimension, t in others
        return np.squeeze(np.tensordot(self.b * self.fourier_coeffs, cos, axes=(0, 0)))


class ShirleyODE(AbstractODE):

    def __init__(self, t, x, E_alpha, E_beta, b, omega):
        super().__init__(autonomous=False, n_dof=2)
        self.t = t
        self.x = x
        self.E_alpha = E_alpha
        self.E_beta = E_beta
        self.b = b
        self.omega = omega

    pass

    def closed_form_derivative(self, variable, t=None, x=None):
        if t is None:
            t = self.t
        if x is None:
            x = self.x

        self.check_dimensions(t, x)

        match variable:
            case "x":
                # cf. Shirley1965: 10.1103/physrev.138.b979
                J = np.zeros((2, 2, *x.shape[1:]), dtype=complex)
                J[0, 0, ...] = -1j * self.E_alpha
                J[0, 1, ...] = -1j * 2 * self.b * np.cos(self.omega * t)
                J[1, 0, ...] = J[0, 1, ...]
                J[1, 1, ...] = -1j * self.E_beta
                return J
            case _:
                raise NotImplementedError(
                    f"Derivative w.r.t {variable} not implemented in closed form."
                )

    def dynamics(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t

        dx_dt = np.zeros_like(x, dtype=complex)
        J = self.closed_form_derivative(variable="x", t=t, x=x)
        if len(x.shape) > 1:
            for k in range(x.shape[1]):
                dx_dt[:, k, ...] = J[:, :, k, ...] @ x[:, k, ...]

        else:
            dx_dt = J @ x

        return dx_dt
