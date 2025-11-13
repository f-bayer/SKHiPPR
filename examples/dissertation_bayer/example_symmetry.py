"""Example system for failing symmetry criterion"""

from skhippr.odes.AbstractODE import AbstractODE
import numpy as np
from abc import abstractmethod
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from skhippr.Fourier import Fourier
from skhippr.solvers.newton import NewtonSolver
from skhippr.cycles.hbm import HBMEquation
from skhippr.stability.ClassicalHill import ClassicalHill


class ClosedFormSystem(AbstractODE):
    """Generate LTP systems with closed-form solutions."""

    def __init__(self, n_dof):
        super().__init__(False, n_dof=n_dof, stability_method=None)
        self.t = 0
        self.x = np.zeros((n_dof))
        self.check_integrity()

    @abstractmethod
    def fundamental_matrix(self, t, t_0=None): ...

    @abstractmethod
    def fundamental_matrix_inv(self, t, t_0=None): ...

    @abstractmethod
    def diff_fundamental_matrix(self, t): ...

    def dynamics(self, t=None, x=None):
        return self.closed_form_derivative("x", t=t) @ x

    def closed_form_derivative(self, variable, t=None, x=None):
        """
        Phi_dot(t) = J(t)*Phi(t)
        --> J(t) = Phi_dot(t) / Phi(t)
        """

        if t is None:
            t = self.t

        match variable:
            case "x":
                Phi_dot = self.diff_fundamental_matrix(t)
                Phi_inv = self.fundamental_matrix_inv(t=t, t_0=0)
                return Phi_dot @ Phi_inv
            case _:
                return super().closed_form_derivative(variable, t, x)

    def check_integrity(self, tol=1e-8):
        """Check the well-posedness of the fundamental matrix, its inverse and its derivative."""

        eye = np.eye(self.n_dof)
        dt = 100 * tol

        for t_0 in [0]:  # , 0.3]:

            # Check that Phi(t_0, t_0) == eye
            assert np.max(np.abs(eye - self.fundamental_matrix(t=t_0, t_0=t_0))) < tol

        for t in np.linspace(t_0 + 0.5, t_0 + 3):
            # Check Phi_dot using finite differences
            Phi = self.fundamental_matrix(t=t, t_0=0)
            Phi_diff_FD = (self.fundamental_matrix(t=t + dt, t_0=0) - Phi) / dt
            assert np.max(np.abs(Phi_diff_FD - self.diff_fundamental_matrix(t=t)))

            # Check inverse
            assert np.max(np.abs(Phi @ self.fundamental_matrix_inv(t, 0) - eye))


class SymmetrySystem(ClosedFormSystem):

    def __init__(self, k, gamma):
        if k % 1 != 0 or k % 2 == 0:
            raise ValueError(f"k must be an odd integer but is {k}")

        self.k = k
        self.gamma = gamma
        super().__init__(2)

    def fundamental_matrix(self, t, t_0=None) -> np.ndarray:
        """
        Define the fundamental matrix of the system.

        Parameters
        ----------
        t : float
            The final time at which to evaluate the fundamental matrix.
        t_0 : float, optional
            The initial time. If `None`, defaults to `self.t`.
        """
        if t_0 is None:
            t_0 = self.t

        cos = np.cos(0.5 * t)
        sin = self.gamma * np.sin(0.5 * self.k * t)

        Phi_t = np.array([[cos, -sin], [sin, cos]])

        # if t != 0:
        #     Phi_t0_inv = self.fundamental_matrix_inv(t=t_0, t_0=0)
        #     Phi_t = Phi_t @ Phi_t0_inv

        return Phi_t

    def fundamental_matrix_inv(self, t, t_0=0) -> np.ndarray:
        """
        Inverse of the fundamental solution matrix defined in self.fundamental_matrix.

        """

        cos = np.cos(0.5 * t)
        sin = self.gamma * np.sin(0.5 * self.k * t)
        det = cos**2 + sin**2

        Phi_t_inv = np.array([[cos, sin], [-sin, cos]]) / det

        # if t != 0:
        #     Phi_t0 = self.fundamental_matrix(t=t_0, t_0=0)
        #     Phi_t_inv = Phi_t0 @ Phi_t_inv

        return Phi_t_inv

    def diff_fundamental_matrix(self, t) -> np.ndarray:
        """
        Derivative of the fundamental solution matrix
        defined in self.fundamental_matrix(t, t_0=0) w.r.t. t.
        """

        diff_cos = -0.5 * np.sin(0.5 * t)
        diff_sin = 0.5 * self.k * self.gamma * np.cos(0.5 * self.k * t)

        diff_Phi_t = np.array(
            [
                [diff_cos, -diff_sin],
                [diff_sin, diff_cos],
            ]
        )

        return diff_Phi_t


class SymmetryAsODE(AbstractODE):
    def __init__(self, t, x, gamma, k):
        self.t = t
        self.x = x
        self.k = k
        self.gamma = gamma
        super().__init__(False, 2, None)

    def J(self, t):
        det = 0.5 / (1 + np.cos(t) + self.gamma**2 * (1 - np.cos(self.k * t)))
        sin = np.sin(t)
        sin_k = (self.gamma**2) * self.k * np.sin(self.k * t)
        cos_neg = self.gamma * (1 - self.k) * np.cos(0.5 * (self.k + 1) * t)
        cos_pos = self.gamma * (1 + self.k) * np.cos(0.5 * (self.k - 1) * t)

        return det * np.array(
            [[-sin + sin_k, cos_neg - cos_pos], [-cos_neg + cos_pos, -sin + sin_k]]
        )

        # cos = np.cos(0.5 * t)
        # sin = self.gamma * np.sin(0.5 * self.k * t)
        # det = cos**2 + sin**2

        # Phi_t_inv = np.array([[cos, sin], [-sin, cos]]) / det

        # diff_cos = -0.5 * np.sin(0.5 * t)
        # diff_sin = 0.5 * self.k * self.gamma * np.cos(0.5 * self.k * t)

        # diff_Phi_t = np.array(
        #     [
        #         [diff_cos, -diff_sin],
        #         [diff_sin, diff_cos],
        #     ]
        # )

        # return diff_Phi_t @ Phi_t_inv

    def closed_form_derivative(self, variable, t=None, x=None):

        if t is None:
            t = self.t

        match variable:
            case "x":
                return self.J(t)
            case _:
                return super().closed_form_derivative(variable, t, x)

    def dynamics(self, t=None, x=None):
        return self.J(t) @ x


def test_solutions(system, t_span, ode=None, **ode_kwargs):

    t_eval = np.linspace(*t_span, num=200)
    errors = []
    for idx_col, x_0 in enumerate(np.eye(system.n_dof)):
        sol = solve_ivp(system.dynamics, t_span, x_0, t_eval=t_eval, **ode_kwargs)
        # sol = solve_ivp(
        #     lambda t, x: ode.J(t) @ x, t_span, x_0, t_eval=t_eval, **ode_kwargs
        # )

        errors = []
        Js = np.zeros((2, 2, len(sol.t)))
        Js_Phi = np.zeros_like(Js)

        for idx, t in enumerate(sol.t):
            x_ivp = sol.y[:, idx]
            Phi_t = system.fundamental_matrix(t=t, t_0=t_span[0])
            errors.append(np.linalg.norm(x_ivp - Phi_t[:, idx_col]))

            Js_Phi[:, :, idx] = system.closed_form_derivative(variable="x", t=t, x=None)
            Js[:, :, idx] = ode.J(t)

            # assert (
            #     np.max(np.abs(system.closed_form_derivative("x", t) - ode.J(t))) < 1e-8
            # )

        fig, ax = plt.subplots(2, 2)
        for row in range(2):
            for col in range(2):
                ax[row, col].plot(sol.t, Js_Phi[row, col, :], label="using inverse")
                ax[row, col].plot(sol.t, Js[row, col, :], label="hard coded")
                ax[row, col].set_title(f"J_{row+1}{col+1}")
        plt.legend()

    plt.figure()
    plt.plot(sol.t, errors)


def plot_hill_matrix(system, N_HBM):
    fourier = Fourier(N_HBM=N_HBM, L_DFT=1024, n_dof=2, real_formulation=False)
    hbm = HBMEquation(
        system,
        omega=1,
        fourier=fourier,
        initial_guess=np.zeros((2 * N_HBM + 1) * system.n_dof),
        stability_method=ClassicalHill(fourier, "imaginary"),
    )
    solver = NewtonSolver(verbose=True)
    solver.solve_equation(hbm, "X")

    eigenvalues, eigenvectors = np.linalg.eig(hbm.hill_matrix())

    pass

    plt.figure()


if __name__ == "__main__":
    system = SymmetrySystem(k=7, gamma=1)
    ode = SymmetryAsODE(system.t, system.x, gamma=system.gamma, k=system.k)
    test_solutions(system, (0, 2 * np.pi), ode=ode)
    plot_hill_matrix(system, N_HBM=10)

    plt.show()
