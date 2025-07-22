"""Shooting method for finding periodic solutions."""

from typing import override, Any
from collections.abc import Callable, Iterable
import numpy as np
from scipy.integrate import solve_ivp

from skhippr.problems.newton import NewtonSolver
from skhippr.stability._StabilityMethod import StabilityEquilibrium


class ShootingProblem(NewtonSolver):
    """
    ShootingProblem implements the shooting method for finding periodic solutions of ODEs.

    This class extends :py:class:`~skhippr.problems.newton.NewtonProblem` to solve the boundary value problem given by integrating the initial value problem over a period using ``scipy.integrate.solve_ivp`` and matching the state at start and end time.
    It supports both autonomous and non-autonomous systems, and can compute stability via Floquet multipliers.

    * For non-autonomous systems, the unknown ``x`` is the state at time ``t = 0``.
    * For autonomous system, the unknown ``x`` is the state at time ```t = 0`` and, as last entry, the period time ``T``. Every Newton update is performed orthogonal to the flow at ``x``.

    Parameters:
    -----------

    f : Callable[[float, np.ndarray], tuple[np.ndarray, dict[str, np.ndarray]]]
        The system of ODEs. Must return the right-hand side of the ODE and a dictionary containing the Jacobian. First input argument must be the time ``t`` and second input argument must be the state. May have additional keyword arguments.
    x0 : np.ndarray
        Initial guess for the state of the periodic solution at time ``t = 0``.
    T : float
        Period of the orbit.
    autonomous : bool, optional
        If ``True``, treats the system as autonomous and includes the period as an unknown (default: ``False``).
    variable : str, optional
        Name of the variable of the first argument of ``f`` (default: ``"x"``).
    tolerance : float, optional
        Tolerance for Newton's method convergence (default: 1e-8).
    max_iterations : int, optional
        Maximum number of Newton iterations (default: 20).
    verbose : bool, optional
        If ``True``, prints detailed output during solving (default: ``False``).
    kwargs_odesolver : dict[str, Any], optional
        Additional keyword arguments for the ODE solver, cf. ``scipy.integrate.solve_ivp()`` (default: ``None``).
    parameters : dict, optional
        Additional keyword arguments for ``f`` (default: ``None``).
    period_k : int, optional
        For non-autonomous systems: The period time of the sought-after periodic solution is ``period_k`` times the excitation period (default: 1).

    """

    def __init__(
        self,
        f: Callable[[float, np.ndarray], tuple[np.ndarray, dict[str, np.ndarray]]],
        x0: np.ndarray,
        T: float,
        autonomous: bool = False,
        variable: str = "x",
        tolerance: float = 1e-8,
        max_iterations: int = 20,
        verbose: bool = False,
        kwargs_odesolver: dict[str, Any] = None,
        parameters=None,
        period_k: int = 1,
    ):

        if kwargs_odesolver == None:
            kwargs_odesolver = dict()
        if parameters is None:
            parameters = dict()

        self.kwargs_odesolver = kwargs_odesolver
        self.n_dof = len(x0)
        self.autonomous = autonomous
        self.period_k = period_k

        if autonomous:
            x0 = np.append(x0, T)

        super().__init__(
            residual_function=f,
            initial_guess=x0,
            variable=variable,
            stability_method=StabilityEquilibrium(n_dof=self.n_dof),
            tolerance=tolerance,
            max_iterations=max_iterations,
            verbose=verbose,
            **parameters,
        )
        self.label = "shooting"
        self._T = T
        self.key_param = None

    @property
    def T(self):
        if self.autonomous:
            return self.x[-1]
        else:
            return self._T

    @T.setter
    def T(self, value):
        if self.autonomous:
            raise AttributeError("T is part of unknowns and can not be set!")
        self._T = value

    @property
    def omega(self):
        return 2 * np.pi * self.period_k / self.T

    @omega.setter
    def omega(self, value):
        if self.autonomous:
            raise AttributeError(
                "T is part of unknowns, thus omega= 2*pi/T can not be set!"
            )
        self._T = 2 * np.pi * self.period_k / value

    @override
    def residual_function(self):
        """
        Computes the residual function ``r = x(T) - x(0)`` of the shooting problem.

        Using :py:func:`~skhippr.problems.shooting.ShootingProblem.integrate_with_fundamental_matrix`, integrate over the period and return the result and the monodromy matrix. Then subtract the initial condition and the identity matrix, respectively.

        Returns
        -------
        tuple[np.ndarray, dict[str, np.ndarray]]
            the residual and the derivative dictionary
        """
        if self.autonomous:
            return self._shooting_residual_aut()
        else:
            return self._shooting_residual_nonaut(self.x)

    def determine_stability(self):
        """
        Determines the stability of the periodic solution by analyzing the Floquet multipliers.

        This method computes the monodromy matrix from the Jacobian matrix
        and determines if the system is stable based on its eigenvalues.
        It then updates the ``self.stable`` attribute to indicate system stability.

        Notes
        -----

        For autonomous systems, the Floquet multiplier
        corresponding to the freedom of phase is excluded from the stability check.
        """

        # Jacobian is Phi_T - eye --> Floquet multipliers are eigenvalues of jacobian + 1
        super().determine_stability()
        self.eigenvalues += 1
        if self.autonomous:
            idx_freedom_of_phase = np.argmin(abs(self.eigenvalues - 1))
            floquet_multipliers = np.delete(self.eigenvalues, idx_freedom_of_phase)
        else:
            floquet_multipliers = self.eigenvalues
        self.stable = all(np.abs(floquet_multipliers) <= 1)

    def x_time(self, t_eval=None):
        """
        Computes the state trajectory over time by integrating the system dynamics using ``scipy.integrate.solve_ivp``.

        Parameters
        ----------
        t_eval : array_like or None, optional
            Time points at which to store the computed solution. If None, a default
            linspace from 0 to ``self.T`` with 150 points is used.

        Returns
        -------
        np.ndarray
            (``n_dof``, ``L``) 2-D array containing the state trajectory evaluated at the specified time points.
        """

        if t_eval is None:
            t_eval = np.linspace(0, self.T, 150)
        sol = solve_ivp(
            lambda t, x: self.residual(t, x)[0],
            (0, self.T),
            self.x[: self.n_dof],
            t_eval=t_eval,
            **self.kwargs_odesolver,
        )
        return sol.y

    def _shooting_residual_nonaut(
        self, x0, stepsize=1e-4
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:

        # _, x, Phi, dx_dmu = self.integrate_with_fundamental_matrix(x0, t=self.T)
        _, x, Phi = self.integrate_with_fundamental_matrix(x0, t=self.T)
        r = x[:, -1].ravel() - x0
        dr_dx0 = Phi[:, :, -1] - np.eye(len(x0))

        derivatives = dict()
        derivatives[self.variable] = dr_dx0

        f_T, _ = self.residual(self.T, x[:, -1])
        derivatives["T"] = f_T.squeeze()

        if self.key_param:
            # Approximate the derivative using finite differences
            param = getattr(self, self.key_param) + stepsize

            # Special case T
            if self.key_param == "T":
                T = param
                key_param = "omega"
                param = 2 * np.pi / param
            else:
                key_param = self.key_param
                T = self.T

            x_pert = solve_ivp(
                lambda t, x: self.residual(t, x, **{key_param: param})[0],
                (0, T),
                x0,
                **self.kwargs_odesolver,
            ).y
            derivatives[self.key_param] = (
                x_pert[:, -1].ravel() - x[:, -1].ravel()
            ) / stepsize

        # if self.key_param:
        #     if self.key_param == "T":
        #         derivatives["T"] += dx_dmu
        #     else:
        #         derivatives[self.key_param] = dx_dmu
        return r, derivatives

    def _shooting_residual_aut(self):
        x0 = self.x[:-1]
        r, derivatives = self._shooting_residual_nonaut(x0)
        # Anchor equation: Updates should be orthogonal to f(0, x0)
        r = np.append(r, 0)
        dr_dx = np.vstack(
            (
                np.hstack(
                    (derivatives[self.variable], derivatives["T"][:, np.newaxis])
                ),
                np.hstack((self.residual(0, x0)[0][np.newaxis, :], [[0]])),
            )
        )

        for key in derivatives:
            if key == self.variable:
                derivatives[key] = dr_dx
            else:
                derivatives[key] = np.append(derivatives[key], 0)
        return r, derivatives

    def integrate_with_fundamental_matrix(
        self, x0: np.ndarray, t: float, **kwargs
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Integrates the system dynamics along with its fundamental matrix.

        This method solves the initial value problem for the system's state and its associated fundamental matrix over a given time interval.

        Parameters
        ----------

        x0 : np.ndarray
            Initial state vector of the system.
        t : float
            Final time up to which the integration is performed.
        **kwargs
            Additional keyword arguments to pass to the dynamics.

        Returns
        -------

        sol.t : np.ndarray
            Array of time points at which the solution was evaluated.

        x : np.ndarray
            Array containing the state trajectory of the system at the time points. Shape is (``n_dof``, ``L``).

        fundamental_matrix : np.ndarray
            Array containing the fundamental matrix at each time point. Shape is (``n_dof``, ``n_dof``, ``L``).

        Notes
        -----
        The method integrates both the state and the fundamental matrix by augmenting the state vector with the flattened fundamental matrix and integrating both simultaneously.
        """
        z0 = np.hstack((x0, np.eye(len(x0)).flatten(order="F")))
        # if self.key_param:
        #     z0 = np.hstack((z0, np.zeros(len(x0))))
        sol = solve_ivp(
            lambda t, z: self._dynamics_x_phi_T_param(t, z, **kwargs),
            (0, t),
            z0,
            **self.kwargs_odesolver,
        )
        x = sol.y[: len(x0), :]
        fundamental_matrix = np.reshape(
            sol.y[len(x0) : (len(x0) + 1) * len(x0), :],
            (len(x0), len(x0), -1),
            order="F",
        )
        # if self.key_param:
        #     dx_dmu = (
        #         fundamental_matrix[:, :, -1].squeeze()
        #         @ sol.y[(len(x0) + 1) * len(x0) :, -1]
        #     )
        # else:
        #     dx_dmu = None

        return sol.t, x, fundamental_matrix  # , dx_dmu

    def _dynamics_x_phi_T_param(self, t: float, z: np.ndarray, **kwargs) -> np.ndarray:
        x = z[: self.n_dof]
        Phi = np.reshape(
            z[self.n_dof : (self.n_dof + 1) * self.n_dof],
            shape=(self.n_dof, self.n_dof),
            order="F",
        )

        f, derivatives = self.residual(t, x, **kwargs)
        dx_dt = f
        df_dx = derivatives[self.variable]
        dPhi_dt = df_dx @ Phi

        dz_dt = np.hstack((dx_dt, dPhi_dt.ravel(order="F")))

        # if self.key_param:
        #     # dxdmu = z[(self.n_dof + 1) * self.n_dof :]
        #     dxdmu_dt = np.linalg.solve(Phi, derivatives[self.key_param])
        #     dz_dt = np.hstack((dz_dt, dxdmu_dt.squeeze()))

        return dz_dt


def finite_difference_fundamental_matrix(
    f: Callable[[float, np.ndarray], tuple[np.ndarray, dict[str, np.ndarray]]],
    x0: np.ndarray,
    t: float,
    h: float = 1e-3,
    **kwargs_odesolver,
) -> np.ndarray:
    "Approximate the monodromy matrix using finite difference for comparison purposes."

    sol = solve_ivp(f, t_span=(0, t), y0=x0, **kwargs_odesolver)
    x_t = sol.y[:, -1]
    Phi_t = np.zeros(shape=(len(x0), len(x0)))
    for k in range(len(x0)):
        y0 = np.zeros_like(x0)
        y0[k] = h

        sol = solve_ivp(f, t_span=(0, t), y0=x0 + y0, **kwargs_odesolver)
        Phi_t[:, k] = (sol.y[:, -1] - x_t) / h
    return Phi_t
