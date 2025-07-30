"""Shooting method for finding periodic solutions."""

from typing import override, Any
from collections.abc import Callable
import numpy as np
from scipy.integrate import solve_ivp

from skhippr.equations.AbstractEquation import AbstractEquation
from skhippr.cycles.AbstractCycleEquation import AbstractCycleEquation
from skhippr.stability._StabilityMethod import StabilityEquilibrium
from skhippr.equations.EquationSystem import EquationSystem


class ShootingBVP(AbstractCycleEquation):
    """
    ShootingBVP implements the boundary value problem with shooting method for finding periodic solutions of nonautonomous ODEs.

    This class extends :py:class:`~skhippr.cycles.newton.NewtonProblem` to solve the boundary value problem given by integrating the initial value problem over a period using ``scipy.integrate.solve_ivp`` and matching the state at start and end time.
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
        ode: Callable[[float, np.ndarray], tuple[np.ndarray, dict[str, np.ndarray]]],
        T: float,
        period_k: int = 1,
        **kwargs_odesolver,
    ):
        super().__init__(
            ode=ode,
            omega=2 * np.pi / T,
            period_k=period_k,
            stability_method=StabilityEquilibrium(ode.n_dof),
        )
        self.t_0 = ode.t
        self.kwargs_odesolver = kwargs_odesolver

    @override
    def residual_function(self):
        """
        Computes the residual function ``r = x(T) - x(0)`` of the shooting problem.

        Using :py:func:`~skhippr.cycles.shooting.ShootingProblem.integrate_with_fundamental_matrix`, integrate over the period and return the result and the monodromy matrix. Then subtract the initial condition and the identity matrix, respectively.

        Returns
        -------
        tuple[np.ndarray, dict[str, np.ndarray]]
            the residual and the derivative dictionary
        """
        x_T = self.x_time(t_eval=self.t_0 + self.T_solution)
        return x_T[:, -1] - self.x

    @override
    def closed_form_derivative(self, variable):
        match variable:
            case "x":
                _, _, Phi_t = self.integrate_with_fundamental_matrix(
                    x_0=self.x, t=self.t_0 + self.T_solution
                )
                return Phi_t[:, :, -1] - np.eye(Phi_t.shape[0])
            case "T":
                if self.ode.autonomous:
                    return self.ode.dynamics(
                        t=self.t_0 + np.squeeze(self.T_solution),
                        x=self.residual(update=False) + self.x,
                    )[:, np.newaxis]
                else:
                    raise NotImplementedError(
                        "Period is parameter of ode; default to FD"
                    )
            case _:
                raise NotImplementedError(
                    "Finite differences more efficient for shooting problem"
                )

    @override
    def stability_criterion(self, eigenvalues):
        # eigenvalues are eigenvalues of Jacobian Phi_T - I.
        # Floquet multipliers are eigs of Phi_T (i.e., eigenvalues + 1)
        # Stable if all Floquet multipliers lie inside unit circle.

        floquet_multipliers = eigenvalues + 1
        if self.ode.autonomous:
            idx_freedom_of_phase = np.argmin(abs(floquet_multipliers - 1))
            floquet_multipliers = np.delete(floquet_multipliers, idx_freedom_of_phase)

        return np.all(np.abs(floquet_multipliers) < 1 + self.stability_method.tol)

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
            t_eval = np.linspace(self.t_0, self.t_0 + self.T_solution, 150)

        if np.squeeze(t_eval).size == 1:
            t_eval = np.insert(np.squeeze(t_eval), 0, 0)

        sol = solve_ivp(
            fun=self.ode.dynamics,
            t_span=np.array((0, np.squeeze(self.T_solution))),
            y0=self.x,
            t_eval=t_eval,
            **self.kwargs_odesolver,
        )
        return sol.y

    def integrate_with_fundamental_matrix(
        self, x_0: np.ndarray = None, t: float = None
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

        if x_0 is None:
            x_0 = self.x

        if t is None:
            t = self.t_0 + self.T_solution

        t_span = np.insert(np.squeeze(t), 0, self.t_0)

        z_0 = np.hstack((x_0, np.eye(len(x_0)).flatten(order="F")))
        sol = solve_ivp(
            self._dynamics_x_phi_T,
            t_span,
            z_0,
            **self.kwargs_odesolver,
        )
        x = sol.y[: len(x_0), :]
        fundamental_matrix = np.reshape(
            sol.y[len(x_0) : (len(x_0) + 1) * len(x_0), :],
            (len(x_0), len(x_0), -1),
            order="F",
        )

        return sol.t, x, fundamental_matrix

    def _dynamics_x_phi_T(self, t: float, z: np.ndarray) -> np.ndarray:
        x = z[: self.ode.n_dof]
        Phi = np.reshape(
            z[self.ode.n_dof : (self.ode.n_dof + 1) * self.ode.n_dof],
            shape=(self.ode.n_dof, self.ode.n_dof),
            order="F",
        )

        dx_dt = self.ode.dynamics(t=t, x=x)
        df_dx = self.ode.derivative(variable="x", t=t, x=x)
        dPhi_dt = df_dx @ Phi

        dz_dt = np.hstack((dx_dt, dPhi_dt.ravel(order="F")))

        return dz_dt


class ShootingPhaseAnchor(AbstractEquation):
    def __init__(self, ode, x):
        super().__init__(None)
        self.ode = ode
        self.x = x

    def residual_function(self):
        return np.atleast_1d(0)

    def closed_form_derivative(self, variable):
        if variable == "x":
            return self.ode.dynamics(x=self.x)[np.newaxis, :]
        else:
            return np.atleast_2d(0)


class ShootingSystem(EquationSystem):
    def __init__(
        self,
        ode: Callable[[float, np.ndarray], tuple[np.ndarray, dict[str, np.ndarray]]],
        T: float,
        period_k: int = 1,
        **kwargs_odesolver,
    ):
        bvp = ShootingBVP(ode, T, period_k, **kwargs_odesolver)
        equations = [bvp]
        unknowns = ["x"]

        if ode.autonomous:
            equations.append(ShootingPhaseAnchor(ode, bvp.x))
            unknowns.append("T")

        super().__init__(
            equations=equations, unknowns=unknowns, equation_determining_stability=bvp
        )
