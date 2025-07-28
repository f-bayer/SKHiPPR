"""Algebraic equation system. Encodes a system of algebraic equations. Base class."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import override
import numpy as np
from copy import copy
import warnings

from skhippr.stability._StabilityMethod import StabilityEquilibrium


class AbstractEquation(ABC):

    def __init__(self, stability_method=None):
        super().__init__()
        self._derivative_dict = {}
        self.residual_value = None
        self.stability_method = stability_method
        self.stable = None
        self.eigenvalues = None

    def residual(self, update=False):
        if update:
            # compute the residual using the attributes
            self.residual_value = self.residual_function()
            if self.residual_value.ndim > 1:
                raise ValueError(
                    f"Residual must be a 1-D numpy array but has shape {self.residual_value.shape}"
                )
        elif self.residual_value is None:
            raise RuntimeError("Residual has not been computed yet!")
        return self.residual_value

    @abstractmethod
    def residual_function(self):  # -> ndarray:
        """Compute the residual function based on attributes."""
        residual: np.ndarray = ...
        return residual

    def visualize(self):
        pass

    def derivative(self, variable: str, update=False, h_fd=1e-4):
        """
        Compute the derivative of the residual with respect to a given variable.
        The derivative is computed using finite differences.
        The variable is assumed to be a property of the AbstractEquationSystem.
        This method can be overwritten in subclasses to return a closed-form derivative.

        Parameters
        ----------

        variable : str
            The name of the variable (property of the system) with respect to which the derivative is computed.
        update : bool, optional
            If True, updates the cached derivative with the newly computed value. Default is False.
        h_fd : float, optional
            Step size for finite difference approximation. Default is 1e-4.

        Returns
        -------

        np.ndarray
            The partial derivative of the residual with respect to the specified variable.

        """

        # use cached derivative?
        if not update and variable in self._derivative_dict:
            return self._derivative_dict[variable]

        ########### DEBUGGING always use finite differences
        # if True:  # variable in ("X", "omega"):
        #     derivative = self.finite_difference_derivative(variable, h_step=h_fd)
        #     # Check sizes
        #     cols_expected = np.atleast_1d(getattr(self, variable)).shape[0]
        #     rows_expected = self.residual(update=False).shape[0]
        #     others_expected = self.residual(update=False).shape[1:]
        #     if derivative.shape != (rows_expected, cols_expected, *others_expected):
        #         raise ValueError(
        #             f"Size mismatch in derivative w.r.t. '{variable}': Expected {(rows_expected, cols_expected, *others_expected)}, got {derivative.shape[:2]}"
        #         )

        #     self._derivative_dict[variable] = derivative
        #     print(
        #         f"Caution overrode '{variable}' closed form derivative in AbstractSystems.py for debugging reasons"
        #       )
        #     warnings.warn(
        #         f"Caution overrode '{variable}' closed form derivative in AbstractSystems.py for debugging reasons"
        #       )

        #     return derivative
        ###########

        try:
            derivative = self.closed_form_derivative(variable)
        except NotImplementedError:
            # Fall back on finite differences.
            derivative = self.finite_difference_derivative(variable, h_step=h_fd)

        # Check sizes
        cols_expected = np.atleast_1d(getattr(self, variable)).shape[0]
        rows_expected = self.residual(update=False).shape[0]
        others_expected = self.residual(update=False).shape[1:]
        if derivative.shape != (rows_expected, cols_expected, *others_expected):
            raise ValueError(
                f"Size mismatch in derivative w.r.t. '{variable}': Expected {(rows_expected, cols_expected, *others_expected)}, got {derivative.shape[:2]}"
            )

        self._derivative_dict[variable] = derivative

        return derivative

    def closed_form_derivative(self, variable):
        # Can be overridden in subclasses to return
        raise NotImplementedError(
            f"Closed-form derivative of residual w.r.t {variable} not implemented."
        )

    def finite_difference_derivative(self, variable, h_step=1e-4) -> np.ndarray:

        x_orig = getattr(self, variable)
        x = np.atleast_1d(x_orig)
        if x.ndim == 1:
            x = x[:, np.newaxis]
        n = x.shape[0]
        f = self.residual(update=True)
        delta = h_step * np.eye(n)
        derivative = np.zeros((f.shape[0], n, *f.shape[1:]), dtype=f.dtype)

        for k in range(n):
            setattr(self, variable, np.squeeze(x + delta[:, [k]]))
            derivative[:, k, ...] = (self.residual_function() - f) / h_step

        setattr(self, variable, x_orig)
        return derivative

    def determine_stability(self, update=False):
        if self.stability_method is None:
            raise AttributeError("Stability method not available")

        if update:
            eigenvalues = self.stability_method.determine_eigenvalues(self)
            self.eigenvalues = eigenvalues
            stable = self.stability_criterion(eigenvalues)
            self.stable = stable

        return self.stable, self.eigenvalues

    def stability_criterion(self, eigenvalues):
        if self.stability_method is None:
            raise ValueError("No stability method available!")
        else:
            raise NotImplementedError(
                "To be implemented in concrete subclasses if needed"
            )


class Equation(AbstractEquation):
    def __init__(
        self,
        residual_function,
        closed_form_derivative: Callable,
        **parameters,
    ):
        super().__init__()
        # Overwrite the residual function and derivatives by passed functions
        self._residual_function = residual_function
        self._closed_form_derivative = closed_form_derivative
        self._list_params = list(parameters.keys())
        self.__dict__.update(parameters)

    @override
    def residual_function(self):
        return self._residual_function(
            **{key: self.__dict__[key] for key in self._list_params}
        )

    @override
    def closed_form_derivative(self, variable):
        return self._closed_form_derivative(
            variable, **{key: self.__dict__[key] for key in self._list_params}
        )


# still an abstract class
class FirstOrderODE(AbstractEquation):
    def __init__(self, autonomous: bool, n_dof: int, stability_method=None):
        """The constructor must set the number of degrees of freedom as well as all required parameter values (including the initial state) as properties."""
        if stability_method is None:
            stability_method = StabilityEquilibrium(n_dof)
        super().__init__(stability_method=stability_method)
        self.autonomous = autonomous
        self.n_dof = n_dof

    @abstractmethod
    def dynamics(self, t=None, x=None) -> np.ndarray:
        """Return the right-hand side of the first-order ode x_dot = f(t, x) as a numpy array of size self.n_dof.
        All parameters are to be obtained as attributes.
        Subclass implementations are expected to check the correct dimensions of x and t.
        """
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        self.check_dimensions(t, x)
        f = ...
        return f

    def check_dimensions(self, t=None, x=None):

        if x is not None and x.shape[0] != self.n_dof:
            raise ValueError(f"{x} must have {self.n_dof} rows but has {x.shape[0]}")
        if t is not None and np.isscalar(t) and len(x.shape) > 1 and x.shape[1] > 1:
            raise ValueError(
                f"{x} must be 1-D or have exactly one column if t is scalar"
            )
        if t is not None and np.squeeze(t).size > 1 and x.shape[1] != len(t):
            raise ValueError(
                f"t and x have incompatible sizes: {t.shape} vs. {x.shape}"
            )

    @override
    def residual_function(self):
        return self.dynamics()

    @override
    def stability_criterion(self, eigenvalues):
        return np.all(np.real(eigenvalues) < self.stability_method.tol)

    @override
    def derivative(self, variable, update=False, h_fd=1e-4, t=None, x=None):
        if t is not None or x is not None:
            update = True

        ########## DEBUGGING always use finite differences
        # if True:  # variable in ("X", "omega"):
        #     warnings.warn("Override closed form derivative in FirstOrderODE")
        #     print(f"Override closed form derivative w.rt. {variable} in FirstOrderODE")
        #     if t is not None:
        #         t_old = self.t
        #         self.t = t

        #     if x is not None:
        #         x_old = self.x
        #         self.x = x

        #     derivative = self.finite_difference_derivative(variable, h_step=h_fd)

        #     if t is not None:
        #         self.t = t_old

        #     if x is not None:
        #         self.x = x_old

        #     # Check sizes
        #     cols_expected = np.atleast_1d(getattr(self, variable)).shape[0]
        #     rows_expected = self.residual(update=False).shape[0]
        #     others_expected = self.residual(update=False).shape[1:]
        #     if derivative.shape != (rows_expected, cols_expected, *others_expected):
        #         raise ValueError(
        #             f"Size mismatch in derivative w.r.t. '{variable}': Expected {(rows_expected, cols_expected, *others_expected)}, got {derivative.shape[:2]}"
        #         )

        #     self._derivative_dict[variable] = derivative

        #     return derivative
        ##########

        if not update:
            return super().derivative(variable, update, h_fd)
        # provide an interface which offers t and x
        try:
            derivative = self.closed_form_derivative(variable, t, x)
        except NotImplementedError:
            t_old = self.t
            x_old = self.x

            self.t = t
            self.x = x

            derivative = super().derivative(variable, update, h_fd)
            self.t = t_old
            self.x = x_old

        self._derivative_dict[variable] = derivative
        return derivative

    def closed_form_derivative(self, variable, t=None, x=None):
        # Provide an interface which offers t and x
        return super().closed_form_derivative(variable)


class AbstractCycleEquation(AbstractEquation):
    def __init__(
        self, ode: FirstOrderODE, omega=None, period_k=1, stability_method=None
    ):
        self.ode = ode
        super().__init__(stability_method)

        if omega is not None:
            ode.omega = omega

        self.period_k = period_k

    @property
    def T(self):
        return 2 * np.pi / self.omega

    @T.setter
    def T(self, value):
        self.omega = 2 * np.pi / value

    @property
    def T_solution(self):
        return self.T * self.period_k

    @property
    def omega_solution(self):
        "Returns the angular frequency of the periodic solution (factor_k times the excitation frequency)."
        return self.omega / self.period_k

    @property
    def T(self):
        return 2 * np.pi / self.omega

    @T.setter
    def T(self, value):
        self.omega = 2 * np.pi / value

    def __getattr__(self, name):
        """Custom attribute extension that searches for attributes also in the ode"""
        if "ode" in self.__dict__:
            return getattr(self.ode, name)
        else:
            raise AttributeError(
                f"'{str(self.__class__)}' object has no attribute '{name}'"
            )

    def __setattr__(self, name, value) -> None:
        """Custom attribute setter.
        Attempts first to change the attribute in self.ode, if applicable.
        """

        if "ode" in self.__dict__ and hasattr(self.ode, name):
            setattr(self.ode, name, value)

        else:
            super().__setattr__(name, value)

    @override
    def derivative(self, variable, update=False, h_fd=0.0001):
        # return super().derivative(variable, update, h_fd)
        ######### DEBUGGING always use finite differences
        if True:  # variable in ("X", "omega"):
            warnings.warn("Override closed form derivative in AbstractCycle")
            print(f"Override closed form derivative w.r.t. {variable} in AbstractCycle")

            derivative = self.finite_difference_derivative(variable, h_step=h_fd)

            # Check sizes
            cols_expected = np.atleast_1d(getattr(self, variable)).shape[0]
            rows_expected = self.residual(update=False).shape[0]
            others_expected = self.residual(update=False).shape[1:]
            if derivative.shape != (rows_expected, cols_expected, *others_expected):
                raise ValueError(
                    f"Size mismatch in derivative w.r.t. '{variable}': Expected {(rows_expected, cols_expected, *others_expected)}, got {derivative.shape[:2]}"
                )

            self._derivative_dict[variable] = derivative

            return derivative
        ##########

    def __copy__(self):
        # Shallow-copy everything manually without calling copy
        cls = self.__class__
        eq_other = cls.__new__(cls)
        eq_other.__dict__.update(self.__dict__)
        # Copy the ode as well
        eq_other.ode = copy(self.ode)
        return eq_other
