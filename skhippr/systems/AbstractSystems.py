"""Algebraic equation system. Encodes a system of algebraic equations. Base class."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, override
import numpy as np


class AbstractEquationSystem(ABC):

    def __init__(self):
        super().__init__()
        self._derivative_dict = {}
        self.residual_value = None

    def residual(self, update=False):
        if update:
            # compute the residual using the attributes
            self.residual_value = self.residual_function()
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


class EquationSystem(AbstractEquationSystem):
    def __init__(
        self,
        residual_function,
        closed_form_derivative: dict[str, Callable],
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
class FirstOrderODE(AbstractEquationSystem):
    def __init__(self, autonomous: bool, n_dof: int):
        """The constructor must set the number of degrees of freedom as well as all required parameter values (including the initial state) as properties."""
        super().__init__()
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
        if t is not None and not np.isscalar(t) and x.shape[1] != len(t):
            raise ValueError(
                f"t and x have incompatible sizes: {t.shape} vs. {x.shape}"
            )

    @override
    def residual_function(self):
        return self.dynamics()

    def closed_form_derivative(self, variable, t=None, x=None):
        # Provide an interface which offers t and x
        return super().closed_form_derivative(variable)
