"""Algebraic equation system. Encodes a system of algebraic equations. Base class."""

from abc import ABC, abstractmethod
from typing import Any, override
import numpy as np

from skhippr.math import finite_difference_derivative


class AbstractEquationSystem(ABC):

    def residual(self, update=False):
        if update:
            # compute the residual using the attributes
            self.residual_value = self.residual_function()
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
        **kwargs
            Additional keyword arguments passed to the residual function.

        Returns
        -------

        np.ndarray
            The Jacobian matrix of the residual with respect to the specified variable.

        """

        # use cached derivative?
        if not update and variable in self._derivative_dict:
            return self._derivative_dict[variable]

        try:
            derivative = self.closed_form_derivative(variable)
        except NotImplementedError:
            # Fall back on finite differences.
            derivative = finite_difference_derivative(self, variable, h_step=h_fd)

        self._derivative_dict[variable] = derivative

        return derivative

    def closed_form_derivative(self, variable):
        # Can be overridden in subclasses to return
        raise NotImplementedError(
            f"Closed-form derivative of residual w.r.t {variable} not implemented."
        )


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
        """
        if x is None:
            x = self.x
        if t is None:
            t = self.t

        f = ...
        return f

    @override
    def residual_function(self):
        return self.dynamics()
