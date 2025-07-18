"""Algebraic equation system. Encodes a system of algebraic equations. Base class."""

from abc import ABC, abstractmethod
from typing import Any, override
import numpy as np

from skhippr.math import finite_differences


class FirstOrderODE(ABC):
    def __init__(self, autonomous: bool, n_dof: int, **kwargs):
        """The constructor must set the number of degrees of freedom as well as all required parameter values (including the initial state) as properties."""
        self.autonomous = autonomous
        self.n_dof = n_dof

    @abstractmethod
    def dynamics(self, **kwargs) -> np.ndarray:
        """Return the right-hand side of the first-order ode x_dot = f(t, x) as a numpy array of size self.n_dof.
        All parameters (including t and x) are to be obtained as attributes if not passed as keyword argument.
        """
        if x is None:
            x = self.x
        f = ...
        return f

    def derivative(self, variable, h_fd=1e-4, **kwargs):
        """Return the partial derivative of f w.r.t <variable> as a self-n_dof x self.n_dof numpy array. Recommended implementation using switch/case and individual methods for all required derivatives."""
        if variable not in kwargs:
            kwargs[variable] = getattr(self, variable)

        derivative = finite_differences(
            self.dynamics, kwargs=kwargs, variable=variable, h_step=h_fd
        )
        return derivative


class AbstractEquationSystem(ABC):

    @abstractmethod
    def residual(self, update=False, **kwargs):
        if not update and not kwargs:
            return self._residual

        else:
            # compute the residual using the attributes and the keyword arguments - to be overridden in the subclass!
            residual = ...

        if update:
            self._residual = residual

    @abstractmethod
    def visualize(self): ...

    def derivative(self, variable: str, update=False, h_fd=1e-4, **kwargs):
        """
        Compute the derivative (Jacobian) of the system residual with respect to a given variable.
        This method should be overwritten in subclasses to return a closed-form derivative if available.
        Otherwise, the derivative is computed using finite differences.

        Parameters
        ----------

        variable : str
            The name of the variable with respect to which the derivative is computed.
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

        # shortcut if Jacobian is already available
        if not update and variable in self._derivative_dict and not kwargs:
            return self._derivative_dict[variable]

        else:
            # Fall back on finite differences.
            # Can/should be overwritten by closed-form expressions in concrete subclasses.
            if variable not in kwargs:
                kwargs[variable] = self.getattr(self, variable)
            kwargs["update"] = False

            derivative = finite_differences(
                self.residual, kwargs, variable, h_step=h_fd
            )
            del kwargs["update"]

        if update:
            self._derivative_dict[variable] = derivative

        return derivative


class EquilibriumSystem(AbstractEquationSystem):
    def __init__(self, ode: FirstOrderODE, **ode_inputs: dict[str, Any]):
        super().__init__()
        self.ode = ode
        self.__dict__.update(ode_inputs)  # Add all ODE inputs as attributes

    @override
    def residual(self):
        return self.ode.dynamics(t=0, **self.__dict__)

    @override
    def derivative(self, variable):
        return self.ode.derivative(variable, t=0, **self.__dict__)

    @override
    def visualize(self):
        raise NotImplementedError("Visualization to be implemented")
