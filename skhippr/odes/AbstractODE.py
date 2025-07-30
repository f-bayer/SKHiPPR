"""Algebraic equation system. Encodes a system of algebraic equations. Base class."""

from abc import ABC, abstractmethod

from typing import override
import numpy as np
from copy import copy
import warnings

from skhippr.stability._StabilityMethod import StabilityEquilibrium
from skhippr.equations.AbstractEquation import AbstractEquation


# still an abstract class
class AbstractODE(AbstractEquation):
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
