import numpy as np
from typing import override
from copy import copy

from skhippr.equations.AbstractEquation import AbstractEquation
from skhippr.odes.AbstractODE import AbstractODE


class AbstractCycleEquation(AbstractEquation):
    def __init__(self, ode: AbstractODE, omega=None, period_k=1, stability_method=None):
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
        ######### DEBUGGING always use finite differences
        if False:  # variable in ("omega"):
            warnings.warn(
                "Override closed form derivative w.r.t. {variable} in AbstractCycle"
            )
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
        return super().derivative(variable, update, h_fd)

    def closed_form_derivative(self, variable):
        raise RuntimeError("This should never be called!")
        return super().closed_form_derivative(variable)

    def __copy__(self):
        # Shallow-copy everything manually without calling copy
        cls = self.__class__
        eq_other = cls.__new__(cls)
        eq_other.__dict__.update(self.__dict__)
        # Copy the ode as well
        eq_other.ode = copy(self.ode)
        return eq_other
