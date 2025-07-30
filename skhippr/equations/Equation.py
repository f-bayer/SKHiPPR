from collections.abc import Callable
from typing import override

from skhippr.equations.AbstractEquation import AbstractEquation


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
