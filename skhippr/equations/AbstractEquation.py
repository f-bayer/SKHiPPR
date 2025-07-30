from abc import ABC, abstractmethod
import numpy as np


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
        # Can be overridden in subclasses to return a numpy array
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
