from collections.abc import Iterable
from copy import copy

from skhippr.equations.AbstractEquation import AbstractEquation


class EquationSystem:
    def __init__(
        self,
        equations: Iterable[AbstractEquation],
        unknowns: Iterable[str],
        equation_determining_stability: AbstractEquation = None,
    ):
        self.equations = equations
        self.unknowns = unknowns
        self.init_unknowns()
        self.equation_determining_stability = equation_determining_stability

        # Initial residual evaluation
        res = self.residual_function(update=True)
        if np.max(np.abs(res)) == 0:
            self.solved = True
        else:
            self.solved = False

    def init_unknowns(self):
        """Make sure that every Equation object (and self) has every unknown as attribute"""
        self.length_unknowns = {}
        for unk in self.unknowns:
            try:
                equ_with_value = next(
                    equ for equ in self.equations if hasattr(equ, unk)
                )
            except StopIteration:
                raise ValueError(
                    f"Variable {unk} is not an attribute for any of the equations!"
                )
            value = np.atleast_1d(getattr(equ_with_value, unk))

            if value.ndim > 1:
                raise ValueError(
                    f"Attribute{equ_with_value}.{unk} has shape {value.shape}, not 1-D: not valid as unknown!"
                )

            # Check integrity of system of equations
            for equ_other in self.equations:
                if hasattr(equ_other, unk):
                    other_value = np.atleast_1d(getattr(equ_other, unk))
                    if not np.array_equal(value, other_value):
                        raise ValueError(
                            f"Error during Newton solver initialization: "
                            f"Equations {equ_with_value} and {equ_other} have the same parameter '{unk}' "
                            f"with conflicting initial values: {value} vs. {other_value}"
                        )

            self.length_unknowns[unk] = value.size
            # custom setter also sets the attribute in all equations
            setattr(self, unk, value)
        self.length_unknowns["total"] = sum(self.length_unknowns.values())

    @property
    def well_posed(self):
        return (
            self.residual_function(update=False).size == self.length_unknowns["total"]
        )

    @property
    def vector_of_unknowns(self):
        """
        Stack all individual unknowns into a single 1-D numpy array.
        For first evaluation:
            If multiple elements of self.equation have a parameter, the first one is taken.

        Returns
        -------
        numpy.ndarray
            A 1-D array containing all unknowns concatenated along axis 0.
        """
        return np.concatenate(
            [np.atleast_1d(getattr(self.equations[0], unk)) for unk in self.unknowns]
        )

    @vector_of_unknowns.setter
    def vector_of_unknowns(self, x: np.ndarray) -> None:
        """
        Separates a 1-D array into individual unknowns components and updates their values for every Equation.
        This method takes a 1-D NumPy array `x`, splits it into segments corresponding to the sizes of the unknown variables, and updates these variables accordingly.

        Parameters
        ----------

        x : np.ndarray
            A 1-D NumPy array containing the concatenated values of all unknown variables.

        Raises
        ------

        ValueError
            If `x` is not a 1-D array.
        """

        unknowns_parsed = self.parse_vector_of_unknowns(x)
        for unk, value in unknowns_parsed.items():
            # custom setter also sets the attribute in all equations
            setattr(self, unk, value)

    @property
    def stable(self):
        return self.equation_determining_stability.stable

    @property
    def eigenvalues(self):
        return self.equation_determining_stability.eigenvalues

    def parse_vector_of_unknowns(self, x=None):
        if x is None:
            x = self.vector_of_unknowns

        if x.ndim != 1:
            raise ValueError(
                f"unknowns must be a 1-D array but array is {len(x.shape)}-D"
            )

        idx = 0
        unknowns_parsed = {}
        for unk in self.unknowns:
            n = self.length_unknowns[unk]
            value = x[idx : idx + n]
            # custom setter also sets the attribute in all equations
            unknowns_parsed[unk] = value
            idx += n
        return unknowns_parsed

    def __getattr__(self, name):
        """Custom attribute extension that searches also in the unknowns"""
        if (
            "unknowns" in self.__dict__
            and "equations" in self.__dict__
            and name in self.unknowns
        ):
            return getattr(self.equations[0], name)
        else:
            raise AttributeError(
                f"'{str(self.__class__)}' object has no attribute '{name}'"
            )

    def __setattr__(self, name, value) -> None:
        """Custom attribute setter.
        If anything is changed, self.solved becomes False.
        If the attribute is part of the unknowns, it is set also in all equations."""

        if (
            "unknowns" in self.__dict__
            and "equations" in self.__dict__
            and name in self.unknowns
        ):
            for equ in self.equations:
                setattr(equ, name, value)

        if name != "solved":
            self.solved = False

        super().__setattr__(name, value)

    def residual_function(
        self, update=False
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Assembles one overall residual from the residuals of self.equations. Raises an error if there are not as many equations as unknowns.

        Returns
        -------

        np.ndarray
            The result of calling the individual residuals.
        """

        res = np.concatenate([equ.residual(update=update) for equ in self.equations])
        return res

    def jacobian(self, update=False, h_fd=1e-4):
        """Assemble the derivative of the residual w.r.t the unknowns.
        NOTE: If update is True, the equations are updated in order. I.e., if one equation relies on Jacobians of the previous equation, that should work.

        """
        jac = np.vstack(
            [
                np.hstack([equ.derivative(unk, update, h_fd) for unk in self.unknowns])
                for equ in self.equations
            ]
        )

        if jac.shape != (self.length_unknowns["total"], self.length_unknowns["total"]):
            raise RuntimeError(f"Size mismatch in Jacobian, got {jac.shape}")

        return jac

    def determine_stability(self, update=False):
        if (
            self.equation_determining_stability is None
            or self.equation_determining_stability.stability_method is None
        ):
            return None, None
        elif not self.solved:
            raise RuntimeError("Equation not solved, stability not well-defined!")
        else:
            return self.equation_determining_stability.determine_stability(
                update=update
            )

    def duplicate(self):
        equations = [copy(equ) for equ in self.equations]
        if self.equation_determining_stability is None:
            equation_determining_stability = None
        else:
            idx_stab = self.equations.index(self.equation_determining_stability)
            equation_determining_stability = equations[idx_stab]

        other = copy(self)
        other.equations = equations
        other.equation_determining_stability = equation_determining_stability
        return other
