from collections.abc import Iterable
from copy import copy
import numpy as np
from numpy._typing._array_like import NDArray

from skhippr.equations.AbstractEquation import AbstractEquation


class EquationSystem:
    """Encodes (one or multiple) :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation` objects, together with a matching set of ``unknowns`` (names of attributes of one or more ``equations``), to be solved by the :py:class:`~skhippr.solvers.newton.NewtonSolver`.

    All unknowns are attributes of the :py:class:`~skhippr.equations.EquationSystem.EquationSystem` and of all its collected :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation` objects. If the unknown is updated in the :py:class:`~skhippr.equations.EquationSystem.EquationSystem`, it is also updated in all equations.

    One of the equations can be chosen to determine the stability of the system. This is done by passing an equation as ``equation_determining_stability`` to the constructor. If no such equation is given, no stability analysis is performed.
    """

    def __init__(
        self,
        equations: Iterable[AbstractEquation],
        unknowns: Iterable[str],
        equation_determining_stability: AbstractEquation = None,
    ):
        self.equations = equations
        self.unknowns = unknowns
        self._init_unknowns()
        self.equation_determining_stability = equation_determining_stability

        # Initial residual evaluation
        res = self.residual_function(update=True)
        if np.max(np.abs(res)) == 0:
            self.solved = True
        else:
            self.solved = False

    def _init_unknowns(self):
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
        """Checks if the system of equations is well-posed, i.e., if the total number of equations matches the total number of unknowns."""
        return (
            self.residual_function(update=False).size == self.length_unknowns["total"]
        )

    @property
    def vector_of_unknowns(self) -> np.ndarray:
        """
        All individual unknowns, stacked into a single 1-D numpy array in the order given by ``self.unknowns``. This property can also be set to update all unknowns in the system.
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
    def stable(self) -> bool:
        """Stability of the equation system, determined by the equation given in ``self.equation_determining_stability``."""
        return self.equation_determining_stability.stable

    @property
    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues governing the stability of the equation system, determined by the equation given in ``self.equation_determining_stability``."""
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

    def residual_function(self, update=False) -> np.ndarray:
        """
        Assembles one overall residual (1-D numpy array) by stacking the residuals of ``self.equations``.
        """

        res = np.concatenate([equ.residual(update=update) for equ in self.equations])
        return res

    def jacobian(self, update=False, h_fd=1e-4) -> np.ndarray:
        """Assemble the derivative of the overall residual w.r.t  all the unknowns (2-D numpy array), with the ordering given by the ordering of ``self.equations`` and ``self.unknowns``, respectively."""
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

    def duplicate(self) -> "EquationSystem":
        """Creates a copy of the EquationSystem, with all equations and the equation determining stability shallow-copied to avoid accidental changes to the original equations."""
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
