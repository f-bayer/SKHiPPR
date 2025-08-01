from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skhippr.equations.AbstractEquation import AbstractEquation


class AbstractStabilityMethod(ABC):
    """
    Abstract base class for stability analysis methods.
    This class defines the commmon interface for stability methods in SKHiPPR.

    Attributes:
    -----------

    label : str
        A descriptive label for the stability method.
    tol : float
        Tolerance value used in stability determination.

    """

    def __init__(self, label: str, tol: float):
        super().__init__()
        self.label = label
        self.tol = tol

    @abstractmethod
    def determine_eigenvalues(self, equation: "AbstractEquation") -> np.ndarray:
        """
        Determine the eigenvalues that govern the stability from a given :py:class:`~skhippr.cycles.newton.NewtonProblem`.

        Parameters
        ----------

        equation : :py:class:`~skhippr.equations.AbstractEquation.AbstractEquation`
            The problem instance containing the (converged) system for which eigenvalues are to be computed.

        Returns
        --------

        np.ndarray
            A 1-D array of eigenvalues associated with the stability of the system.

        Notes
        -----

        Each concrete subclass of this abstract base class can have a different interpretation of the eigenvalues, e.g. Floquet multipliers.

        """
        ...

    @abstractmethod
    def determine_stability(self, eigenvalues: np.ndarray) -> bool:
        """
        Determine the stability based on a set of eigenvalues as returned by ``self.determine_eigenvalues()``.

        Parameters
        ----------

        eigenvalues : np.ndarray
            Array of eigenvalues.

        Returns
        -------

        bool
            ``True`` if the system is stable, ``False`` otherwise.

        Notes
        -----

        The specific stability criterion depends on the implementation. For instance, for equilibria,
        stability is indicated if all eigenvalues have negative real parts; for periodic solutions of nonautonomous systems, all eigenvalues (Floquet multipliers)
        should have magnitudes less than one.
        """

        ...

    def __str__(self):
        return self.label


class StabilityEquilibrium(AbstractStabilityMethod):
    """
    Class for assessing the stability of an equilibrium of a an :py:class:`~skhippr.odes.AbstractODE.AbstractODE`.

    Parameters
    ----------

    n_dof : int
        Number of degrees of freedom of the problem.

    tol : float, optional
        Tolerance for stability assessment. Default is 0.

    """

    def __init__(self, n_dof: int, tol: float = 0):
        super().__init__(label="Equilibrium stability", tol=tol)
        self.n_dof = n_dof

    def determine_eigenvalues(self, ode) -> np.ndarray:
        """Stability is governed immediately by the eigenvalues of the Jacobian matrix, which are returned by this function."""
        return np.linalg.eigvals(ode.derivative("x", update=False))

    def determine_stability(self, eigenvalues) -> bool:
        """returns ``True`` if all eigenvalues have negative real part (smaller than ``self.tol``)."""
        return np.all(np.real(eigenvalues) < self.tol)  # type:ignore
