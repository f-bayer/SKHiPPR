from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skhippr.solvers.newton import NewtonSolver


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
    def determine_eigenvalues(self, problem: "NewtonSolver") -> np.ndarray:
        """
        Determine the eigenvalues that govern the stability from a given :py:class:`~skhippr.cycles.newton.NewtonProblem`.

        Parameters
        ----------

        problem : :py:class:`~skhippr.cycles.newton.NewtonProblem`
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
        Determine the stability based on a set of eigenvalues as returned by
        :py:func:`~skhippr.stability._StabilityMethod._Stabilitymethod.self.determine_eigenvalues`.

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
    Class for assessing stability of equilibria.

    The solution of a :py:class:`~skhippr.cycles.newton.NewtonProblem` corresponds to an equilibrium of the system ::

        x_dot = problem.residual_function(x)

    The stability of this system is asserted by considering the real part of the eigenvalues of the Jacobian matrix.

    Caution
    -------

    Carefully check the sign of the residual function: While the residual multiplied by, say, -1 still yields the same equilibrium,
    the signs of the eigenvalues would flip and thus stability would be asserted wrongly.

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

    def determine_eigenvalues(self, equation) -> np.ndarray:
        """Stability is governed immediately by the eigenvalues of the Jacobian matrix, which are returned by this function."""
        return np.linalg.eigvals(equation.derivative("x", update=False))

    def determine_stability(self, eigenvalues) -> bool:
        """returns ``True`` if all eigenvalues have negative real part (smaller than ``self.tol``)."""
        return np.all(np.real(eigenvalues) < self.tol)  # type:ignore
