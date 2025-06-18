Addendum: Shooting method
=========================

As a reference to compare/check against the HBM methods which are the focus of the toolbox, the shooting method is also implemented as a subclass of :py:class:`~skippr.problems.newton.NewtonProblem`. 

.. caution:: 
    In the shooting method, the frequency of the sought-after periodic solution is governed by the period time ``T`` and not by ``omega``. This leads to the following pitfalls:

    * For autonomous systems, the last state of the unknown has a different interpretation than in the :py:class:`~skhippr.problems.HBMProblem.HBMProblem`. 
    * For nonautonomous systems with frequency-dependent excitation, the computation of frequency response curves using :py:func:`~skhippr.problems.continuation.pseudo_arclength_continuator` starting from a :py:class:`~skhippr.problems.shooting.ShootingProblem` is prone to usage errors due to the nontrivial interaction between the continuation parameter ``T`` and the input argument ``omega`` of the system. It is recommended to compute the frequency response curve instead using the Harmonic Balance method (i.e., starting from a :py:class:`~skhippr.problems.HBM.HBMProblem`), which is immediately parameterized by ``omega``. 

.. autoclass:: skhippr.problems.shooting.ShootingProblem
    :members: