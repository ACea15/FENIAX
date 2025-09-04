"""
Interface to nonlinear and ODE solvers
"""

import importlib
from typing import Callable

def factory(module: str, name: str) -> (Callable, Callable):
    """Factory method for the solvers

    Parameters
    ----------
    module : str
        Name of library providing solvers (diffrax, runge_kutta...)
    name : str
        Name of function to be used (ODE, newton...)

    Returns
    -------
    (Callable, Callable)
        Two functions, to build the solution object and to extract the
        states (qs) from this objects

    """

    library = importlib.import_module(f".{module}", __name__)
    function = getattr(library, name)
    states_puller = getattr(library, "pull_" + name)
    return states_puller, function
