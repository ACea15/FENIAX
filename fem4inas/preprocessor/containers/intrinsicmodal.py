from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pathlib

from fem4inas.preprocessor.utils import field2, field3

@dataclass(order=True, frozen=True)
class Dfiles:

    folder_in: str|pathlib.Path
    folder_out: str|pathlib.Path
    config_file: str|pathlib.Path

@dataclass(order=True, frozen=True)
class Dxloads:

    gravity: float = field2("gravity force [m/s]", 9.807)
    gravity_vect: jnp.array = field2("gravity vector", jnp.array([0, 0, -1]))
    gravity_forces: bool = field2("Include gravity in the analysis", False)
    follower_forces: bool = field2("Include follower forces", False)
    dead_forces: bool = field2("Include dead forces", False)
    aero_forces: bool = field2("Include aerodynamic forces", False)
    follower_points: list[list[int | str]] = field2(
        "Follower force points [component, Node, coordinate]",
        None,
    )
    dead_points: list[list[int]] = field2(
        "Dead force points \
    [component, Node, coordinate]",
        None,
    )
    follower_interpolation: list[list[int]] = field2(
        "(Linear) interpolation of the follower forces \
    [[ti, fi]..](time_points * 2 * NumForces) [seconds, Newton]",
        None,
    )
    dead_interpolation: list[list[int]] = field2(
        "(Linear) interpolation of the dead forces \
    [[ti, fi]] [seconds, Newton]",
        None,
    )


@dataclass(order=True, frozen=True)
class Dgeometry:

    gravity: float = field2("gravity force [m/s]", 9.807)


@dataclass(order=True, frozen=True)
class Dfem:

    gravity: float = field2("gravity force [m/s]", 9.807)


@dataclass(order=True, frozen=True)
class Ddriver:

    subcases: dict[str: Dxloads] = field2("", None)
    supercases: dict[str: (list[Dfem, Dgeometry]|
                           Dfem | Dgeometry)] = field2("", None)


@dataclass(order=True, frozen=True)
class Dsystem:

    typeof: str = field3("Type of system to be solved", 'single',
                       options= ['single', 'serial', 'parallel'])
    t0: float = field2("Initial time", None)
    t1: float = field2("Final time", None)
    tn: int = field2("Number of time steps", None)
    dt: float = field2("Delta time", None)    
    t: jnp.array = field2("Time vector", None)
    solver_library: str = field1("Library solving our system of equations")
    solver_name: str = field1("Name for the solver of the previously defined library")
    

@dataclass(order=True, frozen=True)
class Dsimulation:

    typeof: str = field3("Type of simulation", 'single',
                       options= ['single', 'serial', 'parallel'])
    
    systems: list[Dsystem] | Dsystem = field3("Type of simulation", 'single',
                       options= ['single', 'serial', 'parallel'])


if __name__ == '__main__':

    d1 = Dxloads()
