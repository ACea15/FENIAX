from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import pathlib

from fem4inas.preprocessor.utils import dfield, initialise_Dclass

@dataclass(order=True, frozen=True)
class Dfiles:

    folder_in: str|pathlib.Path
    folder_out: str|pathlib.Path
    config_file: str|pathlib.Path

@dataclass(order=True, frozen=True)
class Dxloads:

    gravity: float = dfield("gravity force [m/s]", 9.807)
    gravity_vect: jnp.array = dfield("gravity vector", jnp.array([0, 0, -1]))
    gravity_forces: bool = dfield("Include gravity in the analysis", False)
    follower_forces: bool = dfield("Include follower forces", False)
    dead_forces: bool = dfield("Include dead forces", False)
    aero_forces: bool = dfield("Include aerodynamic forces", False)
    follower_points: list[list[int | str]] = dfield(
        "Follower force points [component, Node, coordinate]",
        None,
    )
    dead_points: list[list[int]] = dfield(
        "Dead force points \
    [component, Node, coordinate]",
        None,
    )
    follower_interpolation: list[list[int]] = dfield(
        "(Linear) interpolation of the follower forces \
    [[ti, fi]..](time_points * 2 * NumForces) [seconds, Newton]",
        None,
    )
    dead_interpolation: list[list[int]] = dfield(
        "(Linear) interpolation of the dead forces \
    [[ti, fi]] [seconds, Newton]",
        None,
    )


@dataclass(order=True, frozen=True)
class Dgeometry:

    gravity: float = dfield("gravity force [m/s]", 9.807)


@dataclass(order=True, frozen=True)
class Dfem:

    gravity: float = dfield("gravity force [m/s]", 9.807)


@dataclass(order=True, frozen=True)
class Ddriver:

    subcases: dict[str: Dxloads] = dfield("", None)
    supercases: dict[str: (list[Dfem, Dgeometry]|
                           Dfem | Dgeometry)] = dfield("", None)


@dataclass(order=True, frozen=True)
class Dsystem:

    typeof: str | int = dfield("Type of system to be solved", default='single',
                       options= ['single', 'serial', 'parallel'])
    xloads: dict | Dxloads = dfield("External loads dataclass", default=None)
    t0: float = dfield("Initial time", default=None)
    t1: float = dfield("Final time", default=None)
    tn: int = dfield("Number of time steps", default=None)
    dt: float = dfield("Delta time", default=None)    
    t: jnp.array = dfield("Time vector", default=None)
    solver_library: str = dfield("Library solving our system of equations")
    solver_name: str = dfield("Name for the solver of the previously defined library")

    def __post_init__(self):

        self.xloads = initialise_Dclass(self.xloads, Dxloads)

    def _set_label(self):
        ...
        
@dataclass(order=True, frozen=True)
class Dsimulation:

    typeof: str = dfield("Type of simulation", 'single',
                       options= ['single', 'serial', 'parallel'])
    
    systems: list[Dsystem] | Dsystem = dfield("Type of simulation", 'single',
                       options= ['single', 'serial', 'parallel'])


if __name__ == '__main__':

    d1 = Dxloads()
