from dataclasses import dataclass
from typing import Sequence
import pathlib
import jax.numpy as jnp
import pathlib
import pandas as pd
from fem4inas.preprocessor.utils import dfield, initialise_Dclass
from fem4inas.preprocessor.containers.data_container import DataContainer

@dataclass(order=True, frozen=True)
class Dconst:

    I3: jnp.ndarray = dfield("3x3 Identity matrix", default=jnp.eye(3))
    e1: jnp.ndarray = dfield("3x3 Identity matrix",
                             default=jnp.array([1., 0., 0.]))
    EMAT: jnp.ndarray = dfield("3x3 Identity matrix",
                               default=jnp.array([[0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0],
                                                  [0, 0, -1, 0, 0, 0],
                                                  [0, 1, 0, 0, 0, 0]]))
    EMATT: jnp.ndarray = dfield("3x3 Identity matrix", init=False)
    
    def __post_init__(self):

        self.EMATT = self.EMATT.T
@dataclass(order=True, frozen=True)
class Dfiles:

    folder_in: str | pathlib.Path
    folder_out: str | pathlib.Path
    config_file: str | pathlib.Path


@dataclass(order=True, frozen=True)
class Dxloads:
    
    follower_points: list[list[int | str]] = dfield(
        "Follower force points [component, Node, coordinate]",
        init=False,
    )    
    dead_points: list[list[int]] = dfield("Dead force points [component, Node, coordinate]", init=False,
    )
    follower_interpolation: list[list[int]] = dfield(
        "(Linear) interpolation of the follower forces \
    [[ti, fi]..](time_points * 2 * NumForces) [seconds, Newton]",
        init=False,
    )
    dead_interpolation: list[list[int]] = dfield(
        "(Linear) interpolation of the dead forces \
    [[ti, fi]] [seconds, Newton]",
        init=False,
    )

    gravity: float = dfield("gravity force [m/s]",
                            default=9.807)
    gravity_vect: jnp.ndarray = dfield("gravity vector",
                                       default=jnp.array([0, 0, -1]))
    gravity_forces: bool = dfield("Include gravity in the analysis", default=False)
    follower_forces: bool = dfield("Include follower forces", default=False)
    dead_forces: bool = dfield("Include dead forces", default=False)
    aero_forces: bool = dfield("Include aerodynamic forces", default=False)


# @dataclass(order=True, frozen=True)
# class Dgeometry:

#     grid: str | jnp.ndarray = dfield("Grid file or array with Nodes Coordinates, node ID in the FEM and component")
#     connectivity: dict | list = dfield("Connectivities of components")
#     X: jnp.ndarray = dfield("Grid coordinates", init=False)

@dataclass(order=True, frozen=True)
class Dfem(DataContainer):

    connectivity: dict | list = dfield("Connectivities of components")
    folder: str | pathlib.Path = dfield("Folder in which to find Ka, Ma, and grid data (with those names)",
                                        default=None)
    Ka: str | pathlib.Path | jnp.ndarray = dfield("Condensed stiffness matrix", default=None)
    Ma: str | pathlib.Path | jnp.ndarray = dfield("Condensed mass matrix", default=None)
    num_modes: int = dfield("Number of modes in the solution", default=None)
    #
    grid: str | jnp.ndarray | pd.DataFrame = dfield("""Grid file or array with Nodes Coordinates,
    node ID in the FEM and component""", default=None)
    X: jnp.ndarray = dfield("Grid coordinates", default=None)
    fe_order: list | jnp.ndarray = dfield("node ID in the FEM", default=None)
    node_component: list[str | int] | jnp.ndarray = dfield("Grid coordinates", default=None)
    components: set = dfield("Name of components defining the structure", default=None)
    #
    clamped_dof: list[list] = dfield("Grid coordinates", default=None)
    clamped_nodes: int = dfield("Grid coordinates", default=None)
    num_nodes: int = dfield("Grid coordinates", default=None)

    def __post_init__(self):
        ...
    def build_grid(self):
        ...
    def __set_component_nodes(self):
        ...
    def __set_clamped_nodes(self):
        ...
    def __set_clamped_dof(self):
        ...
    def __set_clamped_indices(self):
        ...
    def __set_FEorder(self):
        self.Mfe_order = jnp.zeros((6 * self.num_nodes, 6 * self.num_nodes))
        for i in range(self.num_nodes):
            if i in self.clamped_nodes:
                fe_dof = [(6 * (self.fe_order[i] + 1) + j) for j in self.free_dof[i]]
            else:
                fe_dof = range(6 * self.fe_order[i], 6 * self.fe_order[i] + 6)
            if len(fe_dof) > 0:
                self.Mfe_order[i, fe_dof] = 1.
            
    def __set_averaging_nodes(self):
        ...
    def __set_diff_nodes(self):
        ...
    def __set_delta_nodes(self):
        ...
    def __set_Tba(self):
        ...
    def __set_load_paths(self):
        ...

@dataclass(order=True, frozen=True)
class Ddriver:

    subcases: dict[str:Dxloads] = dfield("", init=False)
    supercases: dict[str:Dfem] = dfield(
        "", init=False)

    
@dataclass(order=True, frozen=True)
class Dsystem:

    xloads: dict | Dxloads = dfield("External loads dataclass", init=False)
    t0: float = dfield("Initial time", init=False)
    t1: float = dfield("Final time", init=False)
    tn: int = dfield("Number of time steps", init=False)
    dt: float = dfield("Delta time", init=False)
    t: jnp.array = dfield("Time vector", init=False)
    solver_library: str = dfield("Library solving our system of equations", init=False)
    solver_name: str = dfield(
        "Name for the solver of the previously defined library", init=False)
    typeof: str | int = dfield("Type of system to be solved",
                               default='single',
                               options=['single', 'serial', 'parallel'])

    def __post_init__(self):

        self.xloads = initialise_Dclass(self.xloads, Dxloads)

    def _set_label(self):
        ...


@dataclass(order=True, frozen=True)
class Dsimulation:

    typeof: str = dfield("Type of simulation",
                         default='single',
                         options=['single', 'serial', 'parallel'])

    systems: list[Dsystem] | Dsystem = dfield(
        "Type of simulation",
        default='single',
        options=['single', 'serial', 'parallel'])


if (__name__ == '__main__'):

    d1 = Dxloads()
