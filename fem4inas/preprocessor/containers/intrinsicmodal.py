from dataclasses import dataclass
from typing import Sequence
import pathlib
import jax.numpy as jnp
import pathlib
import pandas as pd
from fem4inas.preprocessor.utils import dfield, initialise_Dclass
from fem4inas.preprocessor.containers.data_container import DataContainer
import fem4inas.intrinsic.geometry as geometry

@dataclass(frozen=True)
class Dconst(DataContainer):

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
    EMATT: jnp.ndarray = dfield("3x3 Identity matrix", default=None)
    
    def __post_init__(self):

        self.EMATT = self.EMATT.T
@dataclass(frozen=True)
class Dfiles(DataContainer):

    folder_in: str | pathlib.Path
    folder_out: str | pathlib.Path
    config_file: str | pathlib.Path


@dataclass(frozen=False)
class Dxloads(DataContainer):
    
    follower_points: list[list[int | str]] = dfield(
        "Follower force points [component, Node, coordinate]",
        default=None,
    )    
    dead_points: list[list[int]] = dfield("Dead force points [component, Node, coordinate]", default=None,
    )
    follower_interpolation: list[list[int]] = dfield(
        "(Linear) interpolation of the follower forces \
    [[ti, fi]..](time_points * 2 * NumForces) [seconds, Newton]",
        default=None,
    )
    dead_interpolation: list[list[int]] = dfield(
        "(Linear) interpolation of the dead forces \
    [[ti, fi]] [seconds, Newton]",
        default=None,
    )

    gravity: float = dfield("gravity force [m/s]",
                            default=9.807)
    gravity_vect: jnp.ndarray = dfield("gravity vector",
                                       default=jnp.array([0, 0, -1]))
    gravity_forces: bool = dfield("Include gravity in the analysis", default=False)
    follower_forces: bool = dfield("Include follower forces", default=False)
    dead_forces: bool = dfield("Include dead forces", default=False)
    aero_forces: bool = dfield("Include aerodynamic forces", default=False)


# @dataclass(frozen=True)
# class Dgeometry:

#     grid: str | jnp.ndarray = dfield("Grid file or array with Nodes Coordinates, node ID in the FEM and component")
#     connectivity: dict | list = dfield("Connectivities of components")
#     X: jnp.ndarray = dfield("Grid coordinates", default=None)

@dataclass(frozen=True)
class Dfem(DataContainer):

    connectivity: dict | list = dfield("Connectivities of components")
    folder: str | pathlib.Path = dfield("""Folder in which to find Ka, Ma,
    and grid data (with those names)""", default=None)
    Ka: str | pathlib.Path | jnp.ndarray = dfield("Condensed stiffness matrix", default=None)
    Ma: str | pathlib.Path | jnp.ndarray = dfield("Condensed mass matrix", default=None)
    num_modes: int = dfield("Number of modes in the solution", default=None)
    #
    grid: str | jnp.ndarray | pd.DataFrame = dfield("""Grid file or array with Nodes Coordinates,
    node ID in the FEM, and associated component""", default=None)
    X: jnp.ndarray = dfield("Grid coordinates", default=None)
    fe_order: list[int] | jnp.ndarray = dfield("node ID in the FEM", default=None)
    component_vect: list[str] = dfield("Array with component associated to each node",
                                       default=None)
    component_nodes: list[str | int] | jnp.ndarray = dfield("Grid coordinates", init=False)
    component_names: list = dfield("Name of components defining the structure", init=False)
    component_chain: dict[str:list[str]] = dfield(" ", init=False)
    prevnodes: list[int] = dfield("Name of components defining the structure", init=False)
    #
    clamped_dof: list[list] = dfield("Grid coordinates", default=None)
    clamped_nodes: int = dfield("Grid coordinates", default=None)
    num_nodes: int = dfield("Number of nodes", init=False)
    Mavg: jnp.ndarray = dfield("Matrix for tensor average between nodes", init=False)
    Mdiff: jnp.ndarray = dfield("Matrix for tensor difference between nodes", init=False)
    Mfe_order: jnp.ndarray = dfield("""Matrix with 1s and 0s that reorders quantities
    such as eigenvectors in the FE model; nodes in horizontal arrangement.""", init=False)
    Mload_paths: jnp.ndarray = dfield("""Matrix with with 1s and 0s for the load paths
    that each node, in vertical arrangement, need to transverse to sum up to a free-end""",
                                      init=False)
    def __post_init__(self):
        self.connectivity = geometry.list2dict(self.connectivity)
        self.df_grid, self.X, self.fe_order, self.component_vect = geometry.build_grid(
            self.grid, self.X, self.fe_order, self.component_vect)
        self.component_names, self.component_father = geometry.compute_component_father(
            self.connectivity)
        self.num_nodes = len(self.X)
        self.clamped_nodes, self.freeDoF, self.clampedDoF, self.total_clampedDoF = \
            geometry.compute_clamped(self.fe_order)
        self.component_nodes = geometry.compute_component_nodes(self.component_vect)
        self.component_chain = geometry.compute_component_chain(self.component_names,
                                                                self.connectivity)
        self.prevnodes = geometry.compute_prevnode(self.component_vect,
                                                   self.component_nodes,
                                                   self.component_father)
        self.Mavg = geometry.compute_Maverage(self.prevnodes, self.num_nodes)
        self.Mdiff = geometry.compute_Mdiff(self.prevnodes, self.num_nodes)
        self.Mfe_order = geometry.compute_Mfe_order(self.fe_order,
                                                    self.clamped_nodes,
                                                    self.freeDoF,
                                                    self.component_nodes,
                                                    self.component_chain,
                                                    self.num_nodes)
        self.Mload_paths = geometry.compute_Mloadpaths(self.component_vect,
                                                       self.component_nodes,
                                                       self.component_chain,
                                                       self.num_nodes)

@dataclass(frozen=True)
class Ddriver(DataContainer):

    subcases: dict[str:Dxloads] = dfield("", default=None)
    supercases: dict[str:Dfem] = dfield(
        "", default=None)


@dataclass(frozen=False)
class Dsystem(DataContainer):

    name: str = dfield("System name")
    xloads: dict | Dxloads = dfield("External loads dataclass", default=None)
    t0: float = dfield("Initial time", default=0.)
    t1: float = dfield("Final time", default=1.)
    tn: int = dfield("Number of time steps", default=None)
    dt: float = dfield("Delta time", default=None)
    t: jnp.array = dfield("Time vector", default=None)
    solver_library: str = dfield("Library solving our system of equations", default=None)
    solver_name: str = dfield(
        "Name for the solver of the previously defined library", default=None)
    typeof: str | int = dfield("Type of system to be solved",
                               default='single',
                               options=['single', 'serial', 'parallel'])

    def __post_init__(self):

        self.xloads = initialise_Dclass(self.xloads, Dxloads)

    def _set_label(self):
        ...


@dataclass(frozen=True)
class Dsimulation(DataContainer):

    typeof: str = dfield("Type of simulation",
                         default='single',
                         options=['single', 'serial', 'parallel'])
    systems: dict = dfield(
        "Dictionary of systems involved in the simulation",
        default=None
    )
    workflow: dict = dfield(
        """Dictionary that defines which system is run after which.
        The default None implies systems are run in order of the input""",
        default=None
    )

    def __post_init__(self):

        if self.systems is not None:
            for k, v in self.systems:
                setattr(self, k, initialise_Dclass(v, Dsystem))

if (__name__ == '__main__'):

    d1 = Dxloads()
