from dataclasses import dataclass
from typing import Sequence
import pathlib
import jax.numpy as jnp
import pathlib
import pandas as pd
from fem4inas.preprocessor.utils import dfield, initialise_Dclass, load_jnp
from fem4inas.preprocessor.containers.data_container import DataContainer
import fem4inas.intrinsic.geometry as geometry
from fem4inas.intrinsic.functions import coordinate_transform
import jax
from enum import Enum

class Solution(Enum):
    STATIC = 0
    DYNAMIC = 1
    MULTIBODY = 2
    STABILITY = 3
    
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
    EMATT: jnp.ndarray = dfield("3x3 Identity matrix", init=False)
    
    def __post_init__(self):

        object.__setattr__(self, 'EMATT', self.EMAT.T)
        
@dataclass(frozen=True)
class Dfiles(DataContainer):

    folder_in: str | pathlib.Path
    folder_out: str | pathlib.Path
    config: str | pathlib.Path


@dataclass(frozen=False)
class Dxloads(DataContainer):
    
    follower_forces: bool = dfield("Include point follower forces",
                                   default=False)
    dead_forces: bool = dfield("Include point dead forces",
                               default=False)
    gravity_forces: bool = dfield("Include gravity in the analysis",
                                  default=False)    
    aero_forces: bool = dfield("Include aerodynamic forces",
                               default=False)
    x: jnp.array = dfield("x-axis vector for interpolation",
                          default=None)
    force_follower: jnp.ndarray = dfield("""Point follower forces
    (len(x)x6xnum_nodes)""",
                                         default=None)
    force_dead: jnp.ndarray = dfield("""Point follower forces
    (len(x)x6xnum_nodes)""",
                                     default=None)
    follower_points: list[list[int, int]] = dfield(
        "Follower force points [Node, coordinate]",
        default=None,
    )
    dead_points: list[list[int, int]] = dfield(
        "Dead force points [Node, coordinate]",
        default=None,
    )

    follower_interpolation: list[list[float]] = dfield(
        "(Linear) interpolation of the follower forces on t \
        [[f0(t0)..f0(tn)]..[fm(t0)..fm(tn)]]",
        default=None,
    )
    dead_interpolation: list[list[int]] = dfield(
        "(Linear) interpolation of the dead forces on t \
        [[f0(t0)..f0(tn)]..[fm(t0)..fm(tn)]]",
        default=None,
    )

    gravity: float = dfield("gravity force [m/s]",
                            default=9.807)
    gravity_vect: jnp.ndarray = dfield("gravity vector",
                                       default=jnp.array([0, 0, -1]))
    label: str = dfield("""Description of the loading type:
    '1001' = follower point forces, no dead forces, no gravity, aerodynamic forces""",
                        init=False)
    def __post_init__(self):

        if self.x is not None:
            self.x = jnp.array(self.x)
        self.label = f"{int(self.follower_forces)}\
        {int(self.dead_forces)}{self.gravity_forces}{self.aero_forces}"
        
    def build_point_follower(self, num_nodes, C06ab):

        num_interpol_points = len(self.x)
        forces = jnp.zeros((num_interpol_points, 6, num_nodes))
        num_forces = len(self.follower_interpolation)
        for li in range(num_interpol_points):
            for fi in range(num_forces):
                fnode = self.follower_points[fi][0]
                dim = self.follower_points[fi][1]
                forces = forces.at[li, dim, fnode].set(
                    self.follower_interpolation[fi][li]) # Nx_6_Nn
        self.force_follower = coordinate_transform(forces, C06ab,
                                                   jax.lax.Precision.HIGHEST)
        #return self.force_follower

    def build_point_dead(self, num_nodes):

        # TODO: add gravity force, also in modes as M@g
        num_interpol_points = len(self.x)
        self.force_dead = jnp.zeros((num_interpol_points, 6, num_nodes))
        num_forces = len(self.dead_interpolation)
        for li in range(num_interpol_points):
            for fi in range(num_forces):
                fnode = self.dead_points[fi][0]
                dim = self.dead_points[fi][1]
                self.force_dead = self.force_dead.at[li, dim, fnode].set(
                    self.dead_interpolation[fi][li])

        #return self.force_dead

# @dataclass(frozen=True)
# class Dgeometry:

#     grid: str | jnp.ndarray = dfield("Grid file or array with Nodes Coordinates, node ID in the FEM and component")
#     connectivity: dict | list = dfield("Connectivities of components")
#     X: jnp.ndarray = dfield("Grid coordinates", default=None)

@dataclass
class Dfem(DataContainer):

    connectivity: dict | list = dfield("Connectivities of components")
    folder: str | pathlib.Path = dfield("""Folder in which to find Ka, Ma,
    and grid data (with those names)""", default=None)
    Ka_name: str | pathlib.Path  = dfield("Condensed stiffness matrix",
                                          default='Ka.npy')
    Ma_name: str | pathlib.Path  = dfield("Condensed mass matrix",
                                          default='Ma.npy')
    Ka: jnp.ndarray  = dfield("Condensed stiffness matrix",
                              default=None, yaml_save=False)
    Ma: jnp.ndarray  = dfield("Condensed mass matrix",
                              default=None, yaml_save=False)
    num_modes: int = dfield("Number of modes in the solution", default=None)
    eig_type: str = dfield("Calculation of eigenvalues/vectors",
                           default="scipy",
                           options=["scipy", "jax_custom", "inputs"])
    grid: str | pathlib.Path | jnp.ndarray | pd.DataFrame = dfield(
        """Grid file or array with Nodes Coordinates, node ID in the FEM,
        and associated component""", default='structuralGrid')
    df_grid: pd.DataFrame = dfield("""Data Frame associated to Grid file""", init=False)    
    X: jnp.ndarray = dfield("Grid coordinates", default=None, yaml_save=False)
    num_nodes: int = dfield("Number of nodes", init=False)    
    fe_order: list[int] | jnp.ndarray = dfield("node ID in the FEM", default=None)
    fe_order_start: int = dfield("fe_order starting with this index", default=0)
    component_vect: list[str] = dfield("Array with component associated to each node",
                                       default=None)
    num_nodes: int = dfield("Number of nodes", init=False)
    component_names: list = dfield("Name of components defining the structure", init=False)
    component_father: dict[str: str] = dfield(
        "", init=False)
    component_nodes: dict[str: list[int]] = dfield("Node indexes of the component",
                                                            init=False)    
    component_chain: dict[str:list[str]] = dfield(" ", init=False)
    #
    clamped_nodes: list[int] = dfield("List of clamped or multibody nodes", init=False)
    freeDoF: dict[str: list] = dfield("Grid coordinates", init=False)    
    clampedDoF: dict[str: list] = dfield("Grid coordinates", init=False)
    total_clampedDoF: int = dfield("Grid coordinates", init=False)
    #
    prevnodes: list[int] = dfield("""Immediate previous node following """, init=False)    
    Mavg: jnp.ndarray = dfield("Matrix for tensor average between nodes", init=False)
    Mdiff: jnp.ndarray = dfield("Matrix for tensor difference between nodes", init=False)
    Mfe_order: jnp.ndarray = dfield("""Matrix with 1s and 0s that reorders quantities
    such as eigenvectors in the FE model; nodes in horizontal arrangement.""", init=False)
    Mload_paths: jnp.ndarray = dfield("""Matrix with with 1s and 0s for the load paths
    that each node, in vertical arrangement, need to transverse to sum up to a free-end.""",
                                      init=False)
    def __post_init__(self):
        #super()
        self.connectivity = geometry.list2dict(self.connectivity)
        self.Ka_name, self.Ma_name, self.grid = geometry.find_fem(self.folder,
                                                                  self.Ka_name,
                                                                  self.Ma_name,
                                                                  self.grid)
        if self.Ka is None:
            self.Ka = load_jnp(self.Ka_name)
        if self.Ma is None:
            self.Ma = load_jnp(self.Ma_name)
        if self.num_modes is None:
            # full set of modes in the solution
            self.num_modes = len(self.Ka)
        self.df_grid, self.X, self.fe_order, self.component_vect = geometry.build_grid(
            self.grid, self.X, self.fe_order, self.fe_order_start, self.component_vect)
        self.num_nodes = len(self.X)
        self.component_names, self.component_father = geometry.compute_component_father(
            self.connectivity)
        self.component_nodes = geometry.compute_component_nodes(self.component_vect)
        self.component_chain = geometry.compute_component_chain(self.component_names,
                                                                self.connectivity)        
        self.clamped_nodes, self.freeDoF, self.clampedDoF, self.total_clampedDoF = \
            geometry.compute_clamped(self.fe_order.tolist())
        self.prevnodes = geometry.compute_prevnode(self.component_vect,
                                                   self.component_nodes,
                                                   self.component_father)
        self.Mavg = geometry.compute_Maverage(self.prevnodes, self.num_nodes)
        self.Mdiff = geometry.compute_Mdiff(self.prevnodes, self.num_nodes)
        self.Mfe_order = geometry.compute_Mfe_order(self.fe_order,
                                                    self.clamped_nodes,
                                                    self.freeDoF,
                                                    self.total_clampedDoF,
                                                    self.component_nodes,
                                                    self.component_chain,
                                                    self.num_nodes)
        self.Mload_paths = geometry.compute_Mloadpaths(self.component_vect,
                                                       self.component_nodes,
                                                       self.component_chain,
                                                       self.num_nodes)

# @dataclass(frozen=True)
# class Dpresimulation(DataContainer):

#     load: bool = dfield("""Load presimulation data vs
#     load from solution_path""",
#                                          default=True)

@dataclass(frozen=False)
class Ddriver(DataContainer):

    typeof: str = dfield("Driver to manage the simulation",
                         default=True,
                         options=['intrinsic']
                         )
    sol_path: str | pathlib.Path = dfield("Folder path to save results",
                                                default='./')
    compute_presimulation: bool = dfield("""Compute or load presimulation data""",
                                         default=True)
    save_presimulation: bool = dfield("""Save presimulation data""",
                                         default=True)

    subcases: dict[str:Dxloads] = dfield("", default=None)
    supercases: dict[str:Dfem] = dfield(
        "", default=None)

@dataclass(frozen=False)
class Dsystem(DataContainer):

    name: str = dfield("System name")
    solution: str | int = dfield("Type of solution to be solved",
                                 options=['static',
                                          'dynamic',
                                          'multibody',
                                          'stability'])    
    xloads: dict | Dxloads = dfield("External loads dataclass", default=None)
    t0: float = dfield("Initial time", default=0.)
    t1: float = dfield("Final time", default=1.)
    tn: int = dfield("Number of time steps", default=None)
    dt: float = dfield("Delta time", default=None)
    t: jnp.array = dfield("Time vector", default=None)
    solver_library: str = dfield("Library solving our system of equations", default=None)
    solver_function: str = dfield(
        "Name for the solver of the previously defined library", default=None)
    solver_settings: str = dfield(
        "Name for the solver of the previously defined library", default=None)
    nonlinear: bool = dfield(
        "whether to include the nonlinear terms in the eqs. (Gammas)", default=True)
    residualise: bool = dfield(
        "average the higher frequency eqs and make them algebraic", default=False)
    residual_modes: int = dfield(
        "number of modes to residualise", default=0)

    label: str = dfield("""Description of the loading type:
    '1001' = follower point forces, no dead forces, no gravity, aerodynamic forces""",
                        default=None)

    def __post_init__(self):

        self.xloads = initialise_Dclass(self.xloads, Dxloads)
        if self.solver_settings is None:
            self.solver_settings = dict()
        
        if  self.label is None:
            if isinstance(self.solution, str):
                sol_label = Solution[self.solution.upper()].value
            else:
                sol_label = self.solution
            self.label = f"{sol_label}{self.nonlinear}{self.residualise}{self.xloads.label}"

            if self.solver_function is None:  # set default  
                if self.label[0] == 0:
                    self.solver_function = 'newton_raphson'
                elif self.label[0] == 1:
                    self.solver_function = 'ode'
                elif self.label[0] == 2:
                    ...
                    # TODO: implement
                if self.label[0] == 3:
                    ...
                    # TODO: implement

@dataclass(frozen=False)
class Dsystems(DataContainer):

    sett: dict[str: dict] = dfield("Settings ", yaml_save=False)
    sys: dict[str: Dsystem]  = dfield("Dictionary with systems in the simulation",
                                       init=False)

    def __post_init__(self):
        self.sys = dict()
        for k, v in self.sett.items():
            self.sys[k] = initialise_Dclass(
                v, Dsystem, name=k)

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
    save_objs: bool = dfield(
        """Saves the objects output by the solution""",
        default=False
    )

    def __post_init__(self):

        if self.systems is not None:
            for k, v in self.systems:
                setattr(self, k, initialise_Dclass(v, Dsystem))

if (__name__ == '__main__'):

    d1 = Dxloads()
