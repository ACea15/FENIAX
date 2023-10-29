from dataclasses import dataclass
from typing import Sequence
import pathlib
import jax.numpy as jnp
import pathlib
import pandas as pd
from fem4inas.preprocessor.utils import dfield, initialise_Dclass, load_jnp
from fem4inas.preprocessor.containers.data_container import DataContainer
import fem4inas.intrinsic.geometry as geometry
from fem4inas.intrinsic.functions import coordinate_transform, label_generator
import jax
from enum import Enum

@dataclass(frozen=True)
class Dconst(DataContainer):

    I3: jnp.ndarray = dfield("3x3 Identity matrix", default=jnp.eye(3))
    e1: jnp.ndarray = dfield("3-component vector with beam direction in local frame",
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

@dataclass(frozen=True, kw_only=True)
class DGust(DataContainer):
    intensity: float
    
@dataclass(frozen=True, kw_only=True)
class DGust1MC(DGust):
    intensity: float

@dataclass(frozen=True, kw_only=True)
class DController(DataContainer):
    intensity: float

@dataclass(frozen=True)
class Daero(DataContainer):

    u_inf: float = dfield("", default=None)
    rho_inf: float = dfield("", default=None)
    c_ref: float = dfield("", default=None)
    qalpha: jnp.ndarray = dfield("", default=None)
    qx: jnp.ndarray = dfield("", default=None)
    #
    approx: str = dfield("", default="Roger")
    Qk_struct: list[jnp.ndarray,jnp.ndarray] = dfield("", default=None)
    Qk_gust: list[jnp.ndarray,jnp.ndarray] = dfield("", default=None)
    Qk_controls: list[jnp.ndarray,jnp.ndarray] = dfield("", default=None)
    Q0_rigid: list[jnp.ndarray,jnp.ndarray] = dfield("", default=None)
    num_poles: int = dfield("", default=None)
    gust_profile: dict = dfield("", default=None, options=["1mc"])
    gust_settings: dict = dfield("", default=None)
    gust: DGust = dfield("", init=False)
    controller_name: dict = dfield("", default=None)
    controller_settings: dict = dfield("", default=None)
    controller: DController = dfield("", init=False)
    
    def __post_init__(self):
        if self.gust_profile is not None:
            gust_class = globals()[f"DGust{self.gust_profile.upper()}"]
            gust_obj = initialise_Dclass(self.gust_settings, gust_class)
            object.__setattr__(self, "gust", gust_obj)
        else:
            object.__setattr__(self, "gust", None)
        if self.controller_name is not None:
            controller_class = globals()[f"DController{self.controller_name.upper()}"]
            controller_obj = initialise_Dclass(self.controller_settings, controller_class)
            object.__setattr__(self, "controller", controller_obj)
        else:
            object.__setattr__(self, "controller", None)

@dataclass(frozen=True)
class Dxloads(DataContainer):

    follower_forces: bool = dfield("Include point follower forces",
                                   default=False)
    dead_forces: bool = dfield("Include point dead forces",
                               default=False)
    gravity_forces: bool = dfield("Include gravity in the analysis",
                                  default=False)    
    modalaero_forces: bool = dfield("Include aerodynamic forces",
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
    # label: str = dfield("""Description of the loading type:
    # '1001' = follower point forces, no dead forces, no gravity, aerodynamic forces""",
    #                     init=False)
    def __post_init__(self):
        if self.x is not None:
             object.__setattr__(self, "x", jnp.array(self.x))
        # self.label = f"{int(self.follower_forces)}\
        # {int(self.dead_forces)}{self.gravity_forces}{self.aero_forces}"
        
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
        force_follower = coordinate_transform(forces, C06ab,
                                              jax.lax.Precision.HIGHEST)
        object.__setattr__(self, "force_follower",
                           force_follower)
        #return self.force_follower

    def build_point_dead(self, num_nodes):

        # TODO: add gravity force, also in modes as M@g
        num_interpol_points = len(self.x)
        force_dead = jnp.zeros((num_interpol_points, 6, num_nodes))
        num_forces = len(self.dead_interpolation)
        for li in range(num_interpol_points):
            for fi in range(num_forces):
                fnode = self.dead_points[fi][0]
                dim = self.dead_points[fi][1]
                force_dead = force_dead.at[li, dim, fnode].set(
                    self.dead_interpolation[fi][li])
        object.__setattr__(self, "force_dead", force_dead)
        #return self.force_dead

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
    eig_names: list[str | pathlib.Path] = dfield("""name to load
    eigenvalues/vectors in folder""",
                                                 default=["eigenvals.npy",
                                                          "eigenvecs.npy"])
    grid: str | pathlib.Path | jnp.ndarray | pd.DataFrame = dfield(
        """Grid file or array with Nodes Coordinates, node ID in the FEM,
        and associated component""", default='structuralGrid')
    df_grid: pd.DataFrame = dfield("""Data Frame associated to Grid file""", init=False)    
    X: jnp.ndarray = dfield("Grid coordinates", default=None, yaml_save=False)
    Cab_xtol: float = dfield("Tolerance for building the local frame", default=1e-4)    
    num_nodes: int = dfield("Number of nodes", init=False)    
    fe_order: list[int] | jnp.ndarray = dfield("node ID in the FEM", default=None)
    fe_order_start: int = dfield("fe_order starting with this index", default=0)
    component_vect: list[str] = dfield("Array with component associated to each node",
                                       default=None)
    num_nodes: int = dfield("Number of nodes", init=False)
    component_names: list = dfield("Name of components defining the structure", init=False)
    component_father: dict[str: str] = dfield(
        "Map between each component and its father", init=False)
    component_nodes: dict[str: list[int]] = dfield("Node indexes of the component",
                                                            init=False)
    component_names_int: tuple[int] = dfield("Name of components defining the structure", init=False)
    component_father_int: tuple[int] = dfield(
        "Map between each component and its father", init=False)
    component_nodes_int: tuple[list[int]] = dfield("Node indexes of the component",
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
        #set attributes in frozen instance
        setobj = lambda k, v: object.__setattr__(self, k, v)
        connectivity = geometry.list2dict(self.connectivity)
        setobj("connectivity", connectivity)
        Ka_name, Ma_name, grid = geometry.find_fem(self.folder,
                                                   self.Ka_name,
                                                   self.Ma_name,
                                                   self.grid)
        setobj("Ka_name", Ka_name)
        setobj("Ma_name", Ma_name)
        setobj("grid", grid)
        if self.Ka is None:
            setobj("Ka", load_jnp(self.Ka_name))
        if self.Ma is None:
            setobj("Ma", load_jnp(self.Ma_name))
            #setobj("Ma", load_jnp(self.Ma_name))
        if self.num_modes is None:
            # full set of modes in the solution
            setobj("num_modes", len(self.Ka))

        df_grid, X, fe_order, component_vect = geometry.build_grid(
            self.grid, self.X, self.fe_order, self.fe_order_start, self.component_vect)
        setobj("df_grid", df_grid)
        setobj("X", X)
        setobj("fe_order", fe_order)
        setobj("component_vect", component_vect)
        num_nodes = len(self.X)
        setobj("num_nodes", num_nodes)
        component_names, component_father = geometry.compute_component_father(
            self.connectivity)
        setobj("component_names", component_names)
        setobj("component_father", component_father)
        setobj("component_nodes", geometry.compute_component_nodes(self.component_vect))
        setobj("component_chain", geometry.compute_component_chain(self.component_names,
                                                                   self.connectivity))        
        clamped_nodes, freeDoF, clampedDoF, total_clampedDoF = \
            geometry.compute_clamped(self.fe_order.tolist())
        setobj("clamped_nodes", clamped_nodes)
        setobj("freeDoF", freeDoF)
        setobj("clampedDoF", clampedDoF)
        setobj("total_clampedDoF", total_clampedDoF)
        setobj("prevnodes", geometry.compute_prevnode(self.component_vect,
                                                      self.component_nodes,
                                                      self.component_father))
        setobj("Mavg",geometry.compute_Maverage(self.prevnodes, self.num_nodes))
        setobj("Mdiff", geometry.compute_Mdiff(self.prevnodes, self.num_nodes))
        setobj("Mfe_order", geometry.compute_Mfe_order(self.fe_order,
                                                       self.clamped_nodes,
                                                       self.freeDoF,
                                                       self.total_clampedDoF,
                                                       self.component_nodes,
                                                       self.component_chain,
                                                       self.num_nodes))
        setobj("Mload_paths", geometry.compute_Mloadpaths(self.component_vect,
                                                          self.component_nodes,
                                                          self.component_chain,
                                                          self.num_nodes))
        (component_names_int,
         component_nodes_int,
         component_father_int) = geometry.convert_components(self.component_names,
                                                                  self.component_nodes,
                                                                  self.component_father)
        setobj("component_names_int", component_names_int)
        setobj("component_nodes_int", component_nodes_int)
        setobj("component_father_int", component_father_int)
# @dataclass(frozen=True)
# class Dpresimulation(DataContainer):

#     load: bool = dfield("""Load presimulation data vs
#     load from solution_path""",
#                                          default=True)

@dataclass(frozen=True)
class Ddriver(DataContainer):

    typeof: str = dfield("Driver to manage the simulation",
                         default=True,
                         options=['intrinsic']
                         )
    sol_path: str | pathlib.Path = dfield("Folder path to save results",
                                          default='./')
    compute_fem: bool = dfield("""Compute or load presimulation data""",
                               default=True)
    save_fem: bool = dfield("""Save presimulation data""",
                            default=True)
    compute_modalaero: bool = dfield("""Compute presimulation aero data""",
                                     default=False)
    load_modalaero: bool = dfield("""Load presimulation aero data""",
                                  default=False)   
    save_modalaero: bool = dfield("""Save presimulation aero data""",
                                  default=False)
    subcases: dict[str:Dxloads] = dfield("", default=None)
    supercases: dict[str:Dfem] = dfield(
        "", default=None)
    def __post_init__(self):

        object.__setattr__(self, "sol_path",
                           pathlib.Path(self.sol_path))

class SystemSolution(Enum):
    STATIC = 1
    DYNAMIC = 2
    STABILITY = 3    
    MULTIBODY = 4
    CONTROL = 5
    
SimulationTarget = Enum('TARGET', ['LEVEL',
                                   'TRIM',
                                   'MANOEUVRE',
                                   'TURBULENCE'])
BoundaryCond = Enum('BC1', ['CLAMPED', 'FREE', 'PRESCRIBED'])

@dataclass(frozen=True)
class Dsystem(DataContainer):

    name: str = dfield("System name")
    solution: str  = dfield("Type of solution to be solved",
                            options=['static',
                                     'dynamic',
                                     'multibody',
                                     'stability'])
    target: str = dfield("The simulation goal of this system",
                         default="Level",
                         options=SimulationTarget._member_names_)
    bc1: str = dfield("Boundary condition first node",
                      default='clamped',
                      options=BoundaryCond._member_names_)
    save: bool = dfield("Save results of the run system",
                        default=True)
    xloads: dict | Dxloads = dfield("External loads dataclass",
                                    default=None)
    aero: dict | Daero = dfield("Aerodynamic dataclass",
                                default=None)
    t0: float = dfield("Initial time",
                       default=0.)
    t1: float = dfield("Final time",
                       default=1.)
    tn: int = dfield("Number of time steps",
                     default=None)
    dt: float = dfield("Delta time",
                       default=None)
    t: jnp.array = dfield("Time vector",
                          default=None)
    solver_library: str = dfield("Library solving our system of equations",
                                 default=None)
    solver_function: str = dfield(
        "Name for the solver of the previously defined library",
        default=None)
    solver_settings: str = dfield(
        "Name for the solver of the previously defined library", default=None)
    nonlinear: bool = dfield(
        """whether to include the nonlinear terms in the eqs. (Gammas)
        and in the integration""", default=1,
        options=[1, 0, -1,-2])
    residualise: bool = dfield(
        "average the higher frequency eqs and make them algebraic", default=False)
    residual_modes: int = dfield(
        "number of modes to residualise", default=0)
    label: str = dfield("""System label that maps to the solution functional""",
                        default=None)
    label_map: dict = dfield("""label dictionary assigning """,
                        default=None)
    
    states: dict = dfield("""Dictionary with the state variables.""",
                        default=None)
    num_states: int = dfield("""Total number of states""",
                        default=None)
    init_states: dict[str:list] = dfield("""Dictionary with initial conditions for each state""",
                               default=None)
    init_mapper: dict[str:str] = dfield("""Dictionary mapping states types to functions in initcond""",
                                        default=dict(q1="velocity", q2="force"))

    def __post_init__(self):

        if self.t is None:
            if self.tn is None:
                object.__setattr__(self, "tn",
                                   2)
            object.__setattr__(self, "t",
                               jnp.linspace(self.t0, self.t1, self.tn))
        if self.dt is None:
            object.__setattr__(self, "dt", self.t[1] - self.t[0])
        object.__setattr__(self, 'xloads', initialise_Dclass(self.xloads,
                                                             Dxloads))
        object.__setattr__(self, 'aero', initialise_Dclass(self.aero,
                                                           Daero))
        #self.xloads = initialise_Dclass(self.xloads, Dxloads)
        if self.solver_settings is None:
            object.__setattr__(self, "solver_settings", dict())
        
        if self.label is None:
            self.build_label()
    def build_states(self, num_modes):

        state_dict = dict()
        # if self.solution == "static":
        #     state_dict.update(m, kwargs)
        object.__setattr__(self, "states", dict(q1=jnp.arange(num_modes),
                                                q2=jnp.arange(num_modes,
                                                              2 * num_modes))
                           )
        object.__setattr__(self, "num_states",
                           sum([len(i) for i in self.states.values()])
                           )
        
    def build_label(self):
        lmap = dict()
        lmap['soltype'] = SystemSolution[self.solution.upper()].value
        lmap['target'] = SimulationTarget[self.target.upper()].value - 1
        if self.xloads.gravity_forces:
            lmap['gravity'] = "G"
        else:
            lmap['gravity'] = "g"
        lmap['bc1'] = BoundaryCond[self.bc1.upper()].value - 1
        lmap['aero_sol'] = int(self.xloads.modalaero_forces)
        if lmap['aero_sol'] > 0:
            if self.aero.sol.lower() == "rogers":
                lmap['aero_sol'] = 1
            elif self.aero.sol.lower() == "loewner":
                lmap['aero_sol'] = 2
        if self.aero.qalpha is None and self.aero.qx is None:
            lmap['aero_steady'] = 0
        elif self.aero.qalpha is not None and self.aero.qx is None:
            lmap['aero_steady'] = 1
        elif self.aero.qalpha is None and self.aero.qx is not None:
            lmap['aero_steady'] = 2
        else:
            lmap['aero_steady'] = 3
        #
        if self.aero.gust is None and self.aero.controller is None:
            lmap['aero_unsteady'] = 0
        elif self.aero.gust is not None and self.aero.controller is None:
            lmap['aero_unsteady'] = 1
        elif self.aero.gust is None and self.aero.controller is not None:
            lmap['aero_unsteady'] = 2
        else:
            lmap['aero_unsteady'] = 3
        if self.xloads.follower_forces and self.xloads.dead_forces:
            lmap["point_loads"] = 3
        elif self.xloads.follower_forces:
            lmap["point_loads"] = 1
        elif self.xloads.dead_forces:
            lmap["point_loads"] = 2
        else:
            lmap["point_loads"] = 0        
        if self.nonlinear ==1:
            lmap["nonlinear"] = ""
        elif self.nonlinear ==-1:
            lmap["nonlinear"] = "l"
        elif self.nonlinear ==-2:
            lmap["nonlinear"] = "L"
        if self.residualise:
            lmap['residualise'] = "r"
        else:
            lmap['residualise'] = ""
        labelx = list(lmap.values())
        label = label_generator(labelx)        
                     
        # TODO: label dependent
        object.__setattr__(self, "label_map", lmap)
        object.__setattr__(self, "label", f"dq_{label}")

@dataclass(frozen=True)
class Dsystems(DataContainer):

    sett: dict[str: dict] = dfield("Settings ", yaml_save=False)
    mapper: dict[str: Dsystem]  = dfield("Dictionary with systems in the simulation",
                                       init=False)

    def __post_init__(self):
        mapper = dict()
        for k, v in self.sett.items():
            mapper[k] = initialise_Dclass(
                v, Dsystem, name=k)
        object.__setattr__(self, "mapper", mapper)

@dataclass(frozen=True)
class Dsimulation(DataContainer):

    typeof: str = dfield("Type of simulation",
                         default='single',
                         options=['single', 'serial', 'parallel'])
    workflow: dict = dfield(
        """Dictionary that defines which system is run after which.
        The default None implies systems are run in order of the input""",
        default=None
    )
    save_objs: bool = dfield(
        """Saves the objects output by the solution""",
        default=False
    )

if (__name__ == '__main__'):
    pass
