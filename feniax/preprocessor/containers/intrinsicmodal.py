"""
Containers for the intrinsic modal solution settings
"""

import inspect
import math
import pathlib
import os
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import wraps
from typing import Any
from functools import partial

import feniax.intrinsic.geometry as geometry
import jax
import jax.numpy as jnp
import pandas as pd
from feniax.intrinsic.functions import (
    coordinate_transform,
    label_generator,
    reshape_field,
)
from feniax.preprocessor.containers.data_container import DataContainer
from feniax.preprocessor.utils import dfield, initialise_Dclass, load_jnp
import feniax.intrinsic.utils as iutils

def filter_kwargs(cls):
    @wraps(cls)
    def wrapper(*args, **kwargs):
        # Ensure the class is a dataclass
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} is not a dataclass")
        
        # Retrieve the field names for the target dataclass
        dataclass_field_names = {f.name for f in fields(cls)}
        
        # Separate the valid arguments and extra keyword arguments
        valid_kwargs = {k: v for k, v in kwargs.items() if k in dataclass_field_names}
        # extra_kwargs = {k: v for k, v in kwargs.items() if k not in dataclass_field_names}
        
        # Create an instance of the dataclass with the valid arguments
        instance = cls(*args, **valid_kwargs)
        
        # If the dataclass includes an extra_kwargs field, set it
        # if 'extra_kwargs' in dataclass_field_names:
        #     setattr(instance, 'extra_kwargs', extra_kwargs)
        
        return instance
    return wrapper

def Ddataclass(cls):
    return dataclass(cls, frozen=True, kw_only=True)

@filter_kwargs
@Ddataclass
class Dconst(DataContainer):
    """Constants in the configuration

    Parameters
    ----------
    I3 : Array
       3x3 Identity matrix
    e1 : Array
       3-component vector with beam direction in local frame
    EMAT : Array
       3x3 Identity matrix

    Attributes
    ----------
    EMATT : Array
       Transpose EMAT

    """

    I3: jnp.ndarray = dfield("", default=jnp.eye(3))
    e1: jnp.ndarray = dfield(
        "3-component vector with beam direction in local frame",
        default=jnp.array([1.0, 0.0, 0.0]),
    )
    EMAT: jnp.ndarray = dfield(
        "3x3 Identity matrix",
        default=jnp.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ]
        ),
    )
    EMATT: jnp.ndarray = dfield("3x3 Identity matrix", init=False)

    def __post_init__(self):
        object.__setattr__(self, "EMATT", self.EMAT.T)
        self._initialize_attributes()

@Ddataclass
class Dlog(DataContainer):
    """Simulation settings for the management the way each system is run.
    
    Parameters
    ----------
    typeof : str
        Type of simulation ["single", "serial", "parallel"]
    workflow : dict
        Dictionary that defines which system is run after which.
        The default None implies systems are run in order of the input
    """

    to_console: bool = dfield(
        "Output to console", default=True
    )
    to_file: bool = dfield(
        "Output to file", default=True
    )    
    level: str = dfield(
        "Logging level", default="info", options=["notset", "debug", "info", "warning", "error", "critical"]
    )
    file_name: str = dfield("Name of log file",
        default="feniax",
    )
    file_mode: str = dfield(
        "Mode for writing log file", default="w", options=["w", "a"]
    )
    
    path: str = dfield("Path to log file",
        default=None,
    ) 
    def __post_init__(self):
        if self.path is not None:
            object.__setattr__(self, "path", pathlib.Path(self.path))
        self._initialize_attributes()

@Ddataclass
class Dsimulation(DataContainer):
    """Simulation settings for the management the way each system is run.
    
    Parameters
    ----------
    typeof : str
        Type of simulation ["single", "serial", "parallel"]
    workflow : dict
        Dictionary that defines which system is run after which.
        The default None implies systems are run in order of the input
    """

    typeof: str = dfield(
        "", default="single", options=["single", "serial", "parallel"]
    )
    workflow: dict = dfield("",
        default=None,
    )

    def __post_init__(self):
        self._initialize_attributes()


@Ddataclass
class Ddriver(DataContainer):
    """Program initialisation settings and trigger of simulations.

    Parameters
    ----------
    typeof : str
        Driver to manage the simulation options=["intrinsic"]
    sol_path : str | pathlib.Path
        Folder path to save results   
    compute_fem : bool
        Compute or load presimulation data
    save_fem : bool
        Save presimulation data
    ad_on : bool
        Algorithm differentiation ON
    """

    typeof: str = dfield("", default=True, options=["intrinsic", "intrinsicmultibody"])
    sol_path: str | pathlib.Path = dfield("", default="./")
    compute_fem: bool = dfield("", default=True)
    save_fem: bool = dfield("", default=True)
    ad_on: bool = dfield("", default=False)
    fast_on: bool = dfield("", default=False)

    def __post_init__(self):
        if self.sol_path is not None:
            object.__setattr__(self, "sol_path", pathlib.Path(self.sol_path))
            self.sol_path.mkdir(parents=True, exist_ok=True)
            
        self._initialize_attributes()


@Ddataclass
class Dfem(DataContainer):
    """Finite Element and discretisation model settings.

    Parameters
    ----------
    connectivity : dict | list
        Connectivities between components
    folder : str | pathlib.Path
        Folder in which to find Ka, Ma, and grid data (with those names)
    Ka_name : str | pathlib.Path
        Condensed stiffness matrix name
    Ma_name : str | pathlib.Path
        Condensed mass matrix name
    Ka : Array
        Condensed stiffness matrix
    Ma : Array
        Condensed mass matrix 
    Ka0s : Array
        Condensed stiffness matrix augmented with 0s
    Ma0s : Array
        Condensed mass matrix augmented with 0s
    num_modes : int
        Number of modes in the solution
    eig_type : str
        Calculation of eigenvalues/vectors options=["scipy", "jax_custom", "inputs, input_memory"]
    eigenvals : Array
        EigenValues
    eigenvecs : Array
        EigenVectors
    eig_cutoff : float
        Cut-off frequency such that eigenvalues smaller than this are set to 0
    eig_names : list
        name to load eigenvalues/vectors in `folder`
    grid : str | pathlib.Path | jax.Array | pandas.core.frame.DataFrame
        Grid file or array with Nodes Coordinates, node ID in the FEM, and associated component
    Cab_xtol : float
        Tolerance for building the local frame

    Attributes
    ----------
    df_grid : DataFrame
        Data Frame associated to Grid file
    X : Array
        Grid coordinates    
    Xm : Array
        Grid coordinates mid-points
    fe_order : list[int] | jax.Array
        node ID in the FEM
    fe_order_start : int
        fe_order starting with this index
    component_vect : list
        Array with component associated to each node
    dof_vect : list
        Array with DoF associated to each node (for constrained systems)
    num_nodes : int
        Number of nodes 
    component_names : list
        Name of components defining the structure
    component_father : dict
        Map between each component and its father
    component_nodes : dict
        Node indexes of the component
    component_names_int : tuple
        Name of components defining the structure as integers
    component_father_int : tuple
        Map between each component and its father with integers
    component_nodes_int : tuple
        Node indexes of the component
    component_chain : dict
        Dictionary mapping each component to all the components in the load path equilibrium 
    clamped_nodes : list
        List of clamped or multibody nodes
    freeDoF : dict
    clampedDoF : dict
    total_clampedDoF : int
    constrainedDoF : int
    prevnodes : list
        Immediate previous node following 
    Mavg : Array
        Matrix for tensor average between adjent nodes
    Mdiff : Array
        Matrix for tensor difference between nodes
    Mfe_order : Array
        Matrix with 1s and 0s that reorders quantities such as eigenvectors in the FE model; nodes in horizontal arrangement.    
    Mfe_order0s : Array
    Mload_paths : Array
        Matrix with with 1s and 0s for the load paths that each node, in vertical arrangement, need to transverse to sum up to a free-end.

    """

    connectivity: dict | list = dfield("")
    folder: str | pathlib.Path = dfield(
        "",
        default=None,
    )  # yaml_save=False)
    Ka_name: str | pathlib.Path = dfield("", default="Ka.npy")
    Ma_name: str | pathlib.Path = dfield("", default="Ma.npy")
    Ka: jnp.ndarray = dfield("", default=None, yaml_save=False)
    Ma: jnp.ndarray = dfield("", default=None, yaml_save=False)
    Ka0s: jnp.ndarray = dfield(
        "", default=None, yaml_save=False
    )
    Ma0s: jnp.ndarray = dfield(
        "", default=None, yaml_save=False
    )
    num_modes: int = dfield("", default=None)
    eig_type: str = dfield(
        "",
        default="scipy",
        options=["scipy", "jax_custom", "inputs, input_memory"],
    )
    eigenvals: jnp.ndarray = dfield("", default=None, yaml_save=False)
    eigenvecs: jnp.ndarray = dfield("", default=None, yaml_save=False)  # [6Nn x Nm]
    eig_cutoff: float = dfield(
        "",
        default=1e-2,
    )  # -jnp.inf?
    eig_names: list[str | pathlib.Path] = dfield(
        "",
        default=["eigenvals.npy", "eigenvecs.npy"],
    )
    grid: str | pathlib.Path | jnp.ndarray | pd.DataFrame = dfield(
        "",
        default="structuralGrid",
    )
    Cab_xtol: float = dfield("", default=1e-4)
    df_grid: pd.DataFrame = dfield("", init=False)
    X: jnp.ndarray = dfield("", default=None, yaml_save=False)
    Xm: jnp.ndarray = dfield("", default=None, init=False)
    fe_order: list[int] | jnp.ndarray = dfield("", default=None)
    fe_order_start: int = dfield("", default=0, yaml_save=False)
    component_vect: list[str] = dfield("", default=None, yaml_save=False)
    dof_vect: list[str] = dfield(
        "", default=None, yaml_save=False
    )
    num_nodes: int = dfield("", init=False)
    component_names: list = dfield("", init=False)
    component_father: dict[str:str] = dfield(
        "", init=False
    )
    component_nodes: dict[str : list[int]] = dfield("", init=False)
    component_names_int: tuple[int] = dfield(
        "", init=False
    )
    component_father_int: tuple[int] = dfield(
        "", init=False
    )
    component_nodes_int: tuple[list[int]] = dfield("", init=False)

    component_chain: dict[str : list[str]] = dfield("", init=False)
    clamped_nodes: list[int] = dfield("", init=False)
    freeDoF: dict[str:list] = dfield("", init=False)
    clampedDoF: dict[str:list] = dfield("", init=False)
    total_clampedDoF: int = dfield("", init=False)
    constrainedDoF: int = dfield(
        "", init=False
    )
    #
    prevnodes: list[int] = dfield("", init=False)
    Mavg: jnp.ndarray = dfield("", init=False)
    Mdiff: jnp.ndarray = dfield("", init=False)
    Mfe_order: jnp.ndarray = dfield(
        "",
        init=False,
    )
    Mfe_order0s: jnp.ndarray = dfield(
        "",
        init=False,
    )
    Mload_paths: jnp.ndarray = dfield(
        "",
        init=False,
    )

    def __post_init__(self):
        # set attributes in frozen instance
        setobj = lambda k, v: object.__setattr__(self, k, v)
        connectivity = geometry.list2dict(self.connectivity)
        setobj("connectivity", connectivity)
        Ka_name, Ma_name, grid = geometry.find_fem(
            self.folder, self.Ka_name, self.Ma_name, self.grid
        )
        
        if self.folder is not None:
            setobj("folder", pathlib.Path(self.folder).absolute())
        if self.Ka is None:
            if self.folder is None:
                setobj("Ka_name", os.path.abspath(Ka_name))
                
            else:
                setobj("Ka_name", self.folder / Ka_name)

        if self.Ma is None:
            if self.folder is None:
                setobj("Ma_name", os.path.abspath(Ma_name))
            else:
                setobj("Ma_name", self.folder / Ma_name)

        if self.Ka_name is not None and self.Ka is None:
            setobj("Ka", load_jnp(self.Ka_name))
        if self.Ma_name is not None and self.Ma is None:            
            setobj("Ma", load_jnp(self.Ma_name))
        if self.eig_names is not None and self.eigenvals is None:
            eigenvals, eigenvecs = iutils.compute_eigs_load(self.num_modes,
                                                           self.folder,
                                                           self.eig_names)
                                                           
            setobj("eigenvals", eigenvals)
            setobj("eigenvecs", eigenvecs)
            if self.folder is None:
                setobj("eig_names", [os.path.abspath(self.eig_names[0]),
                                     os.path.abspath(self.eig_names[1])])
            else:
                setobj("eig_names", [os.path.abspath(self.folder / self.eig_names[0]),
                                     os.path.abspath(self.folder / self.eig_names[1])])

        if self.folder is None:
            setobj("grid", os.path.abspath(grid))

        else:
            setobj("grid", self.folder / grid)
                        
        if self.num_modes is None:
            # full set of modes in the solution
            setobj("num_modes", len(self.Ka))
        # if self.folder is None:
        #     df_grid, X, fe_order, component_vect, dof_vect = geometry.build_grid(
        #         self.grid,
        #         self.X,
        #         self.fe_order,
        #         self.fe_order_start,
        #         self.component_vect,
        #         self.dof_vect,
        #     )
        # else:
        df_grid, X, fe_order, component_vect, dof_vect = geometry.build_grid(
            self.grid,
            self.X,
            self.fe_order,
            self.fe_order_start,
            self.component_vect,
            self.dof_vect,
        )
        setobj("df_grid", df_grid)
        setobj("X", X)
        setobj("fe_order", fe_order)
        setobj("component_vect", component_vect)
        setobj("dof_vect", dof_vect)
        num_nodes = len(self.X)
        setobj("num_nodes", num_nodes)
        component_names, component_father = geometry.compute_component_father(self.connectivity)
        setobj("component_names", component_names)
        setobj("component_father", component_father)
        setobj("component_nodes", geometry.compute_component_nodes(self.component_vect))
        setobj(
            "component_chain",
            geometry.compute_component_chain(self.component_names, self.connectivity),
        )
        clamped_nodes, freeDoF, clampedDoF, total_clampedDoF, constrainedDoF = (
            geometry.compute_clamped(self.fe_order.tolist(), self.dof_vect)
        )
        setobj("clamped_nodes", clamped_nodes)
        setobj("freeDoF", freeDoF)
        setobj("clampedDoF", clampedDoF)
        setobj("total_clampedDoF", total_clampedDoF)
        setobj("constrainedDoF", constrainedDoF)
        if constrainedDoF:
            Ka0s, Ma0s = geometry.compute_Mconstrained(
                self.Ka, self.Ma, self.fe_order, clamped_nodes, clampedDoF
            )
            setobj("Ka0s", Ka0s)
            setobj("Ma0s", Ma0s)
        setobj(
            "prevnodes",
            geometry.compute_prevnode(
                self.component_vect, self.component_nodes, self.component_father
            ),
        )
        setobj("Mavg", geometry.compute_Maverage(self.prevnodes, self.num_nodes))
        setobj("Xm", jnp.matmul(self.X.T, self.Mavg))
        setobj("Mdiff", geometry.compute_Mdiff(self.prevnodes, self.num_nodes))
        Mfe_order, Mfe_order0s = geometry.compute_Mfe_order(
            self.fe_order,
            self.clamped_nodes,
            self.freeDoF,
            self.total_clampedDoF,
            self.component_nodes,
            self.component_chain,
            self.num_nodes,
        )
        setobj("Mfe_order", Mfe_order)
        setobj("Mfe_order0s", Mfe_order0s)
        setobj(
            "Mload_paths",
            geometry.compute_Mloadpaths(
                self.component_vect,
                self.component_nodes,
                self.component_chain,
                self.num_nodes,
            ),
        )
        (component_names_int, component_nodes_int, component_father_int) = (
            geometry.convert_components(
                self.component_names, self.component_nodes, self.component_father
            )
        )
        setobj("component_names_int", component_names_int)
        setobj("component_nodes_int", component_nodes_int)
        setobj("component_father_int", component_father_int)
        self._initialize_attributes()

# @partial(jax.jit, static_argnames=["min_collocationpoints",
#                                    "max_collocationpoints",
#                                    "gust_length",
#                                    "gust_step"])
def gust_discretisation(
    gust_shift,
    gust_step,
    simulation_time,
    gust_length,
    u_inf,
    min_collocationpoints,
    max_collocationpoints,
):
    # TODO: change 1e-6 as option
    gust_totaltime = gust_length / u_inf
    xgust = jnp.arange(
        min_collocationpoints,  # jnp.min(collocation_points[:,0]),
        max_collocationpoints  # jnp.max(collocation_points[:,0]) +
        + gust_length
        + gust_step,
        gust_step,
    )
    time_discretization = (gust_shift + xgust) / u_inf
    # if time_discretization[-1] < simulation_time[-1]:
    #     time = jnp.hstack(
    #         [time_discretization, time_discretization[-1] + 1e-6, simulation_time[-1]]
    #     )
    # else:
    #     time = time_discretization
    extended_time = jnp.hstack(
        [time_discretization, time_discretization[-1] + 1e-6, simulation_time[-1]]
    )
    time = jax.lax.select(time_discretization[-1] < simulation_time[-1],
                          extended_time,
                          jnp.hstack([time_discretization,
                                      time_discretization[-1] + 1e-6,
                                      time_discretization[-1] + 2*1e-6]) # need to be shame shape!!
                          )

    # if time[0] != 0.0:
    #     time = jnp.hstack([0.0, time[0] - 1e-6, time])
    time = jax.lax.select(time[0] != 0.0,
                          jnp.hstack([0.0, time[0] - 1e-6, time]),
                          jnp.hstack([0.0, 1e-6, 2*1e-6, time[1:]]))

    ntime = len(time)
    # npanels = len(collocation_points)
    return gust_totaltime, xgust, time, ntime
        

# @Ddataclass
class DGust(DataContainer):

    ...

@Ddataclass
class DGustMc(DGust):
    """1-cos gust settings (specialisation from DGust)

    Parameters
    ----------
    u_inf : float
         Flow velocity     
    simulation_time : Array
         Time array for the simulation
    intensity : float
         Gust intensity
    step : float
         Gust discretisation in x-direction --gust dx
    time_epsilon: float
         Epsilon time between the gust first hitting the AC and the next interpolation point
    length : float
         Gust length
    shift : float
         Shift gust position
    panels_dihedral : str | jax.Array
         Proportional array with cosines for dihedral 
    collocation_points : str | jax.Array
         Collocation points coordinates
    shape : str
         Span-wise shape
    
    Attributes
    ----------
    totaltime : float
        gust_length / u_inf
    x : Array
        Discretisation in flow direction
    time : Array
        Times at which gust quantities are interpolated (driven by step and u_inf) 
    ntime : int
        len(time)

    """

    u_inf: float = dfield("", default=None)
    simulation_time: jnp.ndarray = dfield("", default=None, yaml_save=False)
    intensity: float = dfield("", default=None)
    step: float = dfield("", default=None)
    time_epsilon: float = dfield("", default=1e-6)
    length: float = dfield("", default=None)
    shift: float = dfield("", default=0.0)
    panels_dihedral: str | jnp.ndarray = dfield("", default=None, yaml_save=False)
    collocation_points: str | jnp.ndarray = dfield("", default=None, yaml_save=False)
    collocation_points_path: str = dfield("", default=None)
    shape: str = dfield("",
                        default="const")
    fixed_discretisation: dict[str: float] = dfield("",
                                                    default=None)
    totaltime: float = dfield("", init=False)
    x: jnp.ndarray = dfield("", init=False)
    time: jnp.ndarray = dfield("", init=False)
    ntime: int = dfield("", init=False)

    def __post_init__(self):
        if isinstance(self.panels_dihedral, (str, pathlib.Path)):
            object.__setattr__(self, "panels_dihedral", jnp.load(self.panels_dihedral))
        if isinstance(self.collocation_points, (str, pathlib.Path)):
            object.__setattr__(self,
                               "collocation_points_path",
                               os.path.abspath(self.collocation_points)
                               )            
            object.__setattr__(self, "collocation_points", jnp.load(self.collocation_points))
            
        elif self.collocation_points_path is not None:
            object.__setattr__(self, "collocation_points", jnp.load(self.collocation_points_path))
            object.__setattr__(self,
                               "collocation_points_path",
                               os.path.abspath(self.collocation_points_path)
                               )
        object.__setattr__(self, "panels_dihedral", jnp.array(self.panels_dihedral))
        object.__setattr__(self, "collocation_points", jnp.array(self.collocation_points))    
        # self.panels_dihedral = jnp.array(self.panels_dihedral)
        # self.collocation_points = jnp.array(self.collocation_points)
        
        # gust_totaltime, xgust, time, ntime = self._set_gustDiscretization(
        #     self.intensity,
        #     self.panels_dihedral,
        #     self.shift,
        #     self.step,
        #     self.simulation_time,
        #     self.length,
        #     self.u_inf,
        #     jnp.min(self.collocation_points[:, 0]),
        #     jnp.max(self.collocation_points[:, 0]),
        # )
        if self.fixed_discretisation is None:
            gust_totaltime, xgust, time, ntime = gust_discretisation(
                self.shift,
                self.step,
                self.simulation_time,
                self.length,
                self.u_inf,
                float(jnp.min(self.collocation_points[:, 0])),
                float(jnp.max(self.collocation_points[:, 0])),
            )
        else:
            gust_totaltime, xgust, time, ntime = gust_discretisation(
                self.shift,
                self.step,
                self.simulation_time,
                self.fixed_discretisation[0],
                self.fixed_discretisation[1],                
                float(jnp.min(self.collocation_points[:, 0])),
                float(jnp.max(self.collocation_points[:, 0])),
            )
        
        object.__setattr__(self, "totaltime", gust_totaltime)
        object.__setattr__(self, "x", xgust)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "ntime", ntime)
        # del self.simulation_time
        self._initialize_attributes()


@Ddataclass
class DController(DataContainer): ...


@Ddataclass
class Daero(DataContainer):
    """Modal aerodynamic settings for each system

    Parameters
    ----------
    u_inf : float
        Flow velocity
    rho_inf : float
        Flow density
    q_inf : float
        Flow dynamic pressure
    c_ref : float
        Reference chord
    time : Array
        Simulation time array
    qalpha : Array
    qx : Array
    elevator_index : Array
    elevator_link : Array
    approx : str
        Aero approximation Options = Roger
    Qk_struct : list
        Sample frequencies and corresponding AICs for the structure
    Qk_gust : list
    Qk_controls : list
    Q0_rigid : Array
    A : str | jax.Array
    B : str | jax.Array
    C : str | jax.Array
    D : str | jax.Array
    _controls : list
    poles : str | jax.Array
         Poles array
    num_poles : int
         Number of poles
    gust_profile : str
        Gust name options=["mc"]
    gust : dict | __main__.DGust
        Gust settings 
    controller_name : dict
    controller_settings : dict
    controller : DController

    """
    
    u_inf: float = dfield("", default=None)
    rho_inf: float = dfield("", default=None)
    q_inf: float = dfield("", init=False)
    c_ref: float = dfield("", default=None)
    time: jnp.ndarray = dfield("", default=None, yaml_save=False)
    qalpha: jnp.ndarray = dfield("", default=None)
    qx: jnp.ndarray = dfield("", default=None)
    elevator_index: jnp.ndarray = dfield("", default=None)
    elevator_link: jnp.ndarray = dfield("", default=None)
    #
    approx: str = dfield("", default="Roger")
    Qk_struct: list[jnp.ndarray, jnp.ndarray] = dfield(
        "",
        default=None,
        yaml_save=False,
    )
    Qk_gust: list[jnp.ndarray, jnp.ndarray] = dfield("", default=None, yaml_save=False)
    Qk_controls: list[jnp.ndarray, jnp.ndarray] = dfield("", default=None, yaml_save=False)
    Q0_rigid: jnp.ndarray = dfield("", default=None, yaml_save=False)
    A: str | jnp.ndarray = dfield("", default=None, yaml_save=False)
    B: str | jnp.ndarray = dfield("", default=None, yaml_save=False)
    C: str | jnp.ndarray = dfield("", default=None, yaml_save=False)
    D: str | jnp.ndarray = dfield("", default=None, yaml_save=False)
    _controls: list[jnp.ndarray, jnp.ndarray] = dfield("", default=None)
    poles: str | jnp.ndarray = dfield("", default=None)
    num_poles: int = dfield("", default=None)
    gust_profile: str = dfield("", default="mc", options=["mc"])
    # gust_settings: dict = dfield("", default=None, yaml_save=False)
    gust: dict | DGust = dfield("Gust settings", default=None)
    controller_name: dict = dfield("", default=None)
    controller_settings: dict = dfield("", default=None)
    controller: DController = dfield("", init=False)

    def __post_init__(self):
        object.__setattr__(self, "approx", self.approx.capitalize())
        if self.gust is not None:
            gust_class = globals()[f"DGust{self.gust_profile.capitalize()}"]
            object.__setattr__(
                self,
                "gust",
                # initialise_Dclass(self.gust, gust_class))
                initialise_Dclass(
                    self.gust, gust_class, u_inf=self.u_inf, simulation_time=self.time
                ),
            )

        if self.controller_name is not None:
            controller_class = globals()[f"DController{self.controller_name.upper()}"]
            controller_obj = initialise_Dclass(self.controller_settings, controller_class)
            object.__setattr__(self, "controller", controller_obj)
        else:
            object.__setattr__(self, "controller", None)
        if isinstance(self.poles, (str, pathlib.Path)):
            object.__setattr__(self, "poles", jnp.load(self.poles))
        if self.elevator_link is not None:
            object.__setattr__(self, "elevator_link", jnp.array(self.elevator_link))
        if self.elevator_index is not None:
            object.__setattr__(self, "elevator_index", jnp.array(self.elevator_index))
        if self.poles is not None:
            object.__setattr__(self, "num_poles", len(self.poles))
        if self.u_inf is not None and self.rho_inf is not None:
            q_inf = 0.5 * self.rho_inf * self.u_inf**2
            object.__setattr__(self, "q_inf", q_inf)
        if isinstance(self.Q0_rigid, (str, pathlib.Path)):
            object.__setattr__(self, "Q0_rigid", jnp.load(self.Q0_rigid))            
        if isinstance(self.A, (str, pathlib.Path)):
            object.__setattr__(self, "A", jnp.load(self.A))
        if isinstance(self.B, (str, pathlib.Path)):
            object.__setattr__(self, "B", jnp.load(self.B))
        if isinstance(self.C, (str, pathlib.Path)):
            object.__setattr__(self, "C", jnp.load(self.C))
        if isinstance(self.D, (str, pathlib.Path)):
            object.__setattr__(self, "D", jnp.load(self.D))
        if self.qalpha is not None and not isinstance(self.qalpha, jnp.ndarray):
            object.__setattr__(self, "qalpha", jnp.array(self.qalpha))

        self._initialize_attributes()


@Ddataclass
class Dxloads(DataContainer):
    """External loads settings for each system

    Parameters
    ----------
    follower_forces : bool
        Include point follower forces
    dead_forces : bool
    gravity_forces : bool
    modalaero_forces : bool
    x : Array
    force_follower : Array
    force_dead : Array
    follower_points : list
    dead_points : list
    follower_interpolation : list
    dead_interpolation : list
    gravity : float
    gravity_vect : Array

    Attributes
    ----------

    Methods
    -------
    build_point_follower
    build_point_dead
    build_gravity

    """

    follower_forces: bool = dfield("", default=False)
    dead_forces: bool = dfield("", default=False)
    gravity_forces: bool = dfield("Include gravity in the analysis", default=False)
    modalaero_forces: bool = dfield("Include aerodynamic forces", default=False)
    x: jnp.ndarray = dfield("x-axis vector for interpolation", default=None)
    force_follower: jnp.ndarray = dfield(
        """Point follower forces
    (len(x)x6xnum_nodes)""",
        default=None,
    )
    force_dead: jnp.ndarray = dfield(
        """Point follower forces
    (len(x)x6xnum_nodes)""",
        default=None,
    )
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

    gravity: float = dfield("gravity force [m/s]", default=9.807)
    gravity_vect: jnp.ndarray = dfield("gravity vector", default=jnp.array([0, 0, -1]))

    # gravity_steps: int = dfield("steps in which gravity is applied in trim simulation",
    #                                    default=1) manage by t
    # label: str = dfield("""Description of the loading type:
    # '1001' = follower point forces, no dead forces, no gravity, aerodynamic forces""",
    #                     init=False)
    def __post_init__(self):
        if self.x is not None:
            object.__setattr__(self, "x", jnp.array(self.x))
        else:
            object.__setattr__(self, "x", jnp.linspace(0, 1, 2))
        # self.label = f"{int(self.follower_forces)}\
        # {int(self.dead_forces)}{self.gravity_forces}{self.aero_forces}"
        self._initialize_attributes()

    def build_point_follower(self, num_nodes, C06ab):
        num_interpol_points = len(self.x)
        forces = jnp.zeros((num_interpol_points, 6, num_nodes))
        num_forces = len(self.follower_interpolation)
        for li in range(num_interpol_points):
            for fi in range(num_forces):
                fnode = self.follower_points[fi][0]
                dim = self.follower_points[fi][1]
                forces = forces.at[li, dim, fnode].set(
                    self.follower_interpolation[fi][li]
                )  # Nx_6_Nn
        force_follower = coordinate_transform(forces, C06ab, jax.lax.Precision.HIGHEST)
        object.__setattr__(self, "force_follower", force_follower)
        # return self.force_follower

    def build_point_dead(self, num_nodes):
        # TODO: add gravity force, also in modes as M@g
        num_interpol_points = len(self.x)
        force_dead = jnp.zeros((num_interpol_points, 6, num_nodes))
        num_forces = len(self.dead_interpolation)
        for li in range(num_interpol_points):
            for fi in range(num_forces):
                fnode = self.dead_points[fi][0]
                dim = self.dead_points[fi][1]
                force_dead = force_dead.at[li, dim, fnode].set(self.dead_interpolation[fi][li])
        object.__setattr__(self, "force_dead", force_dead)
        # return self.force_dead

    def build_gravity(self, Ma, Mfe_order):
        num_nodes = Mfe_order.shape[1] // 6
        num_nodes_out = Mfe_order.shape[0] // 6
        if self.x is not None and len(self.x) > 1:
            len_x = len(self.x)
        else:
            len_x = 2
        # force_gravity = jnp.zeros((2, 6, num_nodes))
        gravity = self.gravity * self.gravity_vect
        gravity_field = jnp.hstack([jnp.hstack([gravity, 0.0, 0.0, 0.0])] * num_nodes)
        _force_gravity = jnp.matmul(Mfe_order, Ma @ gravity_field)
        gravity_interpol = jnp.vstack([xi * _force_gravity for xi in jnp.linspace(0, 1, len_x)]).T
        force_gravity = reshape_field(
            gravity_interpol, len_x, num_nodes_out
        )  # Becomes  (len_x, 6, Nn)
        # num_forces = len(self.dead_interpolation)
        # for li in range(num_interpol_points):
        #     for fi in range(num_forces):
        #         fnode = self.dead_points[fi][0]
        #         dim = self.dead_points[fi][1]
        #         force_dead = force_dead.at[li, dim, fnode].set(
        #             self.dead_interpolation[fi][li])
        object.__setattr__(self, "force_gravity", force_gravity)


# @Ddataclass
# class Dgeometry:

#     grid: str | jnp.ndarray = dfield("Grid file or array with Nodes Coordinates, node ID in the FEM and component")
#     connectivity: dict | list = dfield("Connectivities of components")
#     X: jnp.ndarray = dfield("Grid coordinates", default=None)


# @Ddataclass
# class Dpresimulation(DataContainer):

#     load: bool = dfield("""Load presimulation data vs
#     load from solution_path""",
#                                          default=True)


class SystemSolution(Enum):
    STATIC = 1
    DYNAMIC = 2
    # STATICAD = 3
    # DYNAMICAD = 4
    STABILITY = 3
    MULTIBODY = 4
    CONTROL = 5


SimulationTarget = Enum("TARGET", ["LEVEL", "TRIM", "MANOEUVRE", "TURBULENCE"])
BoundaryCond = Enum("BC1", ["CLAMPED", "FREE", "PRESCRIBED"])



@Ddataclass
class Dlibrary(DataContainer):
    """Solution library"""

    function: str = dfield("Function wrapper calling the library", default=None)

@filter_kwargs
@Ddataclass
class DdiffraxOde(Dlibrary):
    """Settings for Diffrax ODE solvers

    Parameters
    ----------
    root_finder : dict
    stepsize_controller : dict
    solver_name : str
    save_at : jax.Array | list
    max_steps : int

    """

    root_finder: dict = dfield("", default=None)
    stepsize_controller: dict = dfield("", default=None)
    solver_name: str = dfield("", default="Dopri5")
    save_at: jnp.ndarray | list[float] = dfield("", default=None)
    max_steps: int = dfield("", default=20000)

    def __post_init__(self, **kwargs):
        object.__setattr__(self, "function", "ode")
        self._initialize_attributes()


@Ddataclass
class Drunge_kuttaOde(Dlibrary):
    """Solution settings for Runge-Kutta in-house solvers

    Parameters
    ----------
    solver_name : str
    """

    solver_name: str = dfield("", default="rk4")

    def __post_init__(self):
        object.__setattr__(self, "function", "ode")
        self._initialize_attributes()


@Ddataclass
class DdiffraxNewton(Dlibrary):
    """Settings for Diffrax Newton solver

    Parameters
    ----------
    rtol : float
    atol : float
    max_steps : int
    norm : str
    kappa : float

    """

    rtol: float = dfield("", default=1e-7)
    atol: float = dfield("", default=1e-7)
    max_steps: int = dfield("", default=100)
    norm: str = dfield("", default="linalg_norm")
    kappa: float = dfield("", default=0.01)

    def __post_init__(self):
        object.__setattr__(self, "function", "newton")
        self._initialize_attributes()


@Ddataclass
class DobjectiveArgs(DataContainer):
    """Settings for the objective function in the AD

    Parameters
    ----------
    function : str
    nodes : tuple
    t : tuple
    components : tuple
    axis : int
    _numtime : int
    _numnodes : int
    _numcomponents : int

    """

    nodes: tuple = dfield("", default=None)
    t: tuple = dfield("", default=None)
    components: tuple = dfield("", default=None)
    axis: int = dfield("", default=None)
    _numtime: int = dfield("", default=None)
    _numnodes: int = dfield("", default=None)
    _numcomponents: int = dfield("", default=6)

    def __post_init__(self):
        if self.nodes is None:
            object.__setattr__(self, "nodes", tuple(range(self._numnodes)))
        if self.t is None:
            object.__setattr__(self, "t", tuple(range(self._numtime)))
        if self.components is None:
            object.__setattr__(self, "components", tuple(range(self._numcomponents)))
        self._initialize_attributes()


class ADinputType(Enum):
    POINT_FORCES = 1
    GUST1 = 2
    FEM = 3


@Ddataclass
class DtoAD(DataContainer):
    """Algorithm differentiation settings

    Parameters
    ----------
    function : str
    inputs : dict
    input_type : str
    grad_type : str
    objective_fun : str
    objective_var : str
    objective_args : dict
    _numnodes : int
    _numtime : int
    _numcomponents : int
    label : str

    """

    inputs: dict = dfield("", default=None, yaml_save=False)
    input_type: str = dfield("", default=None, options=ADinputType._member_names_)
    grad_type: str = dfield(
        "",
        default=None,
        options=[  # "grad", "value_grad",
            "jacrev",
            "jacfwd",
            "value",
        ],
    )
    objective_fun: str = dfield("", default=None)
    objective_var: str = dfield("", default=None)
    objective_args: dict | DobjectiveArgs = dfield("", default=None, yaml_save=False)
    _numnodes: int = dfield("", default=None, yaml_save=False)
    _numtime: int = dfield("", default=None, yaml_save=False)
    _numcomponents: int = dfield("", default=6, yaml_save=False)
    label: str = dfield("", default=None, init=False)

    def __post_init__(self):
        label = ADinputType[self.input_type.upper()].value
        object.__setattr__(self, "label", label)

        object.__setattr__(
            self,
            "objective_args",
            initialise_Dclass(
                self.objective_args,
                DobjectiveArgs,
                _numtime=self._numtime,
                _numnodes=self._numnodes,
                _numcomponents=self._numcomponents,
            ),
        )
        self._initialize_attributes()

class ShardinputType(Enum):
    POINTFORCES = 1
    STEADYALPHA = 2
    GUST1 = 3


@Ddataclass
class DShard_pointforces(DataContainer):
    """Point forces

    Parameters
    ----------

    """

    follower_points: jnp.ndarray = dfield("", default=None)
    follower_interpolation: jnp.ndarray = dfield("", default=None)
    dead_points: jnp.ndarray = dfield("", default=None)
    dead_interpolation: jnp.ndarray = dfield("", default=None)
    gravity: jnp.ndarray = dfield("", default=None)
    gravity_vect: jnp.ndarray = dfield("", default=None)
    
    def __post_init__(self):
        if self.follower_points is not None:
            object.__setattr__(self, "follower_points", jnp.array(self.follower_points))
        if self.follower_interpolation is not None:
            object.__setattr__(self, "follower_interpolation", jnp.array(self.follower_interpolation))
        if self.dead_points is not None:
            object.__setattr__(self, "dead_points", jnp.array(self.dead_points))
        if self.dead_interpolation is not None:                 
            object.__setattr__(self, "dead_interpolation", jnp.array(self.dead_interpolation))
        if self.gravity is not None:
            object.__setattr__(self, "gravity", jnp.array(self.gravity))
        if self.gravity_vect is not None:            
            object.__setattr__(self, "gravity_vect", jnp.array(self.gravity_vect))        
        self._initialize_attributes()

@Ddataclass
class DShard_steadyalpha(DataContainer):
    """Point forces

    Parameters
    ----------

    """

    rho_inf: jnp.ndarray = dfield("", default=None)
    u_inf: jnp.ndarray = dfield("", default=None)
    aeromatrix: list[int] = dfield("", default=None)
    def __post_init__(self):
        
        self._initialize_attributes()
        
@Ddataclass
class DShard_gust1(DataContainer):
    """Point forces

    Parameters
    ----------

    """
    rho_inf: jnp.ndarray = dfield("", default=None)
    u_inf: jnp.ndarray = dfield("", default=None)
    length: jnp.ndarray = dfield("", default=None)
    intensity: jnp.ndarray = dfield("", default=None)
    def __post_init__(self):
        
        self._initialize_attributes()
    
    
@Ddataclass
class DShard(DataContainer):
    """ settings

    Parameters
    ----------
    input_type : str
    label : str

    """

    inputs: dict = dfield("", default=None, yaml_save=False)
    input_type: str = dfield("", default=None, options=ShardinputType._member_names_)
    label: str = dfield("", default=None, init=False)

    def __post_init__(self):
        label = ShardinputType[self.input_type.upper()].value
        object.__setattr__(self, "label", label)
        input_class = globals()[f"DShard_{self.input_type.lower()}"]
        object.__setattr__(
            self,
            "inputs",
            initialise_Dclass(
                self.inputs,
                input_class
            ),
        )
        self._initialize_attributes()

@Ddataclass
class Dsystem(DataContainer):
    """System settings for the corresponding equations to be solved

    Parameters
    ----------
    name : str
    _fem : Dfem
    solution : str
    target : str
    bc1 : str
    save : bool
    xloads : dict | __main__.Dxloads
    aero : dict | __main__.Daero
    t0 : float
    t1 : float
    tn : int
    dt : float
    t : Array
    solver_library : str
    solver_function : str
    solver_settings : str
    q0treatment : int
    rb_treatment : int
    nonlinear : bool
    residualise : bool
    residual_modes : int
    label : str
    label_map : dict
    states : dict
    num_states : int
    init_states : dict
    init_mapper : dict
    ad : DtoAD

    """

    name: str = dfield("System name", default="sys1")
    _fem: Dfem = dfield("", default=None, yaml_save=False)
    solution: str = dfield(
        "Type of solution to be solved",
        options=["static", "dynamic", "multibody", "stability"],
    )
    target: str = dfield(
        "The simulation goal of this system",
        default="Level",
        options=SimulationTarget._member_names_,
    )
    bc1: str = dfield(
        "Boundary condition first node",
        default="clamped",
        options=BoundaryCond._member_names_,
    )
    operationalmode: str = dfield(
        "",
        options=["(empty string/default)", "Fast", "AD", "Shard", "ShardMap", "ShardAD"],
        default=""
    )
    
    save: bool = dfield("Save results of the run system", default=True)
    xloads: dict | Dxloads = dfield("External loads dataclass", default=None)
    aero: dict | Daero = dfield("Aerodynamic dataclass", default=None)
    t0: float = dfield("Initial time", default=0.0)
    t1: float = dfield("Final time", default=1.0)
    tn: int = dfield("Number of time steps", default=None)
    dt: float = dfield("Delta time", default=None)
    t: jnp.ndarray = dfield("Time vector", default=None, yaml_save=False)
    solver_library: str = dfield("Library solving our system of equations", default=None)
    solver_function: str = dfield(
        "Name for the solver of the previously defined library", default=None
    )
    solver_settings: str = dfield("Settings for the solver", default=None)
    q0treatment: int = dfield(
        """Modal velocities, q1, and modal forces, q2, are the main variables
        in the intrinsic structural description,
        but the steady aerodynamics part needs a displacement component, q0;
        proportional gain to q2 or  integration of velocities q1
        can be used to obtain this.""",
        default=2,
        options=[2, 1],
    )
    rb_treatment: int = dfield(
        """Rigid-body treatment: 1 to use the first node quaternion to track the body
        dynamics (integration of strains thereafter; 2 to use quaternions at every node.)""",
        default=1,
        options=[1, 2],
    )
    nonlinear: bool = dfield(
        """whether to include the nonlinear terms in the eqs. (Gammas)
        and in the integration""",
        default=1,
        options=[1, 0, -1, -2],
    )
    residualise: bool = dfield(
        "average the higher frequency eqs and make them algebraic", default=False
    )
    residual_modes: int = dfield("number of modes to residualise", default=0)
    label: str = dfield("""System label that maps to the solution functional""", default=None)
    label_map: dict = dfield("""label dictionary assigning """, default=None)

    states: dict = dfield("""Dictionary with the state variables.""", default=None)
    num_states: int = dfield("""Total number of states""", default=None)
    init_states: dict[str:list] = dfield(
        """Dictionary with initial conditions for each state""", default=None
    )
    init_mapper: dict[str:str] = dfield(
        """Dictionary mapping states types to functions in initcond""",
        default=dict(q1="velocity", q2="force"),
    )
    ad: dict | DtoAD = dfield("""Dictionary for AD""", default=None)
    shard: dict | DShard = dfield("""Dictionary for parallelisation""", default=None)

    def __post_init__(self):
        if self.t is not None:
            object.__setattr__(self, "t", jnp.array(self.t))
            object.__setattr__(self, "t1", self.t[-1])
            if (len_t := len(self.t)) < 2:
                object.__setattr__(self, "dt", 0.0)
            else:
                object.__setattr__(self, "dt", self.t[1] - self.t[0])
            object.__setattr__(self, "tn", len_t)
        else:
            if self.dt is not None and self.tn is not None:
                object.__setattr__(self, "t1", self.t0 + (self.tn - 1) * self.dt)
            elif self.tn is not None and self.t1 is not None:
                object.__setattr__(self, "dt", (self.t1 - self.t0) / (self.tn - 1))
            elif self.t1 is not None and self.dt is not None:
                object.__setattr__(self, "tn", math.ceil((self.t1 - self.t0) / self.dt + 1))
                object.__setattr__(self, "t1", self.t0 + (self.tn - 1) * self.dt)
            object.__setattr__(self, "t", jnp.linspace(self.t0, self.t1, self.tn))

        object.__setattr__(self, "xloads", initialise_Dclass(self.xloads, Dxloads))
        if self.aero is not None:
            object.__setattr__(self, "aero", initialise_Dclass(self.aero, Daero, time=self.t))
        # self.xloads = initialise_Dclass(self.xloads, Dxloads)
        if self.solver_settings is None:
            object.__setattr__(self, "solver_settings", dict())

        libsettings_class = globals()[f"D{self.solver_library}{self.solver_function.capitalize()}"]
        object.__setattr__(
            self,
            "solver_settings",
            initialise_Dclass(self.solver_settings, libsettings_class),
        )
        if self.ad is not None:

            if isinstance(self.ad, dict):
                libsettings_class = globals()["DtoAD"]
                object.__setattr__(
                    self,
                    "ad",
                    initialise_Dclass(
                        self.ad,
                        libsettings_class,
                        _numtime=len(self.t),
                        _numnodes=self._fem.num_nodes,
                    ),
                )
            if self.shard is not None:
                object.__setattr__(self, "operationalmode", "ADShard")
                if isinstance(self.shard, dict):
                    libsettings_class = globals()["DShard"]
                    object.__setattr__(
                        self,
                        "shard",
                        initialise_Dclass(
                            self.shard,
                            libsettings_class,
                            #_fem=self._fem,
                            #_aero=self._aero,
                        ),
                    )
                
            else:
                object.__setattr__(self, "operationalmode", "AD")
                
        elif self.shard is not None:
            if self.operationalmode == "":
                object.__setattr__(self, "operationalmode", "Shard")
            else:
                object.__setattr__(self, "operationalmode", self.operationalmode.capitalize())
            if isinstance(self.shard, dict):
                libsettings_class = globals()["DShard"]
                object.__setattr__(
                    self,
                    "shard",
                    initialise_Dclass(
                        self.shard,
                        libsettings_class,
                        #_fem=self._fem,
                        #_aero=self._aero,
                    ),
                )
        elif self.operationalmode == "fast":
            object.__setattr__(self, "operationalmode", "Fast")
        if self.label is None:
            self.build_label()
        self._initialize_attributes()

    def build_states(self, num_modes: int, num_nodes: int):
        
        num_poles = 0
        if self.label_map["aero_sol"] and self.aero.approx.lower() == "roger":
            num_poles = self.aero.num_poles
        tracker = iutils.build_systemstates(self.solution, self.target, self.bc1, self.rb_treatment, self.q0treatment, num_poles, num_modes, num_nodes)
    
        # if self.solution == "static":
        #     state_dict.update(m, kwargs)
        object.__setattr__(self, "states", tracker.states)
        object.__setattr__(self, "num_states", tracker.num_states)

    def build_label(self):
        # WARNING: order dependent for the label
        # nonlinear and residualise should always come last as they are represented
        # with letters
        lmap = dict()
        lmap["soltype"] = SystemSolution[self.solution.upper()].value
        lmap["target"] = SimulationTarget[self.target.upper()].value - 1
        if self.xloads.gravity_forces:
            lmap["gravity"] = "G"
        else:
            lmap["gravity"] = "g"
        lmap["bc1"] = BoundaryCond[self.bc1.upper()].value - 1
        lmap["aero_sol"] = int(self.xloads.modalaero_forces)
        if lmap["aero_sol"] > 0:
            if self.aero.approx.lower() == "roger":
                lmap["aero_sol"] = 1
            elif self.aero.approx.lower() == "loewner":
                lmap["aero_sol"] = 2
            if self.aero.qalpha is None and self.aero.qx is None:
                lmap["aero_steady"] = 0
            elif self.aero.qalpha is not None and self.aero.qx is None:
                lmap["aero_steady"] = 1
            elif self.aero.qalpha is None and self.aero.qx is not None:
                lmap["aero_steady"] = 2
            else:
                lmap["aero_steady"] = 3
            #
            if self.aero.gust is None and self.aero.controller is None:
                lmap["aero_unsteady"] = 0
            elif self.aero.gust is not None and self.aero.controller is None:
                lmap["aero_unsteady"] = 1
            elif self.aero.gust is None and self.aero.controller is not None:
                lmap["aero_unsteady"] = 2
            else:
                lmap["aero_unsteady"] = 3
        else:
            lmap["aero_steady"] = 0
            lmap["aero_unsteady"] = 0
        if self.xloads.follower_forces and self.xloads.dead_forces:
            lmap["point_loads"] = 3
        elif self.xloads.follower_forces:
            lmap["point_loads"] = 1
        elif self.xloads.dead_forces:
            lmap["point_loads"] = 2
        else:
            lmap["point_loads"] = 0
        if self.q0treatment == 2:
            lmap["q0treatment"] = 0
        elif self.q0treatment == 1:
            lmap["q0treatment"] = 1
        if self.nonlinear == 1:
            lmap["nonlinear"] = ""
        elif self.nonlinear == -1:
            lmap["nonlinear"] = "l"
        elif self.nonlinear == -2:
            lmap["nonlinear"] = "L"
        if self.residualise:
            lmap["residualise"] = "r"
        else:
            lmap["residualise"] = ""
        labelx = list(lmap.values())
        label = label_generator(labelx)

        # TODO: label dependent
        object.__setattr__(self, "label_map", lmap)
        object.__setattr__(self, "label", label)  # f"dq_{label}")


@Ddataclass
class Dsystems(DataContainer):
    """Input setting for the range of systems in the simulation

    Parameters
    ----------
    sett : dict
    mapper : dict
    borrow : dict
    _fem : Dfem

    Attributes
    ----------


    """

    sett: dict[str:dict] = dfield("Settings ", yaml_save=True)
    mapper: dict[str:Dsystem] = dfield("Dictionary with systems in the simulation", init=False)
    borrow: dict[str:str] = dfield(
        """Borrow settings from another system:
    if there is only one system, then inactive; otherwise default to take settings from
    the first system unless specified.
    """,
        default=None,
    )
    _fem: Dfem = dfield("", default=None, yaml_save=False)

    def __post_init__(self):
        mapper = dict()
        counter = 0
        for k, v in self.sett.items():
            if self.borrow is None:
                # pass self._fem to the system here, the others should already have
                # a reference
                mapper[k] = initialise_Dclass(v, Dsystem, name=k, _fem=self._fem)
            elif isinstance(self.borrow, str):
                assert self.borrow in self.sett.keys(), "borrow not in system names"
                if k == self.borrow:
                    mapper[k] = initialise_Dclass(v, Dsystem, name=k)
                else:
                    v0 = self.sett[self.borrow]
                    mapper[k] = initialise_Dclass(v0, Dsystem, name=k, **v)
            else:
                if k in self.borrow.keys():
                    v0 = self.sett[self.borrow[k]]
                    mapper[k] = initialise_Dclass(v0, Dsystem, name=k, **v)
                else:
                    mapper[k] = initialise_Dclass(v, Dsystem, name=k)

            counter += 1
        object.__setattr__(self, "mapper", mapper)
        self._initialize_attributes()


@Ddataclass
class Dconstraint(DataContainer):

    type_name: str = dfield("", default="spherical")
    node: int = dfield("", default=None)
    body: str = dfield("", default=None)
    node_father: int = dfield("", default=None)
    body_father: str = dfield("", default=None)    
    axis: jnp.ndarray = dfield("", default=None)
    
    def __post_init__(self):
        self._initialize_attributes()
        
@Ddataclass
class Dmultibody(DataContainer):

    num_body: int = dfield("", default=0)
    num_constraints: int = dfield("", default=0)    
    name_body: list[str] = dfield("", default=None)
    fems: dict[str: Dfem] = dfield("", default=None)
    fems_input: dict =dfield("", default=None, yaml_save=False)
    systems: dict[str: Dsystem] = dfield("", default=None)
    systems_input: dict = dfield("", default=None, yaml_save=False)
    constraints: dict[str: Dconstraint] = dfield("", default=None)
    constraints_input: dict = dfield("", default=None)
    
    def __post_init__(self):
        object.__setattr__(self, "fems", dict())
        object.__setattr__(self, "systems", dict())
        object.__setattr__(self, "constraints", dict())
        for k, v in self.fems_input.items():
            self.fems[k] = Dfem(**v)
        for k, v in self.systems_input.items():            
            self.systems[k] = Dsystem(**v, _fem=self.fems[k])
        for k, v in self.constraints_input.items():            
            self.constraints[k] = Dconstraint(**v)
        self._initialize_attributes()

        
class ForagerinputType(Enum):
    SHARD2ADGUST= 1
        
@Ddataclass
class Dforager(Dlibrary):

    typeof: str = dfield("Type of forager",
                         default=None,
                         options=ForagerinputType._member_names_)
    settings: dict = dfield("", default=None)
    
    def __post_init__(self):
        libsettings_class = globals()[f"Dforager_{self.typeof}"]
        object.__setattr__(
            self,
            "settings",
            initialise_Dclass(
                **self.settings
            ),
        )
        
        self._initialize_attributes()

@Ddataclass
class Dforager_shard2adgust(DataContainer):

    gathersystem_name: str = dfield("", default=None)
    scattersystems_name: str = dfield("", default=None)
    ad: dict = dfield("", default=None)
    
    def __post_init__(self):
        libsettings_class = globals()["DtoAD"]
        object.__setattr__(
            self,
            "ad",
            initialise_Dclass(
                self.ad,
                libsettings_class,
                    ),
                )
        
        self._initialize_attributes()
        
def generate_docstring(cls: Any) -> Any:
    """
    Generate a docstring for a data class based on its fields.
    """
    if not is_dataclass(cls):
        return cls

    lines = [f"{cls.__name__}:\n"]
    for field in fields(cls):
        field_type = field.type.__name__ if hasattr(field.type,
                                                    "__name__") else str(field.type)
        lines.append(f"{field.name} : {field_type}")
        # Here you could add more detailed documentation for each field if needed
    cls.__doc__ = "\n".join(lines)
    return cls


def update_docstrings(module: Any) -> None:
    """Update docstrings for all data classes in the given module."""
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and is_dataclass(obj):
            obj = generate_docstring(obj)
            print(obj.__doc__)


if (__name__ == "__main__"):
    #...
    # run to generate docstring of data and types
    # update_docstrings(sys.modules[__name__])
    #update_docstrings(sys.modules[__name__])
    import itertools
    d = dict(dead_points = [[[9, 2], [18, 2]],
                            [[9, 1], [18, 1]],
                            [[10,2], [17, 2]]
                            ],
             dead_interpolation = [[[0, 1, 2],
                                   [0, 1, 2]
                                    ],
                                   [[0,10,20],
                                    [0,10,20]]
                                   ]
             )
             
    p = list(itertools.product(*d.values()))
