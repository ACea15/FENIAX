"""Functions to define the geometry of load-paths."""
import pandas as pd
import jax.numpy as jnp
import numpy as np
from multipledispatch import dispatch
import pathlib
from typing import Sequence, Any
from fem4inas.utils import flatten_list

def list2dict(obj: list | dict):

    if isinstance(obj, list):
        out = dict()
        for i, v in enumerate(obj):
            out[str(i)] = v
    elif isinstance(obj, dict):
        out = obj
    return out
        
def build_grid(grid: str | jnp.ndarray | pd.DataFrame | None,
               X: jnp.ndarray | None,
               fe_order: list[int] | jnp.ndarray | None,
               component_vect: list[str] | None) -> (pd.DataFrame,
                                                     jnp.ndarray,
                                                     jnp.ndarray,
                                                     list[str]):
    if grid is None:
        assert X is not None, "X needs to be provided \
        when no grid file is given"
        assert fe_order is not None, "fe_order needs to be provided \
        when no grid file is given"
        assert component_vect is not None, "component_vect needs to be \
        provided when no grid file is given"
        df_grid = pd.DataFrame(dict(x1=X[:, 0], x2=X[:, 1], x3=X[:, 2],
                                    fe_order=fe_order, component=component_vect))
    elif isinstance(grid, str):
        path = pathlib.Path(grid)
        df_grid = pd.read_csv(path, comment="#", sep=" ",
                              names=['x1', 'x2', 'x3', 'fe_order', 'component'])

    elif isinstance(grid, jnp.ndarray):
        df_grid = pd.DataFrame(dict(x1=grid[:, 0], x2=grid[:, 1], x3=grid[:, 2],
                                    fe_order=grid[:, 3], component=grid[:, 4]))
    
    elif isinstance(grid, pd.DataFrame):
        df_grid = grid

    if not isinstance(X, jnp.ndarray):
        X = jnp.array(df_grid.to_numpy()[:,:3].astype('float'))
    if not isinstance(fe_order, jnp.ndarray):
        fe_order = jnp.array(df_grid.to_numpy()[:,:3].astype('int'))
    if not isinstance(component_vect, list):
        component_vect = list(df_grid.component.astype('str'))        
    return df_grid, X, fe_order, component_vect

def compute_clamped(fe_order: list[int]) -> (list[int], dict[str: list]):
    clamped_nodes = list()
    freeDoF = dict()
    for i in fe_order:
        if i < 0:
            clamped_nodes.append(i)
    for ni in clamped_nodes:
        fe_ni = str(abs(fe_order[ni]))
        if len(fe_ni) == 6: # 100100 format with 1 being DoF clamped
            # picks the index of free DoF
            freeDoF[ni] = [i for i,j in enumerate(fe_ni) if j =='0']
        else:
            freeDoF[ni] = []
    return clamped_nodes, freeDoF

def compute_component_father(component_connectivity:
                             dict[str:list]) -> (list[str], dict[str:list]):
    """Calculates the father component of each component

    Assuming an outwards flow from the first node, every path in the
    graph is transverse in a particular direction, which defines which
    components follows another

    Parameters
    ----------
    component_connectivity : dict[str:list]
        Connectivity input that sets the components attached to each
        component with the logic above

    Returns
    -------
    dict[str:list]
        Maps the father of each component

    """

    component_names = component_connectivity.keys()
    component_father = {ci: None for ci in component_names}
    for k, v in component_connectivity.items():
        if v is not None:
            for vi in v:
                component_father[vi] = k
    return component_names, component_father

@dispatch(list)
def compute_component_nodes(components_range: list[str]) -> dict[str:list]:
    """Links components to their nodes 

    Links the nodes (as indexes of DataFrame or list) to the
    compononent they belong to

    Parameters
    ----------
    components_range : list[str]
        Component list

    Returns
    -------
    dict[str:list]
        Dictionary with component names and the corresponding nodes

    """

    component_nodes = dict()
    for i, ci in enumerate(components_range):
        if ci not in component_nodes.keys():
            component_nodes[ci] = []
        component_nodes[ci].append(i)
    return component_nodes

@dispatch(pd.DataFrame)
def compute_component_nodes(df: pd.DataFrame) -> dict[str:list]:

    component_nodes = dict()
    components = df.component.unique()
    group = df.groupby('component')
    for ci in components:
        component_nodes[ci] = list(group.get_group(ci).index)
    return component_nodes

def compute_prevnode(components_range: Sequence[str],
                     component_nodes: dict[str:list[int]],
                     component_father: dict[str:int]) -> list[int]:

    prevnodes = list()
    j = 0
    current_component = None
    for i, ci in enumerate(components_range):
        if i==0:
            prevnodes.append(0)
            #j += 1
            current_component = ci
        elif ci != current_component: # change in component
            if component_father[ci] is None: # component starting at first node
                prevnodes.append(0)
                current_component = ci
                j = 0
            else:
                prevnodes.append(component_nodes[component_father[ci]][-1])
                current_component = ci
                j = 0
        else:
            prevnodes.append(component_nodes[current_component][j])
            j += 1
    return prevnodes

def compute_component_children(component_name: str,
                               component_connectivity: dict[str:list],
                               chain:list = None):
    
    if chain is None:
        chain = list()
    children_components = component_connectivity[component_name]
    if children_components is None or len(children_components) == 0:
        pass
    else:
        chain += children_components
        for ci in children_components:
            compute_component_children(ci, component_connectivity, chain)
    return chain

def compute_component_chain(component_names: list[str],
                            component_connectivity: dict[str:list]):

    component_chain = {k: compute_component_children(k, component_connectivity)
                       for k in component_names}
    return component_chain

def compute_Maverage(prevnodes: Sequence[int], num_nodes: int) -> jnp.ndarray:

    M = np.eye(num_nodes)
    for i in range(1, num_nodes):
        M[prevnodes[i], i] = 1
    M *= 0.5
    return jnp.array(M)

def compute_Mdiff(prevnodes: Sequence[int], num_nodes: int) -> jnp.ndarray:

    M = np.eye(num_nodes)
    for i in range(1, num_nodes):
        M[prevnodes[i], i] = -1
    return jnp.array(M)

def compute_Mfe_order(fe_order,
                      clamped_nodes,
                      freeDoF,
                      component_nodes,
                      component_chain,
                      num_nodes) -> jnp.ndarray:

    M = np.zeros((6 * num_nodes, 6 * num_nodes))
    for i in range(6 * num_nodes):
        if i in clamped_nodes:
            fe_dof = [(6 * (fe_order[i] + 1) + j) for j in freeDoF[i]]
        else:
            fe_dof = range(6 * fe_order[i], 6 * fe_order[i] + 6)
        if len(fe_dof) > 0:
            M[i, fe_dof] = 1.

    return jnp.array(M)

def compute_Mloadpaths(components_range,
                       component_nodes,
                       component_chain,
                       num_nodes) -> jnp.ndarray:

    M = np.eye(num_nodes)
    M[:, 0] = 1.
    current_component = components_range[0]
    j = 1
    for i in range(1, num_nodes):
        ci = components_range[i]
        if ci != current_component:
            j = 0
            current_component = ci
        ci_nodes = component_nodes[ci]
        ci_children = component_chain[ci]
        ci_children_nodes = flatten_list([component_nodes[k] for
                                          k in ci_children])
        M[ci_children_nodes, i] = 1.
        M[ci_nodes[j:], i] = 1
        j += 1
    return jnp.array(M)
