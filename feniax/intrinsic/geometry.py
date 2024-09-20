"""Functions to define the geometry of load-paths."""

import pandas as pd
import jax.numpy as jnp
import numpy as np
from multipledispatch import dispatch
import pathlib
from typing import Sequence, Any
from feniax.utils import flatten_list
from collections.abc import Iterable


def find_fem(folder, Ka_name, Ma_name, grid):
    # TODO: add assertions
    if folder is not None:
        Ka_path = list(pathlib.Path(folder).glob(f"**/*{Ka_name}"))[0].name
        Ma_path = list(pathlib.Path(folder).glob(f"**/*{Ma_name}"))[0].name
        if isinstance(grid, str):
            grid_path = list(pathlib.Path(folder).glob(f"**/*{grid}"))[0].name
        else:
            grid_path = grid
    else:
        Ka_path = Ka_name
        Ma_path = Ma_name
        grid_path = grid
    return Ka_path, Ma_path, grid_path


def list2dict(obj: list | dict):
    """Converts a list into a dictionary

    as in dict = {k: v for k,v in enumerate(list)}

    Parameters
    ----------
    obj : list | dict
        List to be convert, if it is a dictionary do nothing

    Returns
    -------
    dict
    """

    if isinstance(obj, list):
        out = dict()
        for i, v in enumerate(obj):
            out[str(i)] = v
    elif isinstance(obj, dict):
        out = obj
    return out


def build_grid(
    grid: str | jnp.ndarray | pd.DataFrame | None,
    X: jnp.ndarray | None,
    fe_order: list[int] | jnp.ndarray | None,
    fe_order_start: int,
    component_vect: list[str] | None,
    dof_vect,
) -> (pd.DataFrame, jnp.ndarray, np.ndarray, list[str], list[str]):
    if grid is None:
        assert X is not None, "X needs to be provided \
        when no grid file is given"
        assert fe_order is not None, "fe_order needs to be provided \
        when no grid file is given"
        assert component_vect is not None, "component_vect needs to be \
        provided when no grid file is given"
        df_grid = pd.DataFrame(
            dict(
                x1=X[:, 0],
                x2=X[:, 1],
                x3=X[:, 2],
                fe_order=fe_order,
                component=component_vect,
                dof_vect=dof_vect,
            )
        )
    elif isinstance(grid, (str, pathlib.Path)):
        df_grid = pd.read_csv(
            grid,
            comment="#",
            sep=" ",
            names=["x1", "x2", "x3", "fe_order", "component", "dof_vect"],
            dtype={
                "x1": float,
                "x2": float,
                "x3": float,
                "fe_order": int,
                "component": str,
                "dof_vect": str,
            },
        )
        df_grid.dof_vect = df_grid.dof_vect.fillna("012345")
    elif isinstance(grid, jnp.ndarray):
        if grid.shape[1] == 5:
            df_grid = pd.DataFrame(
                dict(
                    x1=grid[:, 0],
                    x2=grid[:, 1],
                    x3=grid[:, 2],
                    fe_order=grid[:, 3],
                    component=grid[:, 4],
                    dof_vect=np.nan,
                )
            )
        elif grid.shape[1] == 6:
            df_grid = pd.DataFrame(
                dict(
                    x1=grid[:, 0],
                    x2=grid[:, 1],
                    x3=grid[:, 2],
                    fe_order=grid[:, 3],
                    component=grid[:, 4],
                    dof_vect=grid[:, 5],
                )
            )
    elif isinstance(grid, pd.DataFrame):
        df_grid = grid
    if not isinstance(X, jnp.ndarray):
        X = jnp.array(df_grid.to_numpy()[:, :3].astype("float"))
    if not isinstance(fe_order, jnp.ndarray):
        fe_order = df_grid.to_numpy()[:, 3:4].astype("int").flatten()
        fe_order -= fe_order_start
    if not isinstance(component_vect, list):
        component_vect = list(df_grid.component)
    if not isinstance(dof_vect, list):
        dof_vect = list(df_grid.dof_vect)
    df_grid.fe_order -= fe_order_start

    return df_grid, X, fe_order, component_vect, dof_vect


def compute_clampedold(
    fe_order: list[int],
) -> (list[int], dict[str:list], dict[str:list], int):
    """Computes the clamped characteristics of the model


    A negative int in fe_order indicates clamped node; -1 will be the
    first clamped node, -2 the second, etc. If only DoF are clamped
    (multibody), the format is as follows: -101011 means
    clamped-free-clamped-free-clamped-clamped in the first clamped
    node. -1010112 is the same but in the second clamped node.

    Parameters
    ----------
    fe_order : list[int]
        Array of integers representing the mapping between a node an
        the index in the stiffness and mass matrices

    Returns
    -------
    (list[int], dict[str: list], dict[str: list], int)
        List of clamped nodes (usually just one); dictionaries
        relating the clamped nodes index to free and clamped DoF; int
        with the total number of clampedDoF, usually 6 for one node
        fully clamped


    """

    clamped_nodes = list()
    freeDoF = dict()
    clampedDoF = dict()
    total_clampedDoF = 0
    constrainedDoF = False
    for i in fe_order:
        if i < 0:
            fe_node = str(abs(i))
            if len(fe_node) < 6:  # clamped node, format = -1 (node 0),
                # -5 (node 4) etc
                clamped_nodes.append(abs(i) - 1)
                freeDoF[clamped_nodes[-1]] = []
                clampedDoF[clamped_nodes[-1]] = list(range(6))
                total_clampedDoF += 6
            elif len(fe_node) > 6:  # format = -1010117 with
                # 101011 being clamped/free DoF and 7 being the multibody node
                constrainedDoF = True
                clamped_nodes.append(int(fe_node[6:]))
                freeDoF[clamped_nodes[-1]] = [
                    i for i, j in enumerate(fe_node[:6]) if j == "0"
                ]
                clampedDoF[clamped_nodes[-1]] = [
                    i for i, j in enumerate(fe_node[:6]) if j == "1"
                ]
                total_clampedDoF += len(clampedDoF[clamped_nodes[-1]])
            else:  # len(fe_node) == 6, format = -101011, multibody node assummed at 0
                constrainedDoF = True
                clamped_nodes.append(0)
                # picks the index of free DoF
                freeDoF[0] = [i for i, j in enumerate(fe_node) if j == "0"]
                clampedDoF[0] = [i for i, j in enumerate(fe_node) if j == "1"]
                total_clampedDoF += len(clampedDoF[0])

    return clamped_nodes, freeDoF, clampedDoF, total_clampedDoF, constrainedDoF


def compute_clamped(
    fe_order: list[int], dof_vect
) -> (list[int], dict[str:list], dict[str:list], int):
    """Computes the clamped characteristics of the model


    A negative int in fe_order indicates clamped node; -1 will be the
    first clamped node, -2 the second, etc. If only DoF are clamped
    (multibody), the format is as follows: -101011 means
    clamped-free-clamped-free-clamped-clamped in the first clamped
    node. -1010112 is the same but in the second clamped node.

    Parameters
    ----------
    fe_order : list[int]
        Array of integers representing the mapping between a node an
        the index in the stiffness and mass matrices

    Returns
    -------
    (list[int], dict[str: list], dict[str: list], int)
        List of clamped nodes (usually just one); dictionaries
        relating the clamped nodes index to free and clamped DoF; int
        with the total number of clampedDoF, usually 6 for one node
        fully clamped


    """

    clamped_nodes = list()
    freeDoF = dict()
    clampedDoF = dict()
    total_clampedDoF = 0
    constrainedDoF = False
    for i, fi in enumerate(fe_order):
        if fi < 0:  # fully clamped node
            clamped_nodes.append(i)
            freeDoF[i] = []
            clampedDoF[i] = [di for di in range(6)]
            total_clampedDoF += 6
        else:
            if len(dof_vect[i]) < 6:
                constrainedDoF = True
                clamped_nodes.append(i)
                freeDoF[i] = [int(di) for di in dof_vect[i]]
                clampedDoF[i] = [int(di) for di in range(6) if di not in freeDoF[i]]
                total_clampedDoF += len(clampedDoF[i])
    return clamped_nodes, freeDoF, clampedDoF, total_clampedDoF, constrainedDoF


def compute_component_father(
    component_connectivity: dict[str:list],
) -> (list[str], dict[str:str]):
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
    dict[str:str]
        Maps the father of each component

    """

    component_names = list(component_connectivity.keys())
    component_father = {ci: None for ci in component_names}
    for k, v in component_connectivity.items():
        if v is not None:
            if isinstance(v, Iterable):
                for vi in v:
                    component_father[str(vi)] = k
            else:
                component_father[str(v)] = k
    return component_names, component_father


@dispatch(list)
def compute_component_nodes(components_range: list[str]) -> dict[str:list]:
    """Links components to their nodes

    Links the nodes (as indexes of DataFrame or list) to the
    compononent they belong to

    Parameters
    ----------
    components_range : list[str]
        List of components

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
    """Links components to their nodes

    Links the nodes (as indexes of DataFrame or list) to the
    compononent they belong to

    Parameters
    ----------
    df : pd.DataFrame
        structuralGrid DataFrame with component column

    Returns
    -------
    dict[str:list]
        Dictionary with component names and the corresponding nodes

    """

    component_nodes = dict()
    components = df.component.unique()
    group = df.groupby("component")
    for ci in components:
        component_nodes[ci] = list(group.get_group(ci).index)
    return component_nodes


def compute_prevnode(
    components_range: Sequence[str],
    component_nodes: dict[str : list[int]],
    component_father: dict[str:int],
) -> list[int]:
    """Computes the previous node index to each node

    for a simple graph like 0--1--3--2--4 the output will be [None, 0,
    3, 1, 2]

    Parameters
    ----------
    components_range : Sequence[str]
        List of components corresponding the input nodes belong to.
    component_nodes : dict[str:list[int]]
        Map between components and the nodes in each of them
    component_father : dict[str:int]
        Maps each component to their father component

    Returns
    -------
    list[int]
        The list with each preceding node assuming an outwards flow
        from the very first node


    """

    prevnodes = list()
    j = 0
    current_component = None
    for i, ci in enumerate(components_range):
        if i == 0:
            prevnodes.append(0)
            # j += 1
            current_component = ci
        elif ci != current_component:  # change in component
            if component_father[ci] is None:  # component starting at first node
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


def compute_component_children(
    component_name: str,
    component_connectivity: dict[str : list[str | int]],
    chain: list = None,
) -> list[str]:
    """Computes the children components on any given component

    Parameters
    ----------
    component_name : str
        The component on which we want to calculate the components
        that derived from it, again assuming an outward flow from the
        very first node.
    component_connectivity : dict[str:list[str | int]]
        The connectivity dictionary that links each component to its
        "children"
    chain : list
        variable for recursive function, ultimately the output

    Returns
    -------
    list[str]
        All the children on the input component

    """

    if chain is None:
        chain = list()
    component_name = str(component_name)  # in case components are defined with numbers
    children_components = component_connectivity[component_name]
    if children_components is None or len(children_components) == 0:
        pass
    else:
        chain += [str(cci) for cci in children_components]
        for ci in children_components:
            compute_component_children(ci, component_connectivity, chain)
    return chain


def compute_component_chain(
    component_names: list[str], component_connectivity: dict[str : list[str | int]]
) -> dict[str : list[str]]:
    """Computes the dictionary that maps all the components to their corresponding children

    Parameters
    ----------
    component_names : list[str]
        List with the components in the model
    component_connectivity : dict[str:list[str | int]]
        The connectivity dictionary

    Returns
    -------
    dict[str:list[str]]
       Maps component to all its children
    """

    component_chain = {
        k: compute_component_children(k, component_connectivity)
        for k in component_names
    }
    return component_chain


def compute_Maverage(prevnodes: Sequence[int], num_nodes: int) -> jnp.ndarray:
    """Calculates the matrix that averages between adjacent nodes

    Parameters
    ----------
    prevnodes : Sequence[int]
        Array with the previous node assuming outwards flow from the
        first one.
    num_nodes : int
        Total number of nodes

    Returns
    -------
    jnp.ndarray
        Matrix where each column represents one node and the
        components in that column are 0.5 and 0.5 for the
        corresponding node and its precedent

    """

    M = np.eye(num_nodes)
    M[0, 0] = 0.0  # first node should be made 0 since we have Nn - 1 elements:
    # given a structure like *--o--*--o--*, tensors are given in '*', but
    # only the quantity at 'o' is of interest
    for i in range(1, num_nodes):
        M[prevnodes[i], i] = 1
    M *= 0.5
    return jnp.array(M)


def compute_Mdiff(prevnodes: Sequence[int], num_nodes: int) -> jnp.ndarray:
    """Calculates the matrix that subtracts between adjacent nodes

    Parameters
    ----------
    prevnodes : Sequence[int]
        Array with the previous node assuming outwards flow from the
        first one.
    num_nodes : int
        Total number of nodes

    Returns
    -------
    jnp.ndarray
        Matrix where each column represents one node and the
        components in that column are 1 and -1 for the
        corresponding node and its precedent

    """

    M = np.eye(num_nodes)
    M[0, 0] = 0.0
    for i in range(1, num_nodes):
        M[prevnodes[i], i] = -1
    return jnp.array(M)


def compute_Mfe_order(
    fe_order: np.ndarray,
    clamped_nodes: list[int],
    freeDoF: dict[str:list],
    total_clampedDoF: int,
    component_nodes: dict[str:list],
    component_chain: dict[str:list],
    num_nodes: int,
) -> jnp.ndarray:
    """Finds the order to swap quantities from the FE model to the input nodes

    for instance to map eigenvectors to the geometry in the given
    grid.

    Parameters
    ----------
    fe_order : np.ndarray
        Array of integers representing the mapping between a node an
        the index in the stiffness and mass matrices
    clamped_nodes : list[int]
       List of clamped nodes (usually just one); dictionaries
       relating the clamped nodes index to free and clamped DoF.
    freeDoF : dict[str: list]
       free DoF for each clamped node
    total_clampedDoF : int
        int with the total number of clampedDoF
    component_nodes : dict[str:list]
        Dictionary with component names and the corresponding nodes
    component_chain : dict[str:list]
        Dictionary mapping component to all of its children
    num_nodes : int

    Returns
    -------
    jnp.ndarray
        Matrix where each column represents one node and the
        components in that column are 1s for the DoF

    """

    free_dofs = 0
    costrained_nodes = set()
    M = np.zeros((6 * num_nodes, 6 * num_nodes - total_clampedDoF))
    M2 = np.zeros((6 * num_nodes, 6 * num_nodes))
    for fi, node_index in zip(np.sort(fe_order), np.argsort(fe_order)):
        if node_index in clamped_nodes:
            for di in range(6):
                if di in freeDoF[node_index]:
                    costrained_nodes.add(node_index)
                    M[
                        6 * node_index + di,
                        6 * (fi - (len(costrained_nodes) - 1)) + free_dofs,
                    ] = 1.0
                    M2[6 * node_index + di, 6 * fi + di] = 1.0
                    free_dofs += 1
                else:
                    continue
            # num_costrained_nodes += 1
        else:
            for di in range(6):
                # M[6 * node_index + di, 6 * (fi - num_clamped_nodes) + free_dofs + di] = 1.
                M[
                    6 * node_index + di,
                    6 * (fi - len(costrained_nodes)) + free_dofs + di,
                ] = 1.0
                M2[6 * node_index + di, 6 * fi + di] = 1.0

    return jnp.array(M), jnp.array(M2)


def compute_Mconstrained(Ka, Ma, fe_order, clamped_nodes, clampedDoF):
    Ka2 = Ka.copy()
    Ma2 = Ma.copy()
    for cni in clamped_nodes:
        for di in clampedDoF[cni]:
            Ka2 = jnp.insert(Ka2, 6 * fe_order[cni] + di, 0.0, axis=0)
            Ka2 = jnp.insert(Ka2, 6 * fe_order[cni] + di, 0.0, axis=1)
            Ma2 = jnp.insert(Ma2, 6 * fe_order[cni] + di, 0.0, axis=0)
            Ma2 = jnp.insert(Ma2, 6 * fe_order[cni] + di, 0.0, axis=1)

    # Ka2 = jnp.insert(Ka, 6, 0., axis=0)
    # Ka2 = jnp.insert(Ka2, 7, 0., axis=0)
    # Ka2 = jnp.insert(Ka2, 8, 0., axis=0)
    # Ka2 = jnp.insert(Ka2, 6, 0., axis=1)
    # Ka2 = jnp.insert(Ka2, 7, 0., axis=1)
    # Ka2 = jnp.insert(Ka2, 8, 0., axis=1)

    # Ma2 = jnp.insert(Ma, 6, 0., axis=0)
    # Ma2 = jnp.insert(Ma2, 7, 0., axis=0)
    # Ma2 = jnp.insert(Ma2, 8, 0., axis=0)
    # Ma2 = jnp.insert(Ma2, 6, 0., axis=1)
    # Ma2 = jnp.insert(Ma2, 7, 0., axis=1)
    # Ma2 = jnp.insert(Ma2, 8, 0., axis=1)

    return Ka2, Ma2


def compute_Mloadpaths(
    components_range: list[str],
    component_nodes: dict[str : list[int]],
    component_chain: dict[str : list[str]],
    num_nodes: int,
) -> jnp.ndarray:
    """Finds the load paths for the internal forces at each node.

    Parameters
    ----------
    components_range : list[str]
       List of components
    component_nodes : dict[str:list[int]]
        Dictionary with component names and the corresponding nodes
    component_chain : dict[str: list[str]]
        Dictionary mapping component to all of its children
    num_nodes : int
        Total number of nodes

    Returns
    -------
    jnp.ndarray
        Matrix where each column represents one node and the
        components in that column are 1s for the nodes one
        needs to transverse to recover the loads at that node.
        Note this only works for structures without loops or
        multiple fully clamped nodes, i.e. determinate structures.

    """

    M = np.eye(num_nodes)
    M[:, 0] = 1.0
    current_component = components_range[0]
    j = 1
    for i in range(1, num_nodes):
        ci = components_range[i]
        if ci != current_component:
            j = 0
            current_component = ci
        ci_nodes = component_nodes[ci]
        ci_children = component_chain[ci]
        ci_children_nodes = flatten_list([component_nodes[k] for k in ci_children])
        M[ci_children_nodes, i] = 1.0
        M[ci_nodes[j:], i] = 1
        j += 1
    return jnp.array(M)


def convert_components(
    component_names: list[str],
    component_nodes: dict[str : list[int]],
    component_father: dict[str:int],
):
    num_components = len(component_names)
    names_new = tuple(range(num_components))
    nodes_new = tuple([component_nodes[ci] for ci in component_names])
    father_new = [[] for i in range(num_components)]
    for k, v in component_father.items():
        ind = component_names.index(k)
        if v is not None:
            ind_father = component_names.index(v)
        else:
            ind_father = 0
        father_new[ind] = ind_father
    father_new = tuple(father_new)
    return names_new, nodes_new, father_new
