from dataclasses import field
from typing import Sequence
import pandas as pd
from multipledispatch import dispatch
import numpy as np
import jax.numpy as jnp
from fem4inas.utils import flatten_list

def dfield(description, **kwargs):

    options = kwargs.pop('options', None)
    default = kwargs.pop('default', 'not_defined')
    init = kwargs.pop('init', True)
    if default != 'not_defined':
        if default is None:
            return field(
                default=default, metadata={"description": description, "options": options},
                init=init, **kwargs
            )
        elif isinstance(default, (str, int, bool, float, tuple)):
            return field(
                default=default, metadata={"description": description, "options": options},
                init=init, **kwargs
            )
        else:
            return field(
                default_factory=(lambda: default),
                metadata={"description": description, "options": options},
                init=init, **kwargs
            )
    else:
        return field(
            metadata={"description": description, "options": options},
            init=init, **kwargs
        )

def initialise_Dclass(data, Dclass):

    if data is None:
        return Dclass()
    elif isinstance(data, dict):
        return Dclass(**data)
    elif isinstance(data, Dclass):
        return data
    else:
        raise TypeError("Wrong input type")

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

def compute_component_father(component_connectivity: dict[str:list]) -> dict[str:list]:

    component_names = component_connectivity.keys()
    component_father = {ci: None for ci in component_names}
    for k, v in component_connectivity.items():
        if v is not None:
            for vi in v:
                component_father[vi] = k
    return component_father

@dispatch(list)
def compute_component_nodes(components_range: list) -> dict[str:list]:

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

def compute_dof2insert(free_dof):

    j=0
    dof2insert = []
    for i in range(6):
        if i in free_dof:
            j +=1
        else:
            dof2insert.append(j)
    return dof2insert

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

def compute_Mloadpaths(prevnodes: Sequence[int], num_nodes: int) -> jnp.ndarray:

    M = np.eye(num_nodes)
    M[:,0] = 1.
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
        M[ci_children_nodes, i] = 1.
        M[ci_nodes[j:], i] = 1
        j += 1
    return jnp.array(M)


if __name__ == "__main__":
    comp_conn = dict(c1=['c2','c3', 'c5'], c2=None,
                     c3=['c4'], c4=[], c5=None)
    chain1 = compute_component_children('c1', comp_conn)
    chain2 = compute_component_children('c2', comp_conn)
    chain3 = compute_component_children('c3', comp_conn)
    chain4 = compute_component_children('c4', comp_conn)
    chain5 = compute_component_children('c5', comp_conn)

    components_range = ['c1', 'c1', 'c2', 'c2', 'c4', 'c4', 'c4', 'c3', 'c5', 'c5']
    component_nodes = compute_component_nodes(components_range)
    component_father = compute_component_father(comp_conn)
    prevnodes = compute_prevnode(components_range, component_nodes, component_father)

    comp_conn2 = dict(c1=['c2','c3', 'c5'], c2=None,
                      c3=['c4'], c4=[], c5=None, c6=[])
    chain12 = compute_component_children('c1', comp_conn)
    chain22 = compute_component_children('c2', comp_conn)
    chain32 = compute_component_children('c3', comp_conn)
    chain42 = compute_component_children('c4', comp_conn)
    chain52 = compute_component_children('c5', comp_conn)

    components_range2 = ['c1', 'c1', 'c2', 'c2', 'c4', 'c4', 'c4', 'c3', 'c5', 'c5', 'c6']
    component_nodes2 = compute_component_nodes(components_range2)
    component_father2 = compute_component_father(comp_conn2)
    prevnodes2 = compute_prevnode(components_range2, component_nodes2, component_father2)
