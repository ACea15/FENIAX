from dataclasses import field
from typing import Sequence


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

def get_component_children(component_name: str,
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
            get_component_children(ci, component_connectivity, chain)
    return chain

def get_component_father(component_connectivity: dict[str:list],
                         chain:list = None):

    component_names = component_connectivity.keys()
    component_father = {ci: [] for ci in component_names}
    for k, v in component_connectivity.items():
        for vi in v:
            component_father[vi].append(k)
    return component_father

def compute_component_nodes(components_range):

    prevnodes = list
    j = 0
    current_component = None
    for i, ci in enumerate(components_range):
        if ci != current_component and i != 0: # change in component
            prevnodes.append(component_nodes[component_father[ci]][-1])
            current_component = ci
            j = 0
        elif i==0:
            prevnodes.append(0)
            j += 1
        else:
            prevnodes.append(component_nodes[current_component][j])
            j += 1
    return prevnodes


def compute_prevnode(components_range: Sequence[str],
                     component_nodes: dict[str:list[int]],
                     component_father: dict[str:int]):

    prevnodes = list
    j = 0
    current_component = None
    for i, ci in enumerate(components_range):
        if ci != current_component and i != 0: # change in component
            prevnodes.append(component_nodes[component_father[ci]][-1])
            current_component = ci
            j = 0
        elif i==0:
            prevnodes.append(0)
            j += 1
        else:
            prevnodes.append(component_nodes[current_component][j])
            j += 1
    return prevnodes
            


comp_conn = dict(c1=['c2','c3'], c2=[],
                 c3=['c4'], c4=[])
chain1 = get_component_children('c1', comp_conn)
chain2 = get_component_children('c2', comp_conn)
chain3 = get_component_children('c3', comp_conn)
chain4 = get_component_children('c4', comp_conn)

