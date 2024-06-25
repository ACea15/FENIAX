from dataclasses import field
from typing import Sequence, Any
import pandas as pd
import numpy as np
import jax.numpy as jnp
from fem4inas.utils import flatten_list
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
import pathlib
import argparse

def dfield(description, **kwargs):

    options = kwargs.pop('options', None)
    default = kwargs.pop('default', 'not_defined')
    init = kwargs.pop('init', True)
    yaml_save = kwargs.pop('yaml_save', True)
    metadata={"description": description,
              "options": options,
              "yaml_save": yaml_save}
    if default != 'not_defined':
        if default is None:
            return field(
                default=default,
                metadata=metadata,
                init=init,
                **kwargs
            )
        elif isinstance(default, (str, int, bool, float, tuple)):
            return field(
                default=default, metadata={"description": description,
                                           "options": options,
                                           "yaml_save": yaml_save},
                init=init, **kwargs
            )
        else:
            return field(
                default_factory=(lambda: default),
                metadata=metadata,
                init=init, **kwargs
            )
    else:
        return field(
            metadata=metadata,
            init=init, **kwargs
        )

def update_dict(d, u):
    result = d.copy()  # Start with a shallow copy of the original dictionary
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = update_dict(result[k], v)
        else:
            result[k] = v
    return result

def initialise_Dclass(data, Dclass, **kwargs):

    if data is None:
        return Dclass()
    elif isinstance(data, dict):
        return Dclass(**(update_dict(data,kwargs)))
    elif isinstance(data, Dclass):
        return data
    else:
        raise TypeError("Wrong input type")

def dump_inputs(data: dict[str:list[Any, str]],
                indent:int=0,
                with_comments:bool=True):

    #if ind == 0:
    data = CommentedMap(data)
    for k, v in data.items():
        if isinstance(v, dict):
            data[k] = dump_inputs(v, indent=indent+1,
                                  with_comments=with_comments)
        else:
            data[k] = v[0]
            #data.yaml_add_eol_comment(v[1], k)
            if with_comments:
                data.yaml_set_comment_before_after_key(k,
                                                       before=v[1],
                                                       indent=2*indent)
    return data

def dump_yaml(file_out: str | pathlib.Path,
              data_in: dict[str:list[Any, str]],
              with_comments=True):

    yaml = YAML()
    file_out = pathlib.Path(file_out)
    file_out.parent.mkdir(parents=True, exist_ok=True)
    data = dump_inputs(data_in, with_comments=with_comments)
    with open(file_out, "w") as f:
        yaml.dump(data, f)


def load_jnp(path):

    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)
    assert path.is_file(), f"{str(path)} is not a file"
    A = jnp.load(path)
    return A

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
