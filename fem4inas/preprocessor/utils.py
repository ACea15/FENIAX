from dataclasses import field


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

def component_child(component_name: str,
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
            component_child(ci, component_connectivity, chain)
    return chain

comp_conn = dict(c1=['c2','c3'], c2=[],
                 c3=['c4'], c4=[])
chain1 = component_child('c1', comp_conn)
chain2 = component_child('c2', comp_conn)
chain3 = component_child('c3', comp_conn)
chain4 = component_child('c4', comp_conn)

