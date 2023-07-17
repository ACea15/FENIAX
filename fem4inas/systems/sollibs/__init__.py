import importlib

def factory(module: str, name:str):

    library = importlib.import_module(f".{module}",
                                      __name__)
    function = getattr(library, name)
    states_puller  = getattr(library, "pul_" + name)
    return function
