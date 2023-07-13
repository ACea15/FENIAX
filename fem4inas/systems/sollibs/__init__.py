import importlib

def factory(module: str, name:str):

    library = importlib.import_module(f".{module}",
                                      __name__)
    function = getattr(library, name)
    return function
