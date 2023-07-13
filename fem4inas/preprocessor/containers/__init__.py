import importlib

__all__ = ["intrinsicmodal", "intrinsicsol"]

def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
