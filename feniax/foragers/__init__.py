__all__ = ["factory"]
import os
import importlib
from feniax.foragers.forager import __FORAGER_DICT__


def factory(name):
    return __FORAGER_DICT__[name]


for file in os.listdir(os.path.dirname(__file__)):
    # print(file)
    if file.endswith(".py") and not file.startswith("_") and not file.startswith("."):
        module_name = file[: file.find(".py")]
        # try:
        module = importlib.import_module(f"feniax.foragers.{module_name}")
        # except ModuleNotFoundError:
        #    print(file)
