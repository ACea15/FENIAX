__all__ = ["factory"]
import os
import importlib
from feniax.systems.system import __SYSTEM_DICT__, System
from typing import Type


def factory(name) -> Type[System]:
    return __SYSTEM_DICT__[name]


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_") and not file.startswith("."):
        module_name = file[: file.find(".py")]
        module = importlib.import_module(f"feniax.systems.{module_name}")
        # try:
        #     module = importlib.import_module(f"fem4inas.systems.{module_name}")
        # except ModuleNotFoundError:
        #     print(file)
