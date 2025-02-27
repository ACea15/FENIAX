__all__ = ["factory"]
import os
import importlib
from feniax.simulations.simulation import __SIMULATION_DICT__, Simulation
from typing import Type


def factory(name) -> Type[Simulation]:
    return __SIMULATION_DICT__[name]


for file in os.listdir(os.path.dirname(__file__)):
    # print(file)
    if file.endswith(".py") and not file.startswith("_") and not file.startswith("."):
        module_name = file[: file.find(".py")]
        try:
            module = importlib.import_module(f"feniax.simulations.{module_name}")
        except ModuleNotFoundError:
            print(file)
