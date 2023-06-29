__all__ = ["factory_method"]
import os
import importlib
from fem4inas.simulations.simulation import Simulation, __SIMULATION_DICT__

def factory_method(name):
    return __SIMULATION_DICT__[name]

for file in os.listdir(os.path.dirname(__file__)):
    #print(file)
    if file.endswith('.py') and not file.startswith('_') and not file.startswith('.'):
        module_name = file[:file.find('.py')]
        try:
            module = importlib.import_module(f"fem4inas.simulations.{module_name}")
        except ModuleNotFoundError:
            print(file)
