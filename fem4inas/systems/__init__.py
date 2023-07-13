__all__ = ["factory"]
import os
import importlib
from fem4inas.systems.system import System, __SYSTEM_DICT__

def factory(name):
    return __SYSTEM_DICT__[name]

for file in os.listdir(os.path.dirname(__file__)):
    #print(file)
    if file.endswith('.py') and not file.startswith('_') and not file.startswith('.'):
        module_name = file[:file.find('.py')]
        try:
            module = importlib.import_module(f"fem4inas.systems.{module_name}")
        except ModuleNotFoundError:
            print(file)
