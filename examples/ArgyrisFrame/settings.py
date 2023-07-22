import pathlib
import pdb
import sys

import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = [[1], []]
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 10
inp.fem.fe_order_start = 1
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"

config =  configuration.Config(inp)

# for k, v in config._data_dict['fem'].items():
#     print(f"{k}:{type(v[0])}")


path2config = pathlib.Path("./config.yaml")
#config =  configuration.Config(inp)
#configuration.dump_to_yaml(path2config, config)

fem4inas.fem4inas_main.main(input_obj=config)
# config2 = configuration.Config.from_file(path2config)

# yaml = YAML()
# yaml_dict = yaml.load(pth1)
# for k, v in config2.fem.__dict__.items():
#     if v != getattr(config.fem, k):
#         print(k)

# import pathlib
# import re
# p1 = list(pathlib.Path("./FEM/").glob("*Ka*"))[0]
# for pi in p1:
#     print(list(re.match("*Ka*", pi)))
    
