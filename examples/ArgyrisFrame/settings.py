from fem4inas.preprocessor.config import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import pdb
import sys

import pathlib

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = [[1], []]
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 10
config =  Config(inp)

path2config = pathlib.Path("./config.yaml")
config =  Config(inp)
dump_to_yaml(path2config, config)

config2 = Config.from_file(path2config)

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
    
