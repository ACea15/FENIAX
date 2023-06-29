from fem4inas.preprocessor.config import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs

import pathlib

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = [1]
inp.driver.subcases = {'a':4}

inp.simulation.typeof = "serial"
inp.system.name = "static"
inp.simulation.systems
# conf.system.xloads= dict(dead_interpolation=None,
#                          #follower_points=[[1, -1, [1]]],
#                          dead_points=None,
#                          gravity_forces=0,
#                          #follower_interpolation=[[[[0.0, 2000], [0.0, 2000]]]]
#                          )

path2config = pathlib.Path("./config.yaml")
config =  Config(inp)
dump_to_yaml(path2config, config)

config2 = Config.from_file(path2config)

import fem4inas.simulations

# yaml = YAML()
# yaml_dict = yaml.load(pth1)
# for k, v in config2.fem.__dict__.items():
#     if v != getattr(config.fem, k):
#         print(k)

