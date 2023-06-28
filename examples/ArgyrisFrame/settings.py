from fem4inas.preprocessor.inputs import Inputs, dump_to_yaml
from fem4inas.preprocessor.config import Config

conf = Config()

conf.engine = "intrinsicmodal"
conf.fem.connectivity = [1]
conf.driver.subcases = {'a':4}
#conf.system.name = "static"
# conf.system.xloads= dict(dead_interpolation=None,
#                          #follower_points=[[1, -1, [1]]],
#                          dead_points=None,
#                          gravity_forces=0,
#                          #follower_interpolation=[[[[0.0, 2000], [0.0, 2000]]]]
#                          )

config =  Inputs(conf)
dump_to_yaml("./config.yaml", config)
