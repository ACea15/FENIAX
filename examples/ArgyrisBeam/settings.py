from fem4inas.preprocessor.inputs import Inputs, dump_to_yaml
from fem4inas.preprocessor.config import dict2object, Config

conf = Config()
conf.fem.connectivity  = [[]]
conf.engine = "intrinsicmodal"
config =  Inputs(conf)
