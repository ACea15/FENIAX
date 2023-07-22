from fem4inas.preprocessor.inputs import Config, dump_to_yaml
from fem4inas.preprocessor.configuration import dict2object, Config

conf = Config()
conf.fem.connectivity  = [[]]
conf.engine = "intrinsicmodal"
config =  Config(conf)
