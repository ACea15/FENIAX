from dataclasses import dataclass
from typing import Sequence
import pathlib
import jax.numpy as jnp
import pathlib
import pandas as pd

from fem4inas.preprocessor.utils import dfield, initialise_Dclass, dump_inputs
from fem4inas.preprocessor.inputs import Config, dump_to_yaml
from fem4inas.preprocessor.configuration import Config
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

yaml = YAML()

df = pd.read_csv("/home/ac5015/programs/FEM4INAS/examples/ArgyrisBeam/FEM/structuralgrid",
                 comment="#", sep=" ", names=['x1', 'x2', 'x3', 'fe_order', 'component'])
#df.to_numpy()[:,:3]

c1 = Config()
c1.engine = "intrinsicmodal"
c1.fem.folder = 'ff'
c1.fem.connectivity = [[]]
c1.fem.Ka = 'hello'
config = Config(c1)

#data_yaml = dump_inputs({'a': [5, 'rr'], 'b':{'b1':[1," "]}})

data_yaml = dump_inputs(config._data_dict)

with open("output2.yaml", "w") as f:
    yaml.dump(data_yaml, f)


import importlib
data_container = importlib.import_module("data_container", "fem4inas.preprocessor.containers")
intrinsicmodal = importlib.import_module("intrinsicmodal", "fem4inas.preprocessor.containers")

#dfem = intrinsicmodal.Dfem(**c1.fem)
print(isinstance(config.fem, data_container.DataContainer))
print(isinstance(config.fem, intrinsicmodal.Dfem))

#data = CommentedMap(c1)

# c1d = {k:v for k,v in c1.items()}
# c1d = {'engine': 'intrinsicmodal', 'fem': {'folder': 'ff'}}
# c1dict = deepcopy(c1d)

d1 = CommentedMap({k: v for k, v in config.__dict__.items() if k[0] != "_"})
d1['fem'] = CommentedMap({k: v for k, v in d1['fem'].__dict__.items() if k[0] != "_"})
# Create a Python dictionary
#c1dict = {'fem': {'folder': 'ff', 'connectivity': [[]]}}
# Convert the dictionary to a CommentedMap object
data = CommentedMap(d1)

# Add a comment to a key
data.yaml_add_eol_comment("This is a comment engine ", "fem")
#data.yaml_add_eol_comment("This is a comment engine ", "fem[Ka]")
data['fem'].yaml_add_eol_comment("This is a comment engine ", "Ka")
data.yaml_set_comment_before_after_key("engine", before="comment_here")
# Write the data as YAML
with open("output.yaml", "w") as f:
    yaml.dump(data, f)
