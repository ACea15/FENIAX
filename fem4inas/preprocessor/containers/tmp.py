from dataclasses import dataclass
from typing import Sequence
import pathlib
import jax.numpy as jnp
import pathlib
import pandas as pd
from fem4inas.preprocessor.utils import dfield, initialise_Dclass
from fem4inas.preprocessor.inputs import Inputs
from fem4inas.preprocessor.config import Config

df = pd.read_csv("/home/ac5015/programs/FEM4INAS/examples/ArgyrisBeam/FEM/structuralgrid",
                 comment="#", sep=" ", names=['x1', 'x2', 'x3', 'fe_order', 'component'])
#df.to_numpy()[:,:3]

c1 = Config()
c1.engine = "intrinsicmodal"
c1.fem.folder = 'ff'
c1.fem.connectivity = [[]]
config = Inputs(c1)
