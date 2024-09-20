import numpy as np
import pyNastran
from pyNastran.op2.op2 import OP2
import pyNastran.op4.op4 as op4
from pyNastran.bdf.bdf import BDF
import pathlib


def write_op4modes(
    file_name: pathlib.Path | str,
    num_modes: int,
    op4_name: None,
    matrix_name="PHG",
    return_modes=False,
):
    file_name = pathlib.Path(file_name)
    op2 = OP2()
    op2.read_op2(file_name.with_suffix(".op2"))
    eig1 = op2.eigenvectors[1]
    modesdata = eig1.data
    eigsdata = eig1.eigns
    modes = modesdata[:num_modes]
    op2_nummodes, op2.numnodes, op2.numdim = modes.shape
    modes_reshape = modes.reshape((op2_nummodes, op2.numnodes * op2.numdim)).T
    op4_data = op4.OP4()
    if op4_name is None:
        op4_name = str(file_name.with_suffix(".op4"))
    else:
        op4_name = str(pathlib.Path(op4_name).with_suffix(".op4"))
    op4_data.write_op4(op4_name, {matrix_name: (2, modes_reshape)}, is_binary=False)
    if return_modes:
        return eigsdata, modesdata
