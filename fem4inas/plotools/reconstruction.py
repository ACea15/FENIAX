import fem4inas.plotools.grid as grid
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import pandas as pd
import importlib
import numpy as np
import jax.numpy as jnp
from fem4inas.preprocessor import solution
import fem4inas.plotools.interpolation as interpolation
import fem4inas.plotools.nastranvtk.bdfdef as bdfdef
import fem4inas.unastran.op2reader as op2reader

def rbf_based(nastran_bdf, X, time, ra, Rab, R0ab,
              vtkpath=None, plot_timeinterval=None,
              tolerance=1e-3, size_cards=8):

    bdf_model = BDF(debug=True)
    bdf_model.read_bdf(nastran_bdf, punch=False)
    # X = config.fem.X
    num_time = len(time)
    numnodes = len(X)
    gridmodel = grid.RBE3Model(bdf_model, X, tolerance)
    bdf_def = bdfdef.DefBdf(nastran_bdf)
    X3d = bdf_def.get_nodes()
    numnodes3d = len(X3d)
    uintrinsic = np.zeros((num_time, numnodes3d, 3))
    rintrinsic = np.zeros((num_time, numnodes3d, 3))
    if vtkpath is not None:
        bdf_def.save_vtkpath(vtkpath)
        bdf_def.plot_vtk("ref", size_cards)
    for i, ti in enumerate(time):
        rai = ra[i]
        Rabi = Rab[i]
        gridmodel.set_solution(rai, Rabi, R0ab)
        disp, coord = interpolation.compute(gridmodel.model1_coord,
                                            gridmodel.model1x_coord,
                                            X3d)
        uintrinsic[i] = disp
        rintrinsic[i] = coord
        if plot_timeinterval is not None:
            if i % plot_timeinterval == 0:
                tstep = i // plot_timeinterval
                #coord = nodesX2 + unastran[0,i,:,:3]
                bdf_def.update_bdf(coord, bdf_def.sorted_nodeids)
                bdf_def.plot_vtk(tstep, size_cards)
    return rintrinsic, uintrinsic

def nastran_based():
    ...
