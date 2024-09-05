import feniax.plotools.grid as grid
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import pandas as pd
import importlib
import numpy as np
import jax.numpy as jnp
from feniax.preprocessor import solution
import feniax.plotools.interpolation as interpolation
import feniax.plotools.nastranvtk.bdfdef as bdfdef
import feniax.unastran.op2reader as op2reader


def rbf_based(
    nastran_bdf,
    X,
    time,
    ra,
    Rab,
    R0ab,
    vtkpath=None,
    plot_timeinterval=None,
    plot_ref=True,
    tolerance=1e-3,
    size_cards=8,
    rbe3s_full=True,
    ra_movie=None,
):
    bdf_model = BDF(debug=True)
    bdf_model.read_bdf(nastran_bdf, punch=False)
    # X = config.fem.X
    num_time = len(time)
    numnodes = len(X)
    gridmodel = grid.RBE3Model(bdf_model, X, tolerance, rbe3s_full)
    bdf_def = bdfdef.DefBdf(nastran_bdf)
    X3d = bdf_def.get_nodes()
    numnodes3d = len(X3d)
    uintrinsic = np.zeros((num_time, numnodes3d, 3))
    rintrinsic = np.zeros((num_time, numnodes3d, 3))
    if vtkpath is not None:
        bdf_def.save_vtkpath(vtkpath)
    if plot_ref:
        bdf_def.plot_vtk("ref", size_cards)
    for i, ti in enumerate(time):
        if ra_movie is None:
            rai = ra[i]
        else:
            rai = ra[i] + ra_movie[i].reshape((3, 1))
        Rabi = Rab[i]
        gridmodel.set_solution(rai, Rabi, R0ab)
        disp, coord = interpolation.compute(
            gridmodel.model1_coord_valid, gridmodel.model1x_coord, X3d
        )
        uintrinsic[i] = disp
        rintrinsic[i] = coord
        if plot_timeinterval is not None:
            if i % plot_timeinterval == 0:
                tstep = i // plot_timeinterval
                # coord = nodesX2 + unastran[0,i,:,:3]
                bdf_def.update_bdf(coord, bdf_def.sorted_nodeids)
                bdf_def.plot_vtk(tstep, size_cards)
    return rintrinsic, uintrinsic


def rbf_based_movie(
    nastran_bdf,
    X,
    time_i,
    ra,
    time_movie,
    ra_movie,
    Rab,
    R0ab,
    vtkpath=None,
    plot_timeinterval=None,
    plot_ref=True,
    tolerance=1e-3,
    size_cards=8,
    rbe3s_full=True,
):
    bdf_model = BDF(debug=True)
    bdf_model.read_bdf(nastran_bdf, punch=False)
    # X = config.fem.X
    num_time = len(time_movie)
    numnodes = len(X)
    gridmodel = grid.RBE3Model(bdf_model, X, tolerance, rbe3s_full)
    bdf_def = bdfdef.DefBdf(nastran_bdf)
    X3d = bdf_def.get_nodes()
    numnodes3d = len(X3d)
    uintrinsic = np.zeros((num_time, numnodes3d, 3))
    rintrinsic = np.zeros((num_time, numnodes3d, 3))
    if vtkpath is not None:
        bdf_def.save_vtkpath(vtkpath)
    if plot_ref:
        bdf_def.plot_vtk("ref", size_cards)
    for i, ti in enumerate(time_movie):
        rai = ra[time_i] + ra_movie[i].reshape((3, 1))
        Rabi = Rab[time_i]
        gridmodel.set_solution(rai, Rabi, R0ab)
        disp, coord = interpolation.compute(
            gridmodel.model1_coord_valid, gridmodel.model1x_coord, X3d
        )
        uintrinsic[i] = disp
        rintrinsic[i] = coord
        if plot_timeinterval is not None:
            if i % plot_timeinterval == 0:
                tstep = i // plot_timeinterval
                # coord = nodesX2 + unastran[0,i,:,:3]
                bdf_def.update_bdf(coord, bdf_def.sorted_nodeids)
                bdf_def.plot_vtk(tstep, size_cards)
    return rintrinsic, uintrinsic


def nastran_based(): ...
