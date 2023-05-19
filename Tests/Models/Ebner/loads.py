import numpy as np
#import pdb
import os
model=os.getcwd().split('/')[-1]
import importlib
V=importlib.import_module("Runs"+'.'+model+'.'+'V')
import intrinsic.geometry
import intrinsic.functions
from scipy.interpolate import interp1d
BeamSeg, NumNode, NumNodes, DupNodes, inverseconn  = intrinsic.geometry.geometry_def(V.Grid,V.NumBeams,V.BeamConn,V.start_reading,V.beam_start,V.nodeorder_start,V.node_start,V.Clamped,V.ClampX,V.BeamsClamped)

Fname='F'

NumDLoads = 2
Dead_points_app = [[0,-1,[2]],[1,-1,[2]]]
Dead_interpol = [[[[0.,0.85],[0.,-0.85]]],[[[0.,0.85],[0.,-1.35]]]]



if (__name__ == '__main__'):
    import intrinsic.Tools.write_config_file
    reload(intrinsic.Tools.write_config_file)
    from intrinsic.Tools.write_config_file import write_force,force_specify

    write_force_interpol=1
    if write_force_interpol:
        write_force(Fname,V,locals())
        print('Force Interpolation Saved')


    print('Loads')
