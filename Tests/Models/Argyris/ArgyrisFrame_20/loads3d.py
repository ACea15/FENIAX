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

# Fname='F2d'

# NumFLoads = 1
# Follower_points_app = [[1,-1,[1]]]
# Follower_interpol = [[[[0.,2000],[0.,-2000]]]]

Fname='F3d'
NumFLoads = 1
Follower_points_app = [[1,-1,[1,4]]]
Follower_interpol = [[[[0.,1.e3],[0.,1.e3]],[[0.,50.,75,100,150,200,250,300,400,500,550,600],[0.,5e3,6.5e3,8e3,9e3,10e3,12.5e3,15e3,17.5e3,20e3,22.5e3,25e3]]]]
# Followerinterpol=[]
# for i in range(NumFLoads):
#     Followerinterpol.append([])
#     for d in range(len(Follower_points_app[i][2])):
#         Followerinterpol[i].append(interp1d(Follower_interpol[i][d][0],Follower_interpol[i][d][1]))


# Fname='F2d'

# NumFLoads = 1
# Follower_points_app = [[1,-1,[4]]]
# Follower_interpol = [[[[0.,50.,75,100,150,200,250,300,400,500,550,600],[0.,-5e3,-6.5e3,-8e3,-9e3,-10e3,-12.5e3,-15e3,-17.5e3,-20e3,-22.5e3,-25e3]]]]
# Followerinterpol=[]
# for i in range(NumFLoads):
#     Followerinterpol.append([])
#     for d in range(len(Follower_points_app[i][2])):
#         Followerinterpol[i].append(interp1d(Follower_interpol[i][d][0],Follower_interpol[i][d][1]))


if (__name__ == '__main__'):
    import intrinsic.Tools.write_config_file
    reload(intrinsic.Tools.write_config_file)
    from intrinsic.Tools.write_config_file import write_force,force_specify

    write_force_interpol=1
    if write_force_interpol:
        write_force(Fname,V,locals())
        print('Force Interpolation Saved')



    print('Loads')
