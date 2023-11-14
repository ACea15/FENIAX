import numpy as np
#import pdb
import os
model='wingSP'
import importlib
V=importlib.import_module("Runs"+'.'+model+'.'+'V')
import intrinsic.geometryrb
import intrinsic.functions
from scipy.interpolate import interp1d
BeamSeg, NumNode, NumNodes, DupNodes, inverseconn  = intrinsic.geometryrb.geometry_def(V.Grid,V.NumBeams,V.BeamConn,V.start_reading,V.beam_start,V.nodeorder_start,V.node_start,V.Clamped,V.ClampX,V.BeamsClamped)

Fname='Ftest'
fabs=-5e5
a1=BeamSeg[4].GlobalAxes.T.dot([-2e5,0.,6e5])
#a2=BeamSeg[11].GlobalAxes.T.dot([0,0,5.3e5])
NumFLoads = 1
#Follower_points_app = [[6,-1,[0,1,2]],[11,-1,[0,1,2]]]
#Follower_interpol = [ [[[0.,5.3],[0.,a1[0]]],[[0.,5.3],[0.,a1[1]]],[[0.,5.3],[0.,a1[2]]]] , [[[0.,5.3],[0.,a2[0]]],[[0.,5.3],[0.,a2[1]]],[[0.,5.3],[0.,a2[2]]]]]
Follower_points_app = [[4,-1,[0,1,2]]]
#Follower_interpol = [ [[[0.,5.3],[0.,a1[0]]],[[0.,5.3],[0.,a1[1]]],[[0.,5.3],[0.,a1[2]]]]]
Follower_interpol = [ [[[0.,4.,4.,20.],[0.05*a1[0],1.*a1[0],0.,0.]],[[0.,4.,4.,20.],[0.05*a1[1],1.*a1[1],0.,0.]],[[0.,4.,4.,20.],[0.05*a1[2],1.*a1[2],0.,0.]]]]
# Fname='F3d'

# NumFLoads = 1
# Follower_points_app = [[1,-1,[1,4]]]
# Follower_interpol = [[[[0.,1.e3],[0.,-1.e3]],[[0.,50.,75,100,150,200,250,300,400,500,550,600],[0.,-5e3,-6.5e3,-8e3,-9e3,-10e3,-12.5e3,-15e3,-17.5e3,-20e3,-22.5e3,-25e3]]]]
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
