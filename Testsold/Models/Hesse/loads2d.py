import numpy as np
#import pdb
import os
model='Hesse_25'#os.getcwd().split('/')[-1]
import importlib
V=importlib.import_module("Runs"+'.'+model+'.'+'V')

# NumFLoads = 1
# Follower_points_app = [[0,-1,[0,1,5]]]
# Follower_interpol = [[[[0.,2.5,2.5,10.5],[4.8,4.8,0.,0.]],[[0.,2.5,2.5,10.5],[-6.4,-6.4,0.,0.]],[[0.,2.5,2.5,10.5],[-80.,-80.,0.,0.]]]]

# NumFLoads = 1
# Follower_points_app = [[0,-1,[0,4,5]]]
# Follower_interpol = [[[[0.,2.5,5.,10.5],[0.,20.,0.,0.]],[[0.,2.5,5.,10.5],[0.,100.,0.,0.]],[[0.,2.5,5.,10.5],[0.,200.,0.,0.]]]]

NumDLoads = 1
Dead_points_app = [[0,-1,[0,5]]]
Dead_interpol = [[[[0.,2.5,2.5,10.5],[8.,8.,0.,0.]],[[0.,2.5,2.5,10.5],[-80.,-80.,0.,0.]]]]
Fname='Fdead2d'
# NumDLoads = 1
# Dead_points_app = [[0,-1,[0]]]
# Dead_interpol = [[[[0.,2.5,2.5,10.5],[15.,15.,0.,0.]]]]

# NumDLoads = 1
# Dead_points_app = [[0,-1,[0,4,5]]]
# Dead_interpol = [[[[0.,2.5,5.,10.5],[0.,20.,0.,0.]],[[0.,2.5,5.,10.5],[0.,100.,0.,0.]],[[0.,2.5,5.,10.5],[0.,-200.,0.,0.]]]]



if (__name__ == '__main__'):
    import intrinsic.Tools.write_config_file
    reload(intrinsic.Tools.write_config_file)
    from intrinsic.Tools.write_config_file import write_force,force_specify

    write_force_interpol=1
    if write_force_interpol:
        write_force(Fname,V,locals())
        print('Force Interpolation Saved')

    print('Loads')
