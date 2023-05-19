Dead_interpol=None #Interpolation of the dead forces [ [ [[t0,t1...tn],[f0,f1...fn]]..Coordinates..]..NumFLoads..]
Follower_points_app=[[6, -1, [0, 1, 2]], [11, -1, [0, 1, 2]]] #Points of the applied follower loads [[BeamSeg,Node,[..Coordinates..]]..NumFLoads..]
NumDLoads=0 #Number of (point) dead forces
Dead_points_app=None #Points of the applied dead loads [[BeamSeg,Node,[..Coordinates..]]..NumFLoads..]
Gravity=0 #Gravity loads
NumFLoads=2 #Number of (point) follower forces 
NumALoads=0 #0 or 1 for including aerodynamic forces
Follower_interpol=[[[[0.0, 5.3], [0.0, -2537.4359939558485]], [[0.0, 5.3], [0.0, -4837.801761284348]], [[0.0, 5.3], [0.0, -529971.8455660593]]], [[[0.0, 5.3], [0.0, -2537.4359939558485]], [[0.0, 5.3], [0.0, 4837.801761284348]], [[0.0, 5.3], [0.0, -529971.8455660593]]]] #Interpolation of the follower forces [ [ [[t0,t1...tn],[f0,f1...fn]]..Coordinates..]..NumFLoads..]
