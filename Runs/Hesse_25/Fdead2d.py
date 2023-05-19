Dead_interpol=[[[[0.0, 2.5, 2.5, 10.5], [8.0, 8.0, 0.0, 0.0]], [[0.0, 2.5, 2.5, 10.5], [-80.0, -80.0, 0.0, 0.0]]]] #Interpolation of the dead forces [ [ [[t0,t1...tn],[f0,f1...fn]]..Coordinates..]..NumFLoads..]
Follower_points_app=None #Points of the applied follower loads [[BeamSeg,Node,[..Coordinates..]]..NumFLoads..]
NumDLoads=1 #Number of (point) dead forces
Dead_points_app=[[0, -1, [0, 5]]] #Points of the applied dead loads [[BeamSeg,Node,[..Coordinates..]]..NumFLoads..]
Gravity=0 #Gravity loads
NumFLoads=0 #Number of (point) follower forces 
NumALoads=0 #0 or 1 for including aerodynamic forces
Follower_interpol=None #Interpolation of the follower forces [ [ [[t0,t1...tn],[f0,f1...fn]]..Coordinates..]..NumFLoads..]
