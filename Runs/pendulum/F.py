Dead_interpol=None #Interpolation of the dead forces [ [ [[t0,t1...tn],[f0,f1...fn]]..Coordinates..]..NumFLoads..]
Follower_points_app=None #Points of the applied follower loads [[BeamSeg,Node,[..Coordinates..]]..NumFLoads..]
NumDLoads=0 #Number of (point) dead forces
Dead_points_app=None #Points of the applied dead loads [[BeamSeg,Node,[..Coordinates..]]..NumFLoads..]
Gravity=0 #Gravity loads
NumFLoads=0 #Number of (point) follower forces 
NumALoads=0 #0 or 1 for including aerodynamic forces
Follower_interpol=None #Interpolation of the follower forces [ [ [[t0,t1...tn],[f0,f1...fn]]..Coordinates..]..NumFLoads..]
