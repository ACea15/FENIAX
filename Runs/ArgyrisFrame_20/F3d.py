Dead_interpol=None #Interpolation of the dead forces [ [ [[t0,t1...tn],[f0,f1...fn]]..Coordinates..]..NumFLoads..]
Follower_points_app=[[1, -1, [1, 4]]] #Points of the applied follower loads [[BeamSeg,Node,[..Coordinates..]]..NumFLoads..]
NumDLoads=0 #Number of (point) dead forces
Dead_points_app=None #Points of the applied dead loads [[BeamSeg,Node,[..Coordinates..]]..NumFLoads..]
Gravity=0 #Gravity loads
NumFLoads=1 #Number of (point) follower forces 
NumALoads=0 #0 or 1 for including aerodynamic forces
Follower_interpol=[[[[0.0, 1000.0], [0.0, 1000.0]], [[0.0, 50.0, 75, 100, 150, 200, 250, 300, 400, 500, 550, 600], [0.0, 5000.0, 6500.0, 8000.0, 9000.0, 10000.0, 12500.0, 15000.0, 17500.0, 20000.0, 22500.0, 25000.0]]]] #Interpolation of the follower forces [ [ [[t0,t1...tn],[f0,f1...fn]]..Coordinates..]..NumFLoads..]
