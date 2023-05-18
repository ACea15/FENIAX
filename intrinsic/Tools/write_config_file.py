import os
import pdb
def write_config(var,Vname='V'):

    femL = [['Grid','Grid file where ordered nodes and the beam-segments they belong to are located ']]
    femL.append(['K_a','Stiffness matrix location'])
    femL.append(['M_a','Mass matrix location'])
    femL.append(['feminas_dir','FEM4INAS directory'])
    femL.append(['model_name','Directory to the model in Models'])
    femL.append(['model','Model name'])
    fem_files_default={}
    fem_files_default['op2name'] = [None,'']
    fem_files_default['M_a2'] = ['','Mass matrix location for multibody system (zeros not removed)']
    fem_files_default['K_a2'] = ['','Stiffness matrix location for multibody system (zeros not removed)']
    fem_files_default['cg'] = ['','Location of npy-file with calculated CG']
    #################################################################################################################################
    topoL = [['NumModes','Number of modes in the analysis']]
    topoL.append(['NumBeams','Number of beam-segments'])
    topoL.append(['BeamConn','Connectivities between beam segments [[..[..0-direction..]..BeamNumber..],[..[..1-direction..]..BeamNumber..]]'])
    topo_files_default={}
    topo_files_default['Nastran_modes'] =[0,'Modes directly from Nastran']
    topo_files_default['Nastran_modes_dic'] =[{},'Dictionary to put nastran modes into the current formulation']
    topo_files_default['Check_Phi2'] = [0,'Check equilibrium of forces for Phi2 (in free-model) ']
    topo_files_default['Path4Phi2'] = [0,'0 -> Phi2 calculated in optimal way 1-> Path enforced in the opposite or 1-direction']
    topo_files_default['ReplaceRBmodes'] = [0,'Replace the rigid-body modes']
    topo_files_default['NumModes_res'] = [0,'Number of residualized modes']
    #################################################################################################################################
    readL=[['node_start','NumNode=max([max(BeamSeg[j].NodeOrder) for j in range(NumBeams)])+node_start']]
    readL.append(['start_reading','range(start_reading,len(lin)):'])
    readL.append(['beam_start','j=int(s[4])-beam_start BeamSeg[j]'])
    readL.append(['nodeorder_start','aset start BeamSeg[j].NodeOrder.append(int(s[3])-nodeorder_start'])
    #################################################################################################################################
    timeL = [['t0','Initial time'],['tf','Final time'],['tn','Total time steps']]
    timeL_op = [['dt',''],['ti','']]
    #################################################################################################################################
    constants_files_default={}
    constants_files_default['EMAT'] = ['np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,-1,0,0,0],[0,1,0,0,0,0]])','']
    constants_files_default['I3'] = ['np.eye(3)','']
    constants_files_default['e_1'] = ['np.array([1.,0.,0.])','Beam-segments local direction']
    constants_files_default['g0'] = [[0.,0.,-9.80665,0.,0.,0.],'Gravity acceleration on earth']
    #################################################################################################################################
    boundary_files_default={}
    boundary_files_default['RigidBody_Modes'] = [1,'']
    boundary_files_default['Clamped'] = [0,'Clamped model']
    boundary_files_default['ClampX'] = ['np.array([0.,0.,0.])','Coordinate of clamped node']
    boundary_files_default['initialbeams'] = [[0],'Beam-segments attach to first node']
    boundary_files_default['BeamsClamped'] = [boundary_files_default['initialbeams'][0],'']
    boundary_files_default['MBbeams'] = [[],'Multibody beams (in linear problem)']
    boundary_files_default['MBnode'] = [{},'Position in the in the stiffness and mass matrices of node attach to MB node']
    boundary_files_default['MBnode2'] = [{},'Position of multibody node in the stiffness and mass matrices']
    boundary_files_default['MBdof'] = [{},'Degrees of freedom joined in the Multibody (linear) system']
    boundary_files_default['MBdofree'] = [{},'Degrees of freedom independent in the Multibody (linear) system']
    boundary_files_default['variablesmb'] = [[],'Varibles input  files of the multibodyset']
    boundary_files_default['forcesmb'] = [[],'Force input files of the multibody set']
    boundary_files_default['aeromb'] = [[],'Aero force input files of the multibody set']
    boundary_files_default['results_modesMB'] = [[],'Folder to find in the multibody analysis ']
    boundary_files_default['NumBodies'] = [1,'Total number of bodies']
    boundary_files_default['NumConstrains'] = [0,'Total number of constrains.']
    boundary_files_default['Constrains'] = [{},'Constrains={"c1":[[bodies],[beams],"1/0-constrain displacement,1/2/3 constrain x,y,z axes respectively"]}']
    #################################################################################################################################
    loading_files_default={}
    loading_files_default['gravity_on']=[0,'0 or 1 whether gravity is to be accounted for']
    loading_files_default['loading']=[0,'0 or 1 whether a strcuctural force is defined in the analysis ()']
    loading_files_default['static']=[0,'0 or 1 for static computatitions (nonlinear system of algebraic equations)']
    loading_files_default['dynamic']=[0,'0 or 1 for dynamic computatitions (nonlinear system of ODEs)']
    loading_files_default['linear']=[0,"0 or 1 for 'linear' analysis (removing Gammas, cubic terms, which is not exactly linear)"]
    loading_files_default['init_q0']=[0,'Initial (qs) conditions other than 0']
    loading_files_default['q0_file']=[None,'File to function for Initial qs']
    loading_files_default['print_timeSteps']=[1,'Print time steps in the ODE solution']
    loading_files_default['quadratic_integrals']=[0,'Quadratic terms in the integrals of the nonlinear terms Gamma1 and Gamma2']
    #################################################################################################################################
    solver_files_default={}
    solver_files_default['ODESolver']=['RK4','ODE solver for the dynamic system']
    #################################################################################################################################
    solveroptions_files_default={}
    solveroptions_files_default['rotation_quaternions'] = [0,'']
    solveroptions_files_default['rotation_strains'] = [0,'']
    #######
    # FEM #
    #######
    fem_files = {}
    fem_files['_ff'] = '# FEM Files Settings #'
    for i in femL:
        try:
            fem_files[i[0]] = [var[i[0]],i[1]]
        except:
            print('Input files needed.')

    for ki in fem_files_default.keys():
        try:
            fem_files[ki] = [var[ki],fem_files_default[ki][1]]
        except:
            fem_files[ki] = [fem_files_default[ki][0],fem_files_default[ki][1]]

    #################
    #   Read Grid   #
    #################
    readgrid_files = {}
    readgrid_files['_rg'] = '# Read Grid File Settings #'
    for i in readL:
        try:
            readgrid_files[i[0]] = [var[i[0]],i[1]]
        except:
            print('Read Grid files needed.')

    ###############
    # Topology    #
    ###############
    topo_files = {}
    topo_files['_tof'] = '# Topology Settings #'
    for i in topoL:
        try:
            topo_files[i[0]] = [var[i[0]],i[1]]
        except:
            print('topology files needed.')

    for ki in topo_files_default.keys():
        try:
            topo_files[ki] = [var[ki],topo_files_default[ki][1]]
        except:
            topo_files[ki] = [topo_files_default[ki][0],topo_files_default[ki][1]]

    ##############
    # Read Files #
    ##############
    read_files = {}
    for i in readL:
        try:
            read_files[i[0]] = [var[i[0]],i[1]]
        except:
            print('read parameters needed.')

    ########
    # Time #
    ########
    time_files = {}
    time_files['_tf'] = '# Time Settings #'
    for i in timeL:
        try:
            time_files[i[0]] = [var[i[0]],i[1]]
        except:
            print('time files needed.')
    try:
        time_files['dt'] = [var['dt'],'Increment of time']
    except:
        time_files['dt'] = [(float(var['tf'])-float(var['t0']))/(var['tn']-1),'Increment of time']
    try:
        time_files['ti'] = [var['ti'],'Time vector']
    except:
        time_files['ti'] = ['np.linspace({},{},{})'.format(var['t0'],var['tf'],var['tn']),'Time vector']

    #############
    # Constants #
    #############
    constants_files = {}
    constants_files['_cf'] = '# Constants Settings #'
    for ki in constants_files_default.keys():
        try:
            constants_files[ki] = [var[ki],constants_files_default[ki][1]]
        except:
            constants_files[ki] = [constants_files_default[ki][0],constants_files_default[ki][1]]

    ##############
    # Boundaries #
    ##############
    boundary_files = {}
    boundary_files['_bf'] = '# Boundary Settings #'
    for ki in boundary_files_default.keys():

        try:
            boundary_files[ki] = [var[ki],boundary_files_default[ki][1]]
        except:
            boundary_files[ki] = [boundary_files_default[ki][0],boundary_files_default[ki][1]]
    #pdb.set_trace()
    if boundary_files['MBdof'][0]:
        try:
            boundary_files['MBdofree'] = [var['MBdofree'],boundary_files_default['MBdofree'][1]]
        except:
            d2i={}
            for di in boundary_files['MBdof'][0].keys():
                d2i[di] = list(set([0,1,2,3,4,5])-set(boundary_files['MBdof'][0][di]))
            boundary_files['MBdofree'] = [d2i,boundary_files_default['MBdofree'][1]]
    ###############
    # Loading and redirecting Solvers#
    ###############
    loading_files = {}
    loading_files['_lf'] = '# Loading Settings to redirect solvers#'
    for ki in loading_files_default.keys():
        try:
            loading_files[ki] = [var[ki],loading_files_default[ki][1]]
        except:
            loading_files[ki] = [loading_files_default[ki][0],loading_files_default[ki][1]]

    ############################
    # ODE and algebraic solvers#
    ############################
    solver_files = {}
    solver_files['_ode'] = '# ODE solvers #'
    for ki in solver_files_default.keys():
        try:
            solver_files[ki] = [var[ki],solver_files_default[ki][1]]
        except:
            solver_files[ki] = [solver_files_default[ki][0],solver_files_default[ki][1]]

    solveroptions_files = {}
    solveroptions_files['_options'] = '# Options for solvers #'
    for ki in solveroptions_files_default.keys():
        try:
            solveroptions_files[ki] = [var[ki],solveroptions_files_default[ki][1]]
        except:
            solveroptions_files[ki] = [solveroptions_files_default[ki][0],solveroptions_files_default[ki][1]]

    ##########################################################################################
    files = [readgrid_files,topo_files,time_files,constants_files,boundary_files,loading_files,solveroptions_files]
    files_string = [fem_files,solver_files]
    if not os.path.exists(var['feminas_dir'] + '/Runs/'+var['model']):
      os.makedirs(var['feminas_dir'] + '/Runs/'+var['model'])
      with open(var['feminas_dir'] + '/Runs/'+var['model']+'/__init__.py', 'w') as f2:
        print('Init file created')

    with open(var['feminas_dir'] + '/Runs/'+var['model']+'/'+Vname+'.py', 'w') as f:
      #f.write("""fmodel_name='%s'\n"""% model_name)
      #f.write("""ffeminas_dir = '%s' """ % feminas_dir)
      f.write('import numpy as np \n')
      f.write('\n')
      for i in files_string:
          for j in i.keys():
            if j[0] == '_':
                f.write("""%s""" % '#'*len(i[j])+"""\n""")
                f.write("""%s\n"""% i[j])
                f.write("""%s""" % '#'*len(i[j])+"""\n""")
                for jj in i.keys():
                    if jj !=j:
                        f.write("""%s='%s' #%s\n"""% (jj,i[jj][0],i[jj][1]))
                continue

      for i in files:
          for j in i.keys():
            if j[0] == '_':
                f.write("""%s""" % '#'*len(i[j])+"""\n""")
                f.write("""%s\n"""% i[j])
                f.write("""%s""" % '#'*len(i[j])+"""\n""")
                for jj in i.keys():
                    if jj !=j:
                        f.write("""%s=%s #%s\n"""% (jj,i[jj][0],i[jj][1]))
                continue
    print('Configuration file written')

def write_force(Fname,V,kwargs):

    loading_files={}
    loading_files_default={}
    loading_files_default['Gravity']=[0,'Gravity loads']
    loading_files_default['NumFLoads']=[0,'Number of (point) follower forces ']
    loading_files_default['NumDLoads']=[0,'Number of (point) dead forces']
    loading_files_default['NumALoads']=[0,"0 or 1 for including aerodynamic forces"]
    loading_files_default['Follower_points_app']=[None,'Points of the applied follower loads [[BeamSeg,Node,[..Coordinates..]]..NumFLoads..]']
    loading_files_default['Follower_interpol']=[None,'Interpolation of the follower forces [ [ [[t0,t1...tn],[f0,f1...fn]]..Coordinates..]..NumFLoads..]']
    loading_files_default['Dead_points_app']=[None,'Points of the applied dead loads [[BeamSeg,Node,[..Coordinates..]]..NumFLoads..]']
    loading_files_default['Dead_interpol']=[None,'Interpolation of the dead forces [ [ [[t0,t1...tn],[f0,f1...fn]]..Coordinates..]..NumFLoads..]']
    for ki in loading_files_default.keys():
        try:
            loading_files[ki] = [kwargs[ki],loading_files_default[ki][1]]
        except:
            loading_files[ki] = [loading_files_default[ki][0],loading_files_default[ki][1]]
    with open(V.feminas_dir + '/Runs/'+V.model+'/'+Fname+'.py', 'w') as f:
        for jj in loading_files.keys():

            f.write("""%s=%s #%s\n"""% (jj,loading_files[jj][0],loading_files[jj][1]))

    print('Forces file written')

def write_aero(Aname,V,kwargs):

    loading_files={}
    loading_files_default={}
    loading_files_default['u_inf']=[0.,'Flow velocity ']
    loading_files_default['rho_inf']=[0.,'Flow density']
    loading_files_default['q_inf']=[0.,'Dynamic pressure']
    loading_files_default['c']=[0,'Reference chord']
    loading_files_default['rbd']=[0,'Rigid body defined']
    loading_files_default['rbd_modify']=[0,'Rigid body modified through the simulation']
    loading_files_default['rbd_modify_time']=[0,'Rigid body time to modify']
    loading_files_default['rbdx']=[0,'Rigid body defined through AICsx']
    loading_files_default['NumPoles']=[0,"Number of poles in the analysis"]
    #loading_files_default['NumAeroStates']=[0,'']
    loading_files_default['LocPoles']=[None,'Location of file with poles']
    loading_files_default['Amatrix']=[None,'Location of aerodynamic AIC file (Qhh)']
    loading_files_default['Axmatrix']=[None,'Location of aerodynamic matrix Qx giving initial forces']
    loading_files_default['Agmatrix']=[None,'Location of aerodynamic matrix Qhj for gust response']
    loading_files_default['Aelevator_matrix'] = ['','Location of aerodynamic AIC for elevator']
    loading_files_default['qx']=[[],'Modal component for Qx']
    loading_files_default['TrimOn']=[0,'Initialize trim methodology']
    loading_files_default['Trim_NLin']=[1,'Linear or NL method to track trimming point']
    loading_files_default['K_spring']=[0,'Spring constant for atachement in trim ']
    loading_files_default['D_spring']=[0,'Damper constant for atachement in trim ']
    loading_files_default['time_attachment']=[[0.,V.tf],'Initial and final times for attachment forces to act']
    loading_files_default['Gain_PID']=[[0.,0.,0.],'Controller gains in trim']
    #loading_files_default['Aelevator_matrix'] = ['','Location of aerodynamic AIC for elevator']
    loading_files_default['aelink'] = [[],'Link between elevators working together [+-1]']
    loading_files_default['elevator_index'] = [[],'Index of elevator entris in Qx matrix']
    loading_files_default['trim_time'] = [[[]],'Time interval where trim strategy is on']
    loading_files_default['print_trim_cg'] = [1,'cg']
    loading_files_default['print_trim_PDIgains'] = [1,'']
    loading_files_default['print_trim_eta'] = [1,'']
    loading_files_default['GustOn'] = [0,'Gust loads activation']
    loading_files_default['Time_start_gust'] = [0.,'Gust loads activation']
    loading_files_default['L_g'] = [0,'Gust length']
    loading_files_default['V0_g'] = [0,'Gust vertical velocity']
    loading_files_default['X0_g'] = [0,'Gust origin']
    loading_files_default['time_gust'] = ['np.linspace(0,10,1000)','Time vector for interpolation']
    loading_files_default['Dihedral'] = ['','Dihedral of panels']
    loading_files_default['Gust_type'] = ['1mc','Type of Gust']
    loading_files_default['Control_nodes'] = ['','Control nodes position of each aerodynamic panel -Numpy matrix location-']
    loading_files_default['Panels_LE'] = ['','Leading edge panels (for gust application time) -Numpy array location-']
    loading_files_default['Panels_TE'] = ['','Trailing edge panels (for gust application time)  -Numpy array location-']
    loading_files_default['Gust_shape_span'] = ['','Activate flag to functions that define antisymetric gusts along the span: antisym_span_quad, y:y**2/self.A.wing_span**2 ,antisym_span_lin, y:y/self.A.wing_span default, y:1.']
    loading_files_default['wing_span'] = [0,'Wing span for antisymetric gust definition']
    #loading_files_default['aeroforce']=[0,'Use of total aerodynamic force with dq1']
    #loading_files_default['aeroforcesystem']=[1,'Use of aerodynamic force through Ainv for dq1']
    for ki in loading_files_default.keys():
        try:
            loading_files[ki] = [kwargs[ki],loading_files_default[ki][1]]
        except:
            loading_files[ki] = [loading_files_default[ki][0],loading_files_default[ki][1]]
    if loading_files['q_inf'][0] == 0.:
        loading_files['q_inf'] = 0.5*loading_files['rho_inf'][0]*loading_files['u_inf'][0]**2

    with open(V.feminas_dir + '/Runs/'+V.model+'/'+Aname+'.py', 'w') as f:
        f.write('import numpy as np \n')
        string_files = ['Amatrix','Axmatrix','Agmatrix','Aelevator_matrix','LocPoles','Dihedral','Gust_type','Control_nodes','Panels_LE','Panels_TE','Gust_shape_span']
        for jj in loading_files.keys():

            if jj in string_files:
                f.write("""%s='%s' #%s\n"""% (jj,loading_files[jj][0],loading_files[jj][1]))
            else:
                f.write("""%s=%s #%s\n"""% (jj,loading_files[jj][0],loading_files[jj][1]))

    print('Aero forces file written')


def force_specify():


   # Fa_save='Fa_pressure'
   # BeamForce=[0]
   # NodeForce=[[-1]]
   # load_max = [[120000]]
   # load_direction = [[np.array([0,1,0,0,0,0])]]
   # load_step = 1
   Fa = [np.zeros((V.tn,BeamSeg[i].EnumNodes,6)) for i in range(V.NumBeams)]

   specify=0
   if specify:
    spec=10**3*np.asarray([1.,3.7,7.6,12.1,15.5,17.5,25.2,39.3,48.2,61.,80.,94.5,109.5,120.])
    for i in range(V.tn):

          Fa[0][i][-1][1]=spec[i]


   pressure=1
   if pressure:

    BeamForce=[0]
    pre=[1000*np.asarray([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.96])]
    direc=[np.array([0,1,0])]

    for i in range(len(BeamForce)):
      for ti in range(V.tn):
       for j in range(BeamSeg[i].EnumNodes):
        if j==BeamSeg[i].EnumNodes-1:
         Fa[i][ti][j][0:3]=0.5*pre[i][ti]*direc[i]*BeamSeg[i].L/BeamSeg[i].EnumNode
        else:
         Fa[i][ti][j][0:3]=pre[i][ti]*direc[i]*BeamSeg[i].L/BeamSeg[i].EnumNode


   ramp=0
   if ramp:
         for i in range(len(BeamForce)):
           #if i in BeamsClamped:
               for j in range(len(NodeForce[i])):
                 for ti in range(1,V.tn):
                   Fa[BeamForce[i]][ti][NodeForce[i][j]] = load_max[i][j]*load_direction[i][j]

                 for ti in range(int(round(load_step*(V.tn)))):
                   if ti==0:
                    Fa[BeamForce[i]][ti][NodeForce[i][j]] = load_max[i][j]*load_direction[i][j]*(ti+1)/V.tn/load_step
                   else:
                    Fa[BeamForce[i]][ti][NodeForce[i][j]] = load_max[i][j]*load_direction[i][j]*(ti+1)/V.tn/load_step

   return Fa
