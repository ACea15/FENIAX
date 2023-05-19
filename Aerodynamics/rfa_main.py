import numpy as np
import pdb
import argparse
import os
from intrinsic.functions import my_timer
import Utils.FEM_Readers
import rfa,opt_poles,rfa_plot

terminal_run=0
if terminal_run:
  parser = argparse.ArgumentParser(description='Main Aerodynamics File')
  parser.add_argument('config_file',type=str, help='Define configuration file')
  args = parser.parse_args()
  wdir = os.getcwd()
else:
  class arguments:
    pass
  args=arguments()
  args.config_file = 'Configuration_Aero.txt'
  #args.config_file = 'Configuration_AeroGust.txt'
  #args.config_file = 'Configuration_File.txt'
  #wdir=os.getcwd()+ '/../Models/XRF1/v1_v/NASTRAN3/'
  wdir=os.getcwd()+ '/../Models/XRF1-trim2/NASTRAN/'
  #wdir=os.getcwd()+ '/../Models/HALEXc'

#Input_file = (raw_input('Select the Configuration File:  '))
Input_file=wdir + '/' + args.config_file     #'/Configuration_File.txt'
#Input_file= args.config_file     #'/Configuration_File.txt'
mkaero_file=Utils.FEM_Readers.read_path(Input_file,'mkaero_file',wdir)
reduced_freq=Utils.FEM_Readers.read_mkaero_file(mkaero_file)
RFA_Method=Utils.FEM_Readers.read_string(Input_file,'RFA_Method')
#RFA_Method='e'
Gaf_file=Utils.FEM_Readers.read_path(Input_file,'Gaf_file',wdir)
if Gaf_file.split('.')[-1] == 'npy':
    aero_matrices=np.load(Gaf_file)
    aero_matrices_real = aero_matrices.real
    aero_matrices_imag = aero_matrices.imag
else:
    aero_matrices_real,aero_matrices_imag,aero_matrices=Utils.FEM_Readers.read_complex_nastran_matrices2(Gaf_file)
NumModes=np.shape(aero_matrices_real)[1]
NumPoles=Utils.FEM_Readers.read_float(Input_file,'n_poles')
NumPoles=int(NumPoles)
rfa0 = Utils.FEM_Readers.read_float(Input_file,'rfa0')
kstep = Utils.FEM_Readers.read_float(Input_file,'kstep')
err_type = Utils.FEM_Readers.read_string(Input_file,'err_type')
kmax = reduced_freq[-1]
save_AICs = Utils.FEM_Readers.read_string(Input_file,'save_AICs')
save_Poles = Utils.FEM_Readers.read_string(Input_file,'save_Poles')
opt_brute = Utils.FEM_Readers.read_float(Input_file,'opt_brute')
opt_least = Utils.FEM_Readers.read_float(Input_file,'opt_least')
sol_step = Utils.FEM_Readers.read_float(Input_file,'sol_step')


Matrix_type=Utils.FEM_Readers.read_string(Input_file,'Matrix_type')
Matrix_type=Matrix_type.lower()
aesurf_scale=Utils.FEM_Readers.read_array(Input_file,'aesurf_scale')
Projection=Utils.FEM_Readers.read_string(Input_file,'Projection')
Modal_Matrix_file=Utils.FEM_Readers.read_path(Input_file,'Modal_Matrix_file',wdir)
RB_Modes_QS=Utils.FEM_Readers.read_string(Input_file,'RB_Modes_QS')
RB_Modes_norm=Utils.FEM_Readers.read_string(Input_file,'RB_Modes_norm')
RB_Modes_norm_1=Utils.FEM_Readers.read_string(Input_file,'RB_Modes_norm_1')
mode_scale=Utils.FEM_Readers.read_array(Input_file,'mode_scale')
plot=Utils.FEM_Readers.read_string(Input_file,'plot')
plot_points=Utils.FEM_Readers.read_array2d(Input_file,'plot_points')


poles0=np.zeros((NumPoles))
if NumPoles>=0:
    for i in range(NumPoles):
        poles0[i]=kmax/(i+1)

poles = poles0
#poles = np.load('/media/pcloud/Computations/FEM4INAS/Models/XRF1/v1_v/NASTRAN3/AERO/Poles00_8r20.npy')
#poles = np.load('/media/pcloud/Computations/FEM4INAS/Models/XRF1-2/NASTRAN/AERO/Poles000_8r70.npy')

#rfa0=1
#opt_least = 0
if opt_brute:
  err_brute,poles_brute = opt_poles.min_brute(reduced_freq,RFA_Method,aero_matrices_real,aero_matrices_imag,kmax,kstep,NumPoles,err_type,sol_step,rfa0)
  poles = poles_brute
if opt_least:
  if opt_brute:
    err_least,poles_least = opt_poles.min_least_squares(poles_brute,reduced_freq,RFA_Method,aero_matrices_real,aero_matrices_imag,err_type,sol_step,rfa0)
  else:
    err_least,poles_least = opt_poles.min_least_squares(poles,reduced_freq,RFA_Method,aero_matrices_real,aero_matrices_imag,err_type,sol_step,rfa0)
  poles = poles_least
#k_matrix,k_matrix2,RFA_mat,RFA_mat2,err,err2 = opt.poles.y_poles(poles_brute,reduced_freq,RFA_Method,aero_matrices_real,aero_matrices_imag,err_type)


k_matrix,RFA_mat,err = opt_poles.y_poles(poles,reduced_freq,RFA_Method,aero_matrices_real,aero_matrices_imag,aero_matrices,err_type,sol_step,rfa0)



if save_AICs:
  np.save(wdir+save_AICs+'_%s%s%s'%(NumPoles,RFA_Method,NumModes),np.array(RFA_mat))
if save_Poles:
  np.save(wdir+save_Poles+'_%s%s%s'%(NumPoles,RFA_Method,NumModes),poles)


#np.save('/home/ac5015/Dropbox/Computations/FEM4INAS/Aerodynamics/../Models/GolandWing/NASTRAN/new/../../AERO/Poles4.npy',poles0)


#polesx=poles
#RFA_matx = RFA_mat
plot_points = np.asarray([[1,10],[0,0],[0,2],[0,3],[7.,1],[5,2],[4,3],[1,5],[6,2]])
plot_points = np.asarray([[0,0],[0,1],[0,2],[0,3],[1.,1.],[1.,3.],[1,4],[2,1],[2,2],[4,1],[6,3],[5,8],[9,0],[7,2],[3,10],[4,5]])
plot_points = np.asarray([[10,0],[15,1],[10,8],[1,0],[1.,5.],[10.,20.],[20,100],[2,150],[20,2],[19,116],[8,300],[5,8],[9,0],[7,200],[3,1200],[15,800]])
plot_points = np.asarray([[10,0],[15,1],[10,8],[1,0],[1.,5.],[10.,20.],[20,100],[2,150],[20,2],[19,116],[8,300],[5,8],[9,0],[7,200],[3,1200],[15,800]])
if plot == 'y':

  rfa_plot.plot1(reduced_freq,plot_points,poles,RFA_mat,RFA_Method,aero_matrices_real,aero_matrices_imag)
  #rfa_plot.plot1(reduced_freq,plot_points,polesx,RFA_matx,RFA_Method,aero_matrices_real,aero_matrices_imag)
  rfa_plot.plot3(reduced_freq,plot_points,poles,polesx,RFA_mat,RFA_matx,RFA_Method,aero_matrices_real,aero_matrices_imag)

#rfa.RFA_freq(poles,ki,RFA_mat,RFA_Method)
