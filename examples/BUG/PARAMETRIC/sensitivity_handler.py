from bug_handler import BUGHandler
from concurrent.futures import ThreadPoolExecutor
from nastran_preprocessor import *

class SensitivityHandler:
  def __init__(self,bdfname,model_handler:BUGHandler,nastran_loc,working_dir='./_temp'):
    self.bdfname=bdfname
    self.model_handler=model_handler
    self.preprocessor=NastranPreprocessor(nastran_loc)
    self.working_dir=working_dir
    
  def forward_analysis(self,param,y_control=None,aero=False):
    param_c=self.model_handler.convert_design_param(param,y_control)
    self.preprocessor.set_params(param_c)
    self.preprocessor.eigenvalue_analysis()
    if aero:
      self.preprocessor.aero_analysis()
    
  def get_sensitivity_all(self,param,command_feniax,results_path,fd=True):
    executor = ThreadPoolExecutor()
    e=executor.submit(os.system,command_feniax)
    param_c=self.model_handler.convert_design_param(param)
    self.preprocessor.set_params(param_c)
    #grad_param=self.preprocessor.sensitivity_structure_upwind()
    if fd:
      grad_param=self.preprocessor.sensitivity_structure_fd()
    else:
      grad_param=self.preprocessor.sensitivity_structure()
    print('Nastran sensitivity done')
    _=e.result()
    print('Feniax sensitivity done')
    with open(results_path,'rb') as f:
      grad_feniax=pickle.load(f)
    grad_whole=dict()
    for k in grad_param:
      if k[:2]=='KA':
        key=k[4:]
        #val=[(grad_param[k]*grad_feniax['Ka']).sum(axis=(-1,-2))]
        #val.append((grad_param['MAA'+k[3:]]*grad_feniax['Ma']).sum(axis=(-1,-2)))
        #val.append((grad_param['EVEC'+k[3:]]*grad_feniax['eigenvecs']).sum(axis=(-1,-2)))
        #val.append((grad_param['EVAL'+k[3:]]*grad_feniax['eigenvals']).sum(axis=(-1)))
        val=(grad_param[k]*grad_feniax['Ka']).sum(axis=(-1,-2))
        val+=((grad_param['MAA'+k[3:]]*grad_feniax['Ma']).sum(axis=(-1,-2)))
        val+=((grad_param['EVEC'+k[3:]]*grad_feniax['eigenvecs']).sum(axis=(-1,-2)))
        val+=((grad_param['EVAL'+k[3:]]*grad_feniax['eigenvals']).sum(axis=(-1)))
        grad_whole[key]=val
    self.grad_param=grad_param
    with open(self.preprocessor.fem_dir+"/sensitivity.pkl","wb") as f:
      pickle.dump(grad_whole, f)
    with open(self.preprocessor.fem_dir+"/sensitivity_nastran.pkl","wb") as f:
      pickle.dump(grad_param, f)
    return grad_whole
    
  def set_config(self,working_dir,delta=1e-2,num_parallel=4,fem_dir='./FEM',
                 num_modes_aero=50,num_modes_eig=500):
    self.working_dir=working_dir
    self.delta=delta
    self.num_parallel=num_parallel
    self.fem_dir=fem_dir
    self.num_modes_aero=num_modes_aero
    self.num_modes_eig=num_modes_eig
    self.preprocessor.set_config(self.bdfname,working_dir=working_dir,
                                 delta=delta,num_parallel=num_parallel,
                                 fem_dir=fem_dir,num_modes_aero=num_modes_aero,
                                 num_modes_eig=num_modes_eig)
