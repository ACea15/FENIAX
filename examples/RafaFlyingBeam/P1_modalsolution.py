# [[file:modelgen.org::*Build Nastran models][Build Nastran models:1]]
from pyNastran.bdf.bdf import BDF
import numpy as np
from dataclasses import dataclass
import pathlib
import feniax.unastran.matrixbuilder as matrixbuilder
import feniax.unastran.op2reader as op2reader
from feniax.unastran.asetbuilder import BuildAsetModel

pathlib.Path('./FEM').mkdir(parents=True, exist_ok=True)
pathlib.Path('./NASTRAN/data_out').mkdir(parents=True, exist_ok=True)
pathlib.Path('./NASTRAN/simulations_out').mkdir(parents=True, exist_ok=True)
# Build Nastran models:1 ends here

# [[file:modelgen.org::*Build Nastran models][Build Nastran models:2]]
@dataclass
class Config:
    BAR: bool = False
    L: float = 1.
    THETA0: float = 0.
    M: float = 1000.
    I: float = 1.
    Im: float = M * L **2 /2
    OFFSET_x: float = 0.
    OFFSET_z: float = 0.
    E: float = 1e6
    A: float = 0.05
    J: float = I * 5
    omega: float = None

    def __post_init__(self):

        self.omega = (24*self.E * self.I / (self.M * self.L ** 3) )**0.5
        #self.omega = (12*self.E * self.I / (self.M * self.L ** 3) )**0.5

def build_bdf(config: Config):

    mesh=BDF(debug=True)
    ############################
    node1 = ['GRID', 1, None, 0., 0., 0., None, None, None]
    node2 = ['GRID', 2, None, config.L * np.cos(config.THETA0), 0., config.L * np.sin(config.THETA0), None, None, None]
    node3 = ['GRID', 3, None, -config.L * np.cos(config.THETA0), 0., -config.L * np.sin(config.THETA0), None, None, None]
    mesh.add_card(node1, 'GRID')
    mesh.add_card(node2, 'GRID')
    mesh.add_card(node3, 'GRID')
    ############################  
    # CONM2=['CONM2',Eid,RefGid,0,self.inp.mass[i][k],
    #        self.inp.X1[i][k],self.inp.X2[i][k],self.inp.X3[i][k],None,
    #        self.inp.I11[i][k],self.inp.I21[i][k], self.inp.I22[i][k],
    #        self.inp.I31[i][k],self.inp.I32[i][k],self.inp.I33[i][k]]
    conm21 = ['CONM2', 11, 1, 0, config.M / 2,
              config.OFFSET_x, 0., config.OFFSET_z, None,
              1e-5, 0., config.Im, 0., 0., 1e-5
              ]
    conm22 = ['CONM2', 12, 2, 0, config.M / 4,
              0., 0., 0. , None,
              1e-5, 0., 1e-5, 0., 0., 1e-5
              ]
    conm23 = ['CONM2', 13, 3, 0, config.M / 4,
              0., 0., 0. ,None,
              1e-5, 0., 1e-5, 0., 0., 1e-5
              ]

    mesh.add_card(conm21, 'CONM2')
    mesh.add_card(conm22, 'CONM2')
    mesh.add_card(conm23, 'CONM2')
    ############################  
    # mat1 = ['MAT1',id_mat,Em,None,Nu,rho1]
    mat1 = ['MAT1',21, config.E, None,0.3,None]
    mesh.add_card(mat1, 'MAT1')
    ############################  
    # pbeam = ['PBEAM',id_p,id_mat,Aa,I1a,I2a,I12a,Ja]
    if config.BAR:
        pbeam = ['PBAR', 31, 21, config.A, config.I, config.I * 1e-3, config.J]
        mesh.add_card(pbeam, 'PBAR')
    else:
        pbeam = ['PBEAM', 31, 21, config.A, config.I, config.I * 1e-3, 0., config.J]
        mesh.add_card(pbeam, 'PBEAM')

    ############################  
    # cbeam=['CBEAM',EID,PID,GA,GB,X1,X2,X3]
    if config.BAR:
        cbeam1= ['CBAR', 41, 31, 1, 2, 0., 1., 0.]
        cbeam2= ['CBAR', 42, 31, 1, 3, 0., 1., 0.]  
        mesh.add_card(cbeam1, 'CBAR')
        mesh.add_card(cbeam2, 'CBAR')
    else:
        cbeam1= ['CBEAM', 41, 31, 1, 2, 0., 1., 0.]
        cbeam2= ['CBEAM', 42, 31, 1, 3, 0., 1., 0.]
        mesh.add_card(cbeam1, 'CBEAM')
        mesh.add_card(cbeam2, 'CBEAM')

    ############################
    return mesh
# Build Nastran models:2 ends here

# [[file:modelgen.org::*Create nastran files for FE extraction][Create nastran files for FE extraction:1]]
config1 = Config()
mesh1 = build_bdf(config1)
mesh1.write_bdf("./NASTRAN/model1.bdf", size=8, is_double=False, close=True)
# Create nastran files for FE extraction:1 ends here

# [[file:modelgen.org::*Create nastran files for FE extraction][Create nastran files for FE extraction:1]]
config2 = Config(OFFSET_z = -0.1)
mesh2 = build_bdf(config2)
mesh2.write_bdf("./NASTRAN/model2.bdf", size=8, is_double=False, close=True)
# Create nastran files for FE extraction:1 ends here

# [[file:modelgen.org::*Create nastran files for FE extraction][Create nastran files for FE extraction:1]]
config3 = Config(THETA0=30*np.pi/180)
mesh3 = build_bdf(config3)
mesh3.write_bdf("./NASTRAN/model3.bdf", size=8, is_double=False, close=True)
# Create nastran files for FE extraction:1 ends here

# [[file:modelgen.org::*Create nastran files for FE extraction][Create nastran files for FE extraction:1]]
config4 = Config(OFFSET_z = -0.1, THETA0=30*np.pi/180,)
mesh4 = build_bdf(config4)
mesh4.write_bdf("./NASTRAN/model4.bdf", size=8, is_double=False, close=True)
# Create nastran files for FE extraction:1 ends here

# [[file:modelgen.org::*Read and save FEM and FENIAX grid][Read and save FEM and FENIAX grid:1]]
num_models = 5
eigenvalues_list = []
eigenvectors_list = []
for i in range(1, num_models + 1):
    op2 = op2reader.NastranReader(op2name=f"./NASTRAN/simulations_out/Model{i}_103op2.op2")
    op2.readModel()
    eigenvalues = op2.eigenvalues()
    eigenvectors = op2.eigenvectors()
    eigenvalues_list.append(eigenvalues)
    eigenvectors_list.append(eigenvectors)
    # if i == 5: # Model 5
    #     v = eigenvectors.reshape((18,5*6)).T
    # else:
    v = eigenvectors.reshape((18,18)).T
    np.save(f"./FEM/eigenvals_m{i}.npy", eigenvalues)
    np.save(f"./FEM/eigenvecs_m{i}.npy", v)

    id_list,stiffnessMatrix,massMatrix = matrixbuilder.read_pch(f"./NASTRAN/simulations_out/Model{i}_103pch.pch")
    np.save(f"./FEM/Ka_m{i}.npy", stiffnessMatrix)
    np.save(f"./FEM/Ma_m{i}.npy", massMatrix)
# Read and save FEM and FENIAX grid:1 ends here

# [[file:modelgen.org::*Read and save FEM and FENIAX grid][Read and save FEM and FENIAX grid:2]]
for i in range(1, num_models + 1):

    bdf = BDF()
    bdf.read_bdf(f"./NASTRAN/Model{i}_103op2.bdf", validate=False)
    # if i == 5: # Model 5
    #     components = dict(rbeam=[1,21, 22], lbeam=[31, 32])
    # else:
    components = dict(rbeam=[1,2], lbeam=[3])
    model = BuildAsetModel(components, bdf)          
    model.write_grid(f"./FEM/structuralGrid_m{i}")
# Read and save FEM and FENIAX grid:2 ends here
