# [[file:modelgeneration.org::parameters_modal0][parameters_modal0]]
import feniax.unastran.op4handler as op4handler
from pyNastran.bdf.bdf import BDF
from feniax.unastran.asetbuilder import BuildAsetModel
import feniax.plotools.nastranvtk.bdfdef as bdfdef
import numpy as np

num_modes = 100
sol = "eao" # {c,e}{a,f}{o,p}
WRITE_GRID= True
WRITE_ASETS= True
ASET_MODEL_FULL = False  # ASETS at every CONM2 of half the size (full model gives
                         # problems in 103 and might be too large any way)
# parameters_modal0 ends here

# [[file:modelgeneration.org::*ASETs generation][ASETs generation:1]]
bdf = BDF()#debug=False)
bdf.read_bdf("./NASTRAN/BUG103.bdf", validate=False)

# bdf_conm2 = BDF()
# conm2_ids = list(range(314, 345)) + [376, 377, 378]
# for cmi in conm2_ids:
#     conm2 = bdf.masses[cmi]
#     bdf_conm2.add_conm2(conm2.eid, conm2.nid, conm2.mass, conm2.cid, conm2.X, conm2.I)

# bdf_conm2.write_bdf("./Parts/MTOW_FUEL_RWBOXmod.bdf")

######## BUILD STRUCTURAL MODEL ##############

if ASET_MODEL_FULL:                         
    # Initial model
    components_ids = dict()
    components_ids['FusWing'] = [2000]
    if sol[0] == "c": # clamped model
        components_ids['FusBack'] = [1006, 1007, 1008, 1009]
    elif sol[0] == "e": # free model
        components_ids['FusBack'] = [1005, 1006, 1007, 1008, 1009]
    components_ids['FusFront'] = [1004, 1003, 1002, 1001, 1000]
    components_ids['RWing'] = list(range(2001, 2053))
    components_ids['LWing'] = list(range(10002001, 10002053))
    components_ids['FusTail'] = [1010]
    components_ids['VTP'] = list(range(3000, 3010))
    components_ids['HTP'] = [4000]
    components_ids['VTPTail'] = [3010]
    components_ids['RHTP'] = list(range(4001, 4014))
    components_ids['LHTP'] = list(range(10004001, 10004014))
    if sol[0] == "c": # clamped model
        model_asets = BuildAsetModel(components_ids, bdf, clamped_node=1005)
    elif sol[0] == "e": # free model
        model_asets = BuildAsetModel(components_ids, bdf)          

    if WRITE_ASETS:
        if sol[0] == "c": # clamped model
            model_asets.write_asets("./NASTRAN/Asets/asets_clamped.bdf")
        elif sol[0] == "e": # free model
            model_asets.write_asets("./NASTRAN/Asets/asets_free.bdf")

    if WRITE_GRID:
        model_asets.write_grid(f"./FEM/structuralGridfull_{sol[:-1]}")

else:
    # Initial model removing some ASET nodes along the wing
    components_ids = dict()
    #components_ids['FusWing'] = [2000]
    if sol[0] == "c": # clamped model
        components_ids['FusBack'] = [1006, 1007, 1008, 1009]
    elif sol[0] == "e": # free model
        components_ids['FusBack'] = [1005, 1006, 1007, 1008, 1009]
    components_ids['FusFront'] = [1004, 1003, 1002, 1001, 1000]
    components_ids['RWing'] = [2001, 2003, 2005, 2008, 2010] + list(range(2012, 2053, 2))
    components_ids['LWing'] = ([10002001, 10002003, 10002005, 10002008, 10002010] +
                               list(range(10002012, 10002053, 2)))
    components_ids['FusTail'] = [1010]
    components_ids['VTP'] = list(range(3000+1, 3010-1))
    components_ids['HTP'] = [4000]
    components_ids['VTPTail'] = [3010]
    components_ids['RHTP'] = list(range(4001, 4014))
    components_ids['LHTP'] = list(range(10004001, 10004014))
    if sol[0] == "c": # clamped model
        model_asets = BuildAsetModel(components_ids, bdf, clamped_node=1005)
    elif sol[0] == "e": # free model
        model_asets = BuildAsetModel(components_ids, bdf)

    if WRITE_ASETS:          
        if sol[0] == "c": # clamped model
            model_asets.write_asets("./NASTRAN/Asets/asets_clamped_reduced.bdf")
        elif sol[0] == "e": # free model
            model_asets.write_asets("./NASTRAN/Asets/asets_free_reduced.bdf")              
    if WRITE_GRID:
        model_asets.write_grid(f"./FEM/structuralGrid_{sol[:-1]}")
# ASETs generation:1 ends here

# [[file:modelgeneration.org::*Build modes in OP4, map to ASETs and paraview plot][Build modes in OP4, map to ASETs and paraview plot:1]]
eigs, modes = op4handler.write_op4modes(f"./NASTRAN/simulations_out/BUG103_{sol}.bdf",
                                        num_modes,
                                        op4_name=f"./NASTRAN/data_out/Phi{num_modes}_{sol}",
                                        return_modes=True)
bdf_file = f"./NASTRAN/BUG103_{sol}.bdf"
bdf = BDF()
bdf.read_bdf(bdf_file)
node_ids = bdf.node_ids
assert modes.shape[1] == len(node_ids), "the modes size does not match the node_ids"
sorted_nodeids = sorted(node_ids)
asets_ids = bdf.asets[0].node_ids
asets_ids_sorted = sorted(asets_ids)
asets_idsfull = np.array([sorted_nodeids.index(ai) for ai in asets_ids_sorted])
asets_indexes = np.hstack([[6*i + j for j in range(6)] for i in asets_idsfull])
#modes4simulations = modes[asets_indexes, :]
SAVE = False
if SAVE:
    np.save(f"./FEM/eigenvecs_{sol}{num_modes}.npy", modes4simulations.T)
    np.save(f"./FEM/eigenvals_{sol}{num_modes}.npy", eigs)
# Build modes in OP4, map to ASETs and paraview plot:1 ends here

# [[file:modelgeneration.org::*Build modes in OP4, map to ASETs and paraview plot][Build modes in OP4, map to ASETs and paraview plot:2]]
modes = op4handler.read_data(f"./NASTRAN/data_out/Phi{num_modes}_{sol}.op4", "PHG")
bdf_file = f"./NASTRAN/BUG103_{sol}.bdf"
bdf = BDF()
bdf.read_bdf(bdf_file)
node_ids = bdf.node_ids
assert len(modes)/6 == len(node_ids), "the modes size does not match the node_ids"
sorted_nodeids = sorted(node_ids)
asets_ids = bdf.asets[0].node_ids
asets_ids_sorted = sorted(asets_ids)
asets_idsfull = np.array([sorted_nodeids.index(ai) for ai in asets_ids_sorted])
asets_indexes = np.hstack([[6*i + j for j in range(6)] for i in asets_idsfull])
modes4simulations = modes[asets_indexes, :]
SAVE = True
if SAVE:
    np.save(f"./FEM/eigenvecs_{sol}{num_modes}.npy", modes4simulations)
    np.save(f"./FEM/eigenvals_{sol}{num_modes}.npy", eigs)
# Build modes in OP4, map to ASETs and paraview plot:2 ends here

# [[file:modelgeneration.org::*Plot VTK modes][Plot VTK modes:1]]
op2_file = f"./NASTRAN/simulations_out/BUG103_{sol}.op2" 
bdf_file = f"./NASTRAN/BUG103_{sol}.bdf"   
bdfdef.vtkModes_fromop2(bdf_file,
                        op2_file,
                        scale = 100.,
                        modes2plot=list(range(num_modes)),
                        write_path=f"./paraview/Modes_{sol}/",
                        plot_ref=False)

#bdfdef.vtkRef("./NASTRAN/Paraview/BUG_103cao.bdf")  # write full FE paraview
# Plot VTK modes:1 ends here

# [[file:modelgeneration.org::*Read pch][Read pch:1]]
import feniax.unastran.matrixbuilder as matrixbuilder
soli = sol[:-1]
id_list,stiffnessMatrix,massMatrix = matrixbuilder.read_pch(f"./NASTRAN/simulations_out/BUG103_{soli}p.pch")
SAVE_FE = True
if SAVE_FE:
    np.save(f"./FEM/Ka_{soli}.npy", stiffnessMatrix)
    np.save(f"./FEM/Ma_{soli}.npy", massMatrix)
try:
    assert len(asets_indexes) == len(stiffnessMatrix), "the FE matrices size does not match the indexes used to build the aset modes from the full set"
except NameError:
    print("Careful, no aset-matrix sizes checked")
# Read pch:1 ends here
