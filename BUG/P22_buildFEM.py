# [[file:modelgeneration.org::*Build modes in OP4, map to ASETs and paraview plot][Build modes in OP4, map to ASETs and paraview plot:1]]
import numpy as np  
import feniax.unastran.op4handler as op4handler
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
