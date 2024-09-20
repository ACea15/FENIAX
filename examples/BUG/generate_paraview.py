import feniax.plotools.nastranvtk.bdfdef as bdfdef
import feniax.unastran.op2reader as op2reader


bdf_file = "/home/acea/projects/FENIAX/examples/BUG/NASTRAN/BUG_103efo.bdf"
op2_file = "/home/acea/projects/FENIAX/examples/BUG/NASTRAN/results_runs/BUG_103efo-02_06_24-18_10_18.op2"

reader = op2reader.NastranReader(op2_file, bdf_file)
bdfdef.vtkModes_fromop2(bdf_file, op2_file, scale = 100., modes2plot=list(range(50)))
