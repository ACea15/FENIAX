import fem4inas.plotools.nastranvtk.bdfdef as bdfdef


bdf_file = "./NASTRAN/BUG_103efo.bdf"
op2_file = "./NASTRAN/BUG_103efo.op2"
bdfdef.vtkModes_fromop2(bdf_file, op2_file, scale = 100., modes2plot=None)
