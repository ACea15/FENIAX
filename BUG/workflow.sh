python P1_buildAsets.py   # Output Nastran ASET and build FENIAX grid 
source P2_runmodal.sh    # Run Nastran modal solution
python P22_buildFEM.py   # build modal solution for FENIAX and Ka/Ma
python P3_dlm.py         # generate dlm model
source P4_rundihedral.sh  # run DMAP for normal of panels
python P5_getdihedral.py  # save normal of panels
python P6_outputgafs.py  # build gafs
source P7_rungafs.sh     # run Nastran to get GAFs
source P8_rungafSteady.sh   # build gafs manoeuvre
python P9_getgafSteady.py    # save gafs manoeuvre
python P10_rogerRFA.py    # run roger approx
