#! /bin/bash

#python variables.py
#python loads_follower.py
#python ../../../../pyfem2nl_maint.py Simo45_15 confi_simo_foll

python variables.py
python loads_dead.py
feminas_main.py Simo45_15 confi_simo_dead

#python ../../test_modes.py
