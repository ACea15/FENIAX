#! /bin/bash

python variables.py
python loads_follower.py
python ../../../../pyfem2nl_maintmb.py Simo45_15 confi_simo_foll

python variables.py
python loads_dead.py
python ../../../../pyfem2nl_maintmb.py Simo45_15 confi_simo_dead

#python ../../test_modes.py
