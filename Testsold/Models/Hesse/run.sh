#! /bin/bash
python variables.py
python loads2d.py
feminas_main.py Hesse_25 confi2d

python variables.py
python loads3d.py
feminas_main.py Hesse_25 confi3d
