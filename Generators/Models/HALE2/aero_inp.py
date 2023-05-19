import numpy as np
from Generators.aero_inp0 import *
NumSurfaces = 1
aero = {}
caero1 = [{} for i in range(NumSurfaces)]
paero1 = [{} for i in range(NumSurfaces)]
spline67 = [[] for i in range(NumSurfaces)]
aelist = [{} for i in range(NumSurfaces)]
set1 = [{} for i in range(NumSurfaces)]

aero['velocity'] = 25.
aero['cref'] = 1.
aero['rho_ref'] = 0.0899
for i in range(NumSurfaces):

    paero1[i]['pid'] = 1000+1+1

    caero1[i]['eid'] = i+1
    caero1[i]['pid'] = paero1[i]['pid'] # PAERO1
    caero1[i]['igroup'] = 1  # Group number
    #caero1[i]['igid'] = 1  # Group number
    caero1[i]['p1'] = [0.5,0.,0.]   # 1-4|2-3 parallelogram points
    caero1[i]['x12'] = 1.# Distance from 1 to 2
    caero1[i]['p4'] = [0.5,16.,0.]   #  "
    caero1[i]['x43'] = 1.  # "
    caero1[i]['nspan'] = 50  # boxes across y-direction
    caero1[i]['nchord'] = 8  # boxes across x-direction

    aelist[i]['sid'] = i+1
    aelist[i]['elements'] = range(caero1[i]['eid'],caero1[i]['eid']+
                                  caero1[i]['nspan']*caero1[i]['nchord'])

    set1[i]['sid'] = 100+i
    set1[i]['ids'] = range(2,1+26)

    sp_num = 6
    spline67[i]=[sp_num,'SPLINE%s'%sp_num,i+1,caero1[i]['eid'],aelist[i]['sid'],None,
                     set1[i]['sid']]  # EID,CAERO,AELIST,NONE,SETG
