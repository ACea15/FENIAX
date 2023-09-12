import numpy as np

def wb2w(point_le, x_te, z_bottom, pct2le, pct2te):
    """ Wing-box to wing"""

    x = x_te - point_le[0]
    pct_wb = (pct2te - pct2le)
    chord = x / pct_wb
    LE = point_le[0] - chord * pct2le
    point_LE = [LE, point_le[1], (point_le[2] +
                                  z_bottom) / 2]
    return point_LE, chord

NumSurfaces = 9
aero = {}
caero1 = [{} for i in range(NumSurfaces)]
paero1 = [{} for i in range(NumSurfaces)]
spline67 = [[] for i in range(NumSurfaces)]
aelist = [{} for i in range(NumSurfaces)]
set1 = [{} for i in range(NumSurfaces)]


aero['velocity'] = 1.
aero['cref'] = 7.271
aero['rho_ref'] = 1.
#aero['s_ref'] = 361.6
#aero['b_ref'] = 58.0
#aero['X_ref'] = 36.3495

# Initialise
p1=[]
x12=[]
p4=[]
x43=[]
nspan=[]
nchord=[]
set1x=[]
components = []
##############FUSELAGE#######################
###
##############InnerWingR#######################
components.append("InnerWingR")
zcorr = 0.40135 / 0.3795
point_LE1, l1 = wb2w(point_le=[3.94, 0, zcorr * 0.231],
                     x_te=8.49,
                     z_bottom=-0.49 * zcorr,
                     pct2le=0.15,
                     pct2te=0.6)
point_LE2, l2 = wb2w(point_le=[3.94, 2.793, 0.231 * zcorr],
                     x_te=8.49,
                     z_bottom=-0.49 * zcorr,
                     pct2le=0.15,
                     pct2te=0.6)

p1.append(point_LE1)
x12.append(l1)
p4.append(point_LE2)
x43.append(l2)
nspan.append(4)
nchord.append(10)
#set1x.append([10040069, 10040066, 10040063, 10040060, 10040000])
set1x.append([999023, 999022, 999021, 999020, 999000])
##################OuterWingR#####################
components.append("OuterWingR")
point_LE1, l1 = wb2w(point_le=[3.94, 2.793, 0.231 * zcorr],
                     x_te=8.49,
                     z_bottom=-0.49 * zcorr,
                     pct2le=0.15,
                     pct2te=0.6)
point_LE2, l2 = wb2w(point_le=[19.23, 28.8, -0.304 * zcorr],
                     x_te=20.47,
                     z_bottom=-0.455 * zcorr,
                     pct2le=0.15,
                     pct2te=0.6)
p1.append(point_LE1)
x12.append(l1)
p4.append(point_LE2)
x43.append(l2)
nspan.append(19)
nchord.append(10)
set1x.append([999000,
              999001,
              999002,
              999003,
              999004,
              999005,
              999006,
              999007,
              999008,
              999009,
              999010,
              999011,
              999012,
              999013,
              999014,
              999015,
              999016,
              999017,
              999018,
              999019])

##############InnerWingL#######################
components.append("InnerWingL")
point_LE1, l1 = wb2w(point_le=[3.94, 0, 0.231 * zcorr],
                     x_te=8.49,
                     z_bottom=-0.49 * zcorr,
                     pct2le=0.15,
                     pct2te=0.6)
point_LE2, l2 = wb2w(point_le=[3.94, -2.793, 0.231 * zcorr],
                     x_te=8.49,
                     z_bottom=-0.49 * zcorr,
                     pct2le=0.15,
                     pct2te=0.6)

p1.append(point_LE1)
x12.append(l1)
p4.append(point_LE2)
x43.append(l2)
nspan.append(4)
nchord.append(10)
#set1x.append([10040069, 10040066, 10040063, 10040060, 10040000])
set1x.append([999023, 999057, 999056, 999055, 999035])
##################OuterWingL#####################
components.append("OuterWingL")
point_LE1, l1 = wb2w(point_le=[3.94, -2.793, 0.231 * zcorr],
                     x_te=8.49,
                     z_bottom=-0.49,
                     pct2le=0.15,
                     pct2te=0.6)
point_LE2, l2 = wb2w(point_le=[19.23, -28.8, -0.304 * zcorr],
                     x_te=20.47,
                     z_bottom=-0.455,
                     pct2le=0.15,
                     pct2te=0.6)
p1.append(point_LE1)
x12.append(l1)
p4.append(point_LE2)
x43.append(l2)
nspan.append(19)
nchord.append(10)
set1x.append([999035,
              999036,
              999037,
              999038,
              999039,
              999040,
              999041,
              999042,
              999043,
              999044,
              999045,
              999046,
              999047,
              999048,
              999049,
              999050,
              999051,
              999052,
              999053,
              999054])

##################ENGINES######################
###
#####################TAILPLANE######
#hstabilizerInnerR
components.append("hstabilizerInnerR")
point_LE1, l1 = wb2w(point_le=[36.363, 0, 3.225],
                     x_te=40.236,
                     z_bottom=2.875,
                     pct2le=0.15,
                     pct2te=0.6)
point_LE2, l2 = wb2w(point_le=[36.363, 0.5, 3.225],
                     x_te=40.236,
                     z_bottom=2.875,
                     pct2le=0.15,
                     pct2te=0.6)

p1.append(point_LE1)
x12.append(l1)
p4.append(point_LE2)
x43.append(l2)
nspan.append(1)
nchord.append(10)
set1x.append([999034, 999024])
#hstabilizerOuterR
components.append("hstabilizerOuterR")
point_LE1, l1 = wb2w(point_le=[36.363, 0.5, 3.225],
                     x_te=40.236,
                     z_bottom=2.875,
                     pct2le=0.15,
                     pct2te=0.6)
point_LE2, l2 = wb2w(point_le=[42.377, 8.9, 3.96],
                     x_te=43.781,
                     z_bottom=3.82,
                     pct2le=0.15,
                     pct2te=0.6)

p1.append(point_LE1)
x12.append(l1)
p4.append(point_LE2)
x43.append(l2)
nspan.append(9)
nchord.append(10)
set1x.append([999024,
              999025,
              999026,
              999027,
              999028,
              999029,
              999030,
              999031,
              999032,
              999033])
#hstabilizerInnerL
components.append("hstabilizerInnerL")
point_LE1, l1 = wb2w(point_le=[36.363, 0, 3.225],
                     x_te=40.236,
                     z_bottom=2.875,
                     pct2le=0.15,
                     pct2te=0.6)
point_LE2, l2 = wb2w(point_le=[36.363, -0.5, 3.225],
                     x_te=40.236,
                     z_bottom=2.875,
                     pct2le=0.15,
                     pct2te=0.6)

p1.append(point_LE1)
x12.append(l1)
p4.append(point_LE2)
x43.append(l2)
nspan.append(1)
nchord.append(10)
set1x.append([999034,
              999068])

#hstabilizerOuterL
components.append("hstabilizerOuterL")
point_LE1, l1 = wb2w(point_le=[36.363, -0.5, 3.225],
                     x_te=40.236,
                     z_bottom=2.875,
                     pct2le=0.15,
                     pct2te=0.6)
point_LE2, l2 = wb2w(point_le=[42.377, -8.9, 3.96],
                     x_te=43.781,
                     z_bottom=3.82,
                     pct2le=0.15,
                     pct2te=0.6)

p1.append(point_LE1)
x12.append(l1)
p4.append(point_LE2)
x43.append(l2)
nspan.append(9)
nchord.append(10)
set1x.append([999058,
              999059,
              999060,
              999061,
              999062,
              999063,
              999064,
              999065,
              999066,
              999067])
#vstabilizer
# WARNING: y-midplane not implemented
components.append("vstabilizer")
point_LE1, l1 = wb2w(point_le=[34.54, 0., 3.3],
                     x_te=39.983,
                     z_bottom=3.3,
                     pct2le=0.15,
                     pct2te=0.6)
point_LE2, l2 = wb2w(point_le=[42.236, 0., 13],
                     x_te=44.227,
                     z_bottom=13,
                     pct2le=0.15,
                     pct2te=0.6)

p1.append(point_LE1)
x12.append(l1)
p4.append(point_LE2)
x43.append(l2)
nspan.append(9)
nchord.append(10)
set1x.append([999068,
              999069,
              999070,
              999071,
              999072,
              999073,
              999074,
              999075,
              999076,
              999077])


def dlm(NumSurfaces,paero1,caero1,aelist,set1,spline67,p1,x12,p4,x43,nspan,nchord,set1x):
    npanels = 0
    for i in range(NumSurfaces):
        paero1[i]['pid'] = 950+i
        caero1[i]['eid'] = 10000000 +i*10000
        caero1[i]['pid'] = paero1[0]['pid'] # PAERO1
        caero1[i]['igroup'] = 1  # Group number
        #caero1[i]['igid'] = 1  # Group number
        caero1[i]['p1'] = p1[i] #[6.383,2.793,-0.981]   # 1-4|2-3 parallelogram points
        caero1[i]['x12'] = x12[i] #28.531-6.383# Distance from 1 to 2
        caero1[i]['p4'] = p4[i]#[6.383,-2.793,-0.981]   #  "
        caero1[i]['x43'] = x43[i]#28.531-6.383  # "
        caero1[i]['nspan'] = nspan[i]#4  # boxes across y-direction
        caero1[i]['nchord'] = nchord[i]#10  # boxes across x-direction
        aelist[i]['sid'] = 666000 +i
        aelist[i]['elements'] = list(range(caero1[i]['eid'],caero1[i]['eid']+
                                      caero1[i]['nspan']*caero1[i]['nchord']))
        set1[i]['sid'] = aelist[i]['sid']
        set1[i]['ids'] = set1x[i]
        sp_num = 6
        spline67[i]=['SPLINE%s'%sp_num,aelist[i]['sid'],caero1[i]['eid'],aelist[i]['sid'],None,
                         set1[i]['sid']]  # EID,CAERO,AELIST,NONE,SETG
        npanels +=  nspan[i]*nchord[i]
    return npanels

    
npanels = dlm(NumSurfaces,paero1,caero1,aelist,set1,spline67,p1,x12,p4,x43,nspan,nchord,set1x)

import PostProcessing.panels as panels
from pyNastran.bdf.bdf import BDF
################################################################################
# AERODYNAMICS
################################################################################
machs = [0.8]
reduced_freqs = np.linspace(1e-9,1, 20)

model=BDF(debug=True,log=None)
for i in range(NumSurfaces):
    model.add_aelist(**aelist[i], comment=components[i])
    model.add_set1(**set1[i], comment=components[i])
    model.add_caero1(**caero1[i], comment=components[i])
    model.add_paero1(**paero1[i], comment=components[i])
    model.add_card(spline67[i], 'SPLINE6', comment=components[i])
model.add_mkaero1(machs,reduced_freqs)
model.write_bdf("./aero.bdf")


# from pyNastran.op2.export_to_vtk import export_to_vtk_filename
# export_to_vtk_filename(bdf_filename, op2_filename, vtk_filename,
#                             debug=False, log=None)

grid = panels.caero2grid(components, caero1)
panels.build_gridmesh(grid, 'dlm_mesh')
