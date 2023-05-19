import numpy as np
from Generators.aero_inp0 import *
NumSurfaces = 29
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


# Fuselage
##############################################################################
# front

p1=[]
x12=[]
p4=[]
x43=[]
nspan=[]
nchord=[]
set1x=[]

##############FUSELAGE#######################
#fs1
p1.append([6.383,0,-0.981])
x12.append(28.531-6.383)
p4.append([6.383,2.793,-0.981])
x43.append(28.531-6.383)
nspan.append(2)
nchord.append(10)
set1x.append([999130, 999131, 999132, 999133, 999134, 999135, 999136, 999137, 999138, 999139, 999140, 999141, 999142, 999143, 999144, 999145, 999146])
#
p1.append([6.383,0.,-0.981])
x12.append(28.531-6.383)
p4.append([6.383,-2.793,-0.981])
x43.append(28.531-6.383)
nspan.append(2)
nchord.append(10)
set1x.append([999130, 999131, 999132, 999133, 999134, 999135, 999136, 999137, 999138, 999139, 999140, 999141, 999142, 999143, 999144, 999145, 999146])
#fs2
p1.append([28.531,0.,-0.981])
x12.append(11.206)
p4.append([28.531,2.793,-0.981])
x43.append(11.206)
nspan.append(2)
nchord.append(5)
set1x.append([999146, 999147, 999148, 999149, 999150, 999151, 999152, 999153])
#
p1.append([28.531,0.,-0.981])
x12.append(11.206)
p4.append([28.531,-2.793,-0.981])
x43.append(11.206)
nspan.append(2)
nchord.append(5)
set1x.append([999146, 999147, 999148, 999149, 999150, 999151, 999152, 999153])
#fs3
p1.append([28.531+11.206,0.,-0.981])
x12.append(59.87-(28.531+11.206))
p4.append([28.531+11.206,2.793,-0.981])
x43.append(59.87-(28.531+11.206))
nspan.append(2)
nchord.append(9)
set1x.append([999153, 999154, 999155, 999156, 999157, 999158, 999159, 999160, 999161, 999162])
#
p1.append([28.531+11.206,0.,-0.981])
x12.append(59.87-(28.531+11.206))
p4.append([28.531+11.206,-2.793,-0.981])
x43.append(59.87-(28.531+11.206))
nspan.append(2)
nchord.append(9)
set1x.append([999153, 999154, 999155, 999156, 999157, 999158, 999159, 999160, 999161, 999162])

##############WING#######################
#ws1
p1.append([28.531,2.793,-0.981])
x12.append(11.206)
p4.append([32.584,8.399,-0.4799])
x43.append(7.748)
nspan.append(6)
nchord.append(10)
set1x.append([999089, 999090, 999091, 999092, 999093, 999094, 999095, 999096, 999097])
#
p1.append([28.531,-2.793,-0.981])
x12.append(11.206)
p4.append([32.584,-8.399,-0.4799])
x43.append(7.748)
nspan.append(6)
nchord.append(10)
set1x.append([999048, 999049, 999050, 999051, 999052, 999053, 999054, 999055, 999056])          
#ws2
p1.append([32.584,8.399,-0.4799])
x12.append(7.748)
p4.append([33.286,9.37,-0.3932])
x43.append(7.373)
nspan.append(1)
nchord.append(10)
set1x.append([999097, 999098, 999099])
#
p1.append([32.584,-8.399,-0.4799])
x12.append(7.748)
p4.append([33.286,-9.37,-0.3932])
x43.append(7.373)
nspan.append(1)
nchord.append(10)
set1x.append([999056, 999057, 999058])
#ws3
p1.append([33.286,9.37,-0.3932])
x12.append(7.373)
p4.append([45.0043,28.1843,1.2867])
x43.append(2.738)
nspan.append(23)
nchord.append(10)
set1x.append([999099, 999100, 999101, 999102, 999103, 999104, 999105, 999106, 999107, 999108, 999109, 999110, 999111, 999112, 999113, 999114, 999115, 999116, 999117, 999118, 999119, 999120, 999121, 999122, 999123, 999124, 999125, 999126, 999127])
#
p1.append([33.286,-9.37,-0.3932])
x12.append(7.373)
p4.append([45.0043,-28.1843,1.2867])
x43.append(2.738)
nspan.append(23)
nchord.append(10)
set1x.append([999058, 999059, 999060, 999061, 999062, 999063, 999064, 999065, 999066, 999067, 999068, 999069, 999070, 999071, 999072, 999073, 999074, 999075, 999076, 999077, 999078, 999079, 999080, 999081, 999082, 999083, 999084, 999085, 999086])

##################WINGTIP#####################
#wT1
p1.append([45.0043,28.1843,1.2867])
x12.append(2.738)
p4.append([48.7,33.99,1.793])
x43.append(1.264)
nspan.append(6)
nchord.append(10)
set1x.append([70999119, 70999120, 70999121, 70999122, 70999123, 70999124, 70999125, 70999126, 70999127, 70999128])
#
p1.append([45.0043,-28.1843,1.2867])
x12.append(2.738)
p4.append([48.7,-33.99,1.793])
x43.append(1.264)
nspan.append(6)
nchord.append(10)
set1x.append([60999078, 60999079, 60999080, 60999081, 60999082, 60999083, 60999084, 60999085, 60999086, 60999087])

##################ENGINES######################

# p1.append([27.088,7.753,-1.821])
# x12.append(6.198)
# p4.append([27.088,8.227,-1.347])
# x43.append(1.264)
# nspan.append(6.198)
# nchord.append(12)
# set1x.append([])
# #
# p1.append([27.088,-7.753,-1.821])
# x12.append(6.198)
# p4.append([27.088,-8.227,-1.347])
# x43.append(1.264)
# nspan.append(6.198)
# nchord.append(12)
# set1x.append([])
#sh
p1.append([27.088,7.753,-2.491])
x12.append(6.198)
p4.append([27.088,10.987,-2.491])
x43.append(6.198)
nspan.append(6)
nchord.append(12)
set1x.append([999098, 999166,9380001])
#
p1.append([27.088,-7.753,-2.491])
x12.append(6.198)
p4.append([27.088,-10.987,-2.491])
x43.append(6.198)
nspan.append(6)
nchord.append(12)
set1x.append([999057, 999165,9370001])
#sv
p1.append([27.088,9.37,-0.874])
x12.append(6.198)
p4.append([27.088,9.37,-4.107])
x43.append(6.198)
nspan.append(6)
nchord.append(12)
set1x.append([999098, 999166,9380001])
#
p1.append([27.088,-9.37,-0.874])
x12.append(6.198)
p4.append([27.088,-9.37,-4.107])
x43.append(6.198)
nspan.append(6)
nchord.append(12)
set1x.append([999057, 999165,9370001])
#spylon
p1.append([30.062,9.37,-0.393])
x12.append(6.198)
p4.append([30.062,9.37,-0.874])
x43.append(6.198)
nspan.append(2)
nchord.append(4)
set1x.append([999098, 999166,9380001])
#
p1.append([30.062,-9.37,-0.393])
x12.append(6.198)
p4.append([30.062,-9.37,-0.874])
x43.append(6.198)
nspan.append(2)
nchord.append(4)
set1x.append([999057, 999165,9370001])
#######################################TAILPLANE############################
#htp
p1.append([63.299,0.925,1.585])
x12.append(5.028)
p4.append([69.12,9.702,2.599])
x43.append(2.019)
nspan.append(10)
nchord.append(6)
set1x.append([999004, 999021, 999022, 999023, 999024, 999025, 999026, 999027, 999028, 999029])
#
p1.append([63.299,-0.925,1.585])
x12.append(5.028)
p4.append([69.12,-9.702,2.599])
x43.append(2.019)
nspan.append(10)
nchord.append(6)
set1x.append([999004, 999008, 999009, 999010, 999011, 999012, 999013, 999014, 999015, 999016])
#htp_stub
p1.append([59.869,0.,1.585])
x12.append(8.458)
p4.append([59.869,0.925,1.585])
x43.append(8.458)
nspan.append(1)
nchord.append(11)
set1x.append([999000, 999001, 999002, 999003, 999004, 999005, 999161, 999162, 999163, 999164])
#
p1.append([59.869,0.,1.585])
x12.append(8.458)
p4.append([59.869,-0.925,1.585])
x43.append(8.458)
nspan.append(1)
nchord.append(11)
set1x.append([999000, 999001, 999002, 999003, 999004, 999005, 999161, 999162, 999163, 999164])
#htp_stub_aft
p1.append([68.327,0.,1.585])
x12.append(1.962)
p4.append([68.327,0.925,1.585])
x43.append(1.962)
nspan.append(1)
nchord.append(3)
set1x.append([999005, 999006, 999007])
#
p1.append([68.327,0.,1.585])
x12.append(1.962)
p4.append([68.327,-0.925,1.585])
x43.append(1.962)
nspan.append(1)
nchord.append(3)
set1x.append([999005, 999006, 999007])
#vtp
#s1
p1.append([60.44,0.,0.])
x12.append(7.887)
p4.append([60.44,0.,1.585])
x43.append(7.887)
nspan.append(1)
nchord.append(8)
set1x.append([999167, 999001, 999005])
#s2
p1.append([60.44,0.,1.585])
x12.append(7.887)
p4.append([60.44,0.,2.82])
x43.append(7.887)
nspan.append(1)
nchord.append(8)
set1x.append([999167, 999001, 999005])
#s3
p1.append([60.44,0.,2.82])
x12.append(7.887)
p4.append([68.872,0.,11.52])
x43.append(3.1)
nspan.append(8)
nchord.append(8)
set1x.append([999164, 999167, 999168, 999169, 999170, 999171, 999172, 999173, 999174, 999175, 999176, 999177])



          
          
def dlm(NumSurfaces,paero1,caero1,aelist,set1,spline67,p1,x12,p4,x43,nspan,nchord,set1x):
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
        aelist[i]['elements'] = range(caero1[i]['eid'],caero1[i]['eid']+
                                      caero1[i]['nspan']*caero1[i]['nchord'])
        set1[i]['sid'] = aelist[i]['sid']
        set1[i]['ids'] = set1x[i]
        sp_num = 6
        spline67[i]=[sp_num,'SPLINE%s'%sp_num,aelist[i]['sid'],caero1[i]['eid'],aelist[i]['sid'],None,
                         set1[i]['sid']]  # EID,CAERO,AELIST,NONE,SETG


dlm(NumSurfaces,paero1,caero1,aelist,set1,spline67,p1,x12,p4,x43,nspan,nchord,set1x)

