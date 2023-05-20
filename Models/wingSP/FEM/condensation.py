import numpy as np
import Condensation.condensation as cond
reload(cond)
import Utils.FEM_MatrixBuilder as fmb

#c1=cond.Condense('./Kaa.npy','./Maa.npy',30)
#K=fmb.readOP4_matrices(readKM=['../NASTRAN/Ka.op4'],nameKM=['KAA'],saveKM=['./Kaa.npy'])
#M=fmb.readOP4_matrices(readKM=['../NASTRAN/Ma.op4'],nameKM=['MAA'],saveKM=['./Maa.npy'])
lines_new=[]
count =0
#b=[]
with open('../NASTRAN/Ma.op4') as fp:
    for line in fp:
        if '1MAA' in line and '1P' in line:
            #b.append(line)
            count +=1
        if count>0:    
            lines_new.append(line)
        if line == ' 1.00000000000000000000E+00\n':
            count-=1
with open('../NASTRAN/Ma.op4','w+') as fp:
    for li in lines_new:
        fp.write(li)

c2=cond.Condense('../NASTRAN/Ka.op4','../NASTRAN/Ma.op4',23)
#c3=cond.Condense('../NASTRAN/Ka.op4','./Maa.npy',30)
c2.cond_iter(1)

c2.cond_classic(c2.Dg[0])

nK = len(c2.Kf)
for i in range(nK):
    if np.allclose(c2.Kf[i,:],np.zeros((nK,1)),rtol=1e-05, atol=1e-08):
        print i

wg = np.sqrt(c2.Dg)/(2*np.pi)
wi = np.sqrt(c2.D_i)/(2*np.pi)
ww0 = np.sqrt(c2.D_w0)/(2*np.pi)
# np.save('./Ki.npy',c2.K_i)
# np.save('./Mi.npy',c2.M_i)

# np.save('./Ka.npy',c2.Ka_g)
# np.save('./Ma.npy',c2.Ma_g)

# np.save('./Kw0.npy',c2.Ka_w0)
# np.save('./Mw0.npy',c2.Ma_w0)

#np.save('./K.npy',c2.Kf)
#np.save('./M.npy',c2.Mf)

