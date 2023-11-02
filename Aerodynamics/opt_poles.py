import itertools
from scipy.optimize import least_squares
import numpy as np
import Utils.common as common
#import rfa
import Aerodynamics.rfa as rfa
import pdb

def combine_loops(init0,n,final):

    def combine_list(l1,l2):
        u=[]
        for i in range(len(l2)):
            u.append(l1+[l2[i]])
        return u

    u = [init0]+[[] for i in range(n-1)]
    for i in range(n-1):
        for j in range(len(u[i])):
            u[i+1] += combine_list(u[i][j],range(u[i][j][-1]+1,final+i))

    return u[-1]

def optimize_poles(poles,reduced_freq,RFA_Method,aero_matrices_real,aero_matrices_imag,err_type,sol_step=0.,rfa0=0):

    k_matrix = rfa.frequency_matrix(poles,*(reduced_freq,RFA_Method))
    RFA_mat = rfa.RFA_matrix(poles,*(k_matrix,reduced_freq,aero_matrices_real,aero_matrices_imag,RFA_Method,rfa0))
    err = rfa.err_RFA(poles,*(RFA_mat,RFA_Method,reduced_freq,aero_matrices_real,aero_matrices_imag,err_type,sol_step))
    return err

def y_poles2(poles,reduced_freq,RFA_Method,aero_matrices_real,aero_matrices_imag,aero_matrices,err_type,sol_step=0.,rfa0=0):

    k_matrix = rfa.frequency_matrix(poles,*(reduced_freq,RFA_Method))
    k_matrix2 = rfa.frequency_matrix2(poles,*(reduced_freq,np.shape(aero_matrices_real)[1],RFA_Method))
    RFA_mat = rfa.RFA_matrix(poles,*(k_matrix,reduced_freq,aero_matrices_real,aero_matrices_imag,RFA_Method,rfa0))
    RFA_mat2 = rfa.RFA_matrix2(poles,*(k_matrix2,reduced_freq,aero_matrices_real,aero_matrices_imag,RFA_Method,rfa0))
    err = rfa.err_RFA(poles,*(RFA_mat,RFA_Method,reduced_freq,aero_matrices_real,aero_matrices_imag,err_type,sol_step))
    err2 = rfa.err_RFA(poles,*(RFA_mat2,RFA_Method,reduced_freq,aero_matrices_real,aero_matrices_imag,err_type,sol_step))
    return k_matrix,k_matrix2,RFA_mat,RFA_mat2,err,err2

def y_poles(poles,reduced_freq,RFA_Method,aero_matrices_real,aero_matrices_imag,aero_matrices,err_type,sol_step=0.,rfa0=0):

    k_matrix = rfa.frequency_matrix(poles,*(reduced_freq,RFA_Method))
    RFA_mat = rfa.RFA_matrix(poles,*(k_matrix,reduced_freq,aero_matrices_real,aero_matrices_imag,RFA_Method,rfa0))
    err = rfa.err_RFA(poles,*(RFA_mat,RFA_Method,reduced_freq,aero_matrices_real,aero_matrices_imag,err_type,sol_step))
    return k_matrix,RFA_mat,err


def optimize_least_squares(fun,x0,*args):
    res = least_squares(fun,x0,args=args, verbose=1)
    return res

def optimize_brute(fun_opt,fun_err,reduced_freq,RFA_Method,aero_matrices_real,aero_matrices_imag,kmax,kstep,NumPoles,err_type,sol_step,rfa0):
    #pdb.set_trace()
    li = np.arange(kmax,0.,-kstep)
    nl = len(li)
    ll= []
    err=[]
    trak ={}
    k=0

    init0 = [[i] for i in range(nl-(NumPoles-1))]
    final = nl-NumPoles+2
    u = combine_loops(init0,NumPoles,final)
    for ui in u:
        pol = np.round([li[ui[pi]] for pi in range(NumPoles)],4)
        e = fun_opt(pol,reduced_freq,RFA_Method,aero_matrices_real,aero_matrices_imag,err_type,sol_step,rfa0)
        #err.append(np.linalg.norm(e))
        err.append(fun_err(e,err_type))
        #print pol
        trak[k] = pol
        k+=1
    return err,trak


def min_brute(*args):
    err,trak = optimize_brute(optimize_poles,common.err_tensor,*args)
    err_min = min(err)
    poles_opt = trak[err.index(err_min)]
    return err_min,poles_opt
#reduced_freq,RFA_Method,aero_matrices_real,aero_matrices_imag,err_type,sol_step=0.,rfa0=0
def min_least_squares(x0,*args):
    #print rfa0,'min_least_squares'
    res = optimize_least_squares(optimize_poles,x0,*args)
    return common.err_tensor(res.fun,args[-3]),res.x

# nl=6
# NumPoles=0

# u=[]
# for i in range(nl-(NumPoles-1)):
#     for i2 in range(i+1,nl-(NumPoles-2)):
#         for i3 in range(i2+1,nl-(NumPoles-3)):
#             for i4 in range(i3+1,nl-(NumPoles-4)):
#                 u.append([i,i2,i3,i4])




# init0 = [[i] for i in range(nl-(NumPoles-1))]
# n = 4
# final = nl-NumPoles+2
# u2=combine_loops(init0,n,final)
# print np.asarray(u)-np.asarray(u2)


# print min(err)
# print err.index(min(err))
# poles_opt = trak[err.index(min(err))]
# print poles_opt
