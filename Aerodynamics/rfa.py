import numpy as np
import pdb
from intrinsic.functions import my_timer
import itertools
import functools
import Utils.common as common

def RFA_freq(poles,ki,RFA_mat,RFA_Method):

    RFA_ki = RFA_mat[0] + RFA_mat[1]*1j*ki - RFA_mat[2]*ki**2
    if RFA_Method == 'R' or RFA_Method == 'r':
        for i in range(len(poles)):
            RFA_ki = RFA_ki + RFA_mat[i+3]*ki*1j/(poles[i]+ki*1j)
    elif RFA_Method == 'E' or RFA_Method == 'e':
        for i in range(len(poles)):
            RFA_ki = RFA_ki + RFA_mat[i+3]/(poles[i]+ki*1j)
    return RFA_ki

def err_RFA(poles,RFA_mat,RFA_Method,reduced_freq,aero_matrices_real,aero_matrices_imag,err_type,sol_step=0):
    #print sol_step
    err = []
    for i in range(len(reduced_freq)):
        X = RFA_freq(poles,reduced_freq[i],RFA_mat,RFA_Method)
        Dr = common.diff_tensor(aero_matrices_real[i],X.real)
        Di = common.diff_tensor(aero_matrices_imag[i],X.imag)
        err.append(common.err_tensor(Dr,err_type))
        err.append(common.err_tensor(Di,err_type))
    err = np.asarray(err)
    if sol_step>0:
        for i in range(len(poles)-1):
            #print poles[i+1],poles[i]
            if poles[i]-poles[i+1]+0.00001<sol_step:
                #print poles[i]-poles[i+1],'ff'
                #print sol_step
                err = err+ np.ones(len(err))

    return err


def frequency_matrix(poles,reduced_freq,RFA_Method):

    num_reduced_freq = len(reduced_freq)
    npoles = len(poles)
    k_matrix=np.zeros((num_reduced_freq*2,3+npoles))
    ###########################################################################################################################################
    #Roger's Method
    ###########################################################################################################################################
    if RFA_Method == 'R' or RFA_Method == 'r':
        for row in range (0,num_reduced_freq*2,2):
            for column in range (npoles+3):
                if column==0:
                    k_matrix[row,column]=1.
                    k_matrix[row+1,column]=0.
                if column==1:
                    k_matrix[row,column]=0.
                    k_matrix[row+1,column]=reduced_freq[row/2]
                if column==2:
                    k_matrix[row,column]=-reduced_freq[row/2]**2
                    k_matrix[row+1,column]=0.
                if column>=3:
                    k_matrix[row,column]=(reduced_freq[row/2]**2)/(reduced_freq[row/2]**2+poles[column-3]**2)
                    k_matrix[row+1,column]=(reduced_freq[row/2]*poles[column-3])/(reduced_freq[row/2]**2+poles[column-3]**2)

    ###########################################################################################################################################
    #Eversman's Method
    ###########################################################################################################################################
    if RFA_Method == 'E' or RFA_Method == 'e':
        for row in range (0,num_reduced_freq*2,2):
            for column in range (npoles+3):
                if column==0:
                    k_matrix[row,column]=1.
                    k_matrix[row+1,column]=0.
                if column==1:
                    k_matrix[row,column]=0.
                    k_matrix[row+1,column]=reduced_freq[row/2]
                if column==2:
                    k_matrix[row,column]=-reduced_freq[row/2]**2
                    k_matrix[row+1,column]=0.
                if column>=3:
                    k_matrix[row,column]=(poles[column-3])/(reduced_freq[row/2]**2+poles[column-3]**2)
                    k_matrix[row+1,column]=(-reduced_freq[row/2])/(reduced_freq[row/2]**2+poles[column-3]**2)

    return k_matrix

def frequency_matrix2(poles,reduced_freq,nmodes,RFA_Method):

    num_reduced_freq = len(reduced_freq)
    npoles = len(poles)
    #k_matrix=np.zeros((num_reduced_freq*2,3+npoles))
    In = np.identity(nmodes)
    ###########################################################################################################################################
    #Roger's Method
    ###########################################################################################################################################
    #pdb.set_trace()
    if RFA_Method == 'E' or RFA_Method == 'e':

        kmatrix = np.hstack([In,1j*reduced_freq[0]*In,-(reduced_freq[0]**2)*In])
        for pi in range(npoles):
            kmatrix = np.hstack([kmatrix,In*(1j*reduced_freq[0]/(poles[pi]+1j*reduced_freq[0]))])
        for ki in range(1,num_reduced_freq):
            kmatrix_ki = np.hstack([In,1j*reduced_freq[ki]*In,-(reduced_freq[ki]**2)*In])
            for pi in range(npoles):
                kmatrix_ki = np.hstack([kmatrix_ki,In/(poles[pi]+1j*reduced_freq[ki])])
            kmatrix = np.vstack([kmatrix,kmatrix_ki])

    ###########################################################################################################################################
    #Eversman's Method
    ###########################################################################################################################################
    if RFA_Method == 'R' or RFA_Method == 'r':


        kmatrix = np.hstack([In,1j*reduced_freq[0]*In,-(reduced_freq[0]**2)*In])
        for pi in range(npoles):
            kmatrix = np.hstack([kmatrix,In*(1j*reduced_freq[0]/(poles[pi]+1j*reduced_freq[0]))])
        for ki in range(1,num_reduced_freq):
            kmatrix_ki = np.hstack([In,1j*reduced_freq[ki]*In,-(reduced_freq[ki]**2)*In])
            for pi in range(npoles):
                kmatrix_ki = np.hstack([kmatrix_ki,In*(1j*reduced_freq[ki]/(poles[pi]+1j*reduced_freq[ki]))])
            kmatrix = np.vstack([kmatrix,kmatrix_ki])
    return kmatrix

#@my_timer
def RFA_matrix(poles,k_matrix,reduced_freq,aero_matrices_real,aero_matrices_imag,RFA_Method,rfa0=0):
    #print rfa0,'rfa_matrix'
    #print rfa0
    if rfa0 and (RFA_Method == 'R' or RFA_Method == 'r'):
        k_matrix = k_matrix[2:,1:]
        reduced_freq = reduced_freq[1:]
        shift = 1
    else:
        shift = 0
    k_matrix_inv=np.linalg.pinv(k_matrix)
    num_reduced_freq = len(reduced_freq)
    npoles = len(poles)
    Qn=np.zeros((num_reduced_freq*2,1))
    #Qn_QS=np.zeros((num_reduced_freq_QS*2,1))

    rows=len(aero_matrices_real[0][:,0])
    columns=len(aero_matrices_real[0][0,:])

    RFA_mat=[]
    for pole in range(npoles+3):
        RFA_mat.append(np.zeros((rows,columns)))

    #Loop over the different GAF terms
    for row in range (rows):
        for column in range (columns):
            for red_freq in range(num_reduced_freq):
                if rfa0 and (RFA_Method == 'R' or RFA_Method == 'r'):
                    A0r = aero_matrices_real[0][row,column]
                    A0i = aero_matrices_imag[0][row,column]
                else:
                    A0r = 0.
                    A0i = 0.

                Qn[red_freq*2,0] = aero_matrices_real[red_freq+shift][row,column] - A0r
                Qn[red_freq*2+1,0] = aero_matrices_imag[red_freq+shift][row,column] - A0i

            ##if RB_Modes_QS == 'y' and (column<6) and (Matrix_type=='qaa'):
            # if RB_Modes_QS == 'y' and (column<6):
            #     for k in range (num_reduced_freq_QS*2):
            #         Qn_QS[k,0]=Qn[k,0]

            #Evaluate the RFA cofficient
            A=np.dot(k_matrix_inv,Qn)
            #A_QS=np.dot(k_matrix_QS_inv,Qn_QS)
            if RFA_Method == 'E' or RFA_Method == 'e':
                RFA_mat[0][row,column] = A[0,0]#aero_matrices_real[0][row,column]
            if RFA_Method == 'R' or RFA_Method == 'r':
                if rfa0:
                    RFA_mat[0][row,column] = aero_matrices_real[0][row,column]
                else:
                    RFA_mat[0][row,column] = A[0,0]#aero_matrices_real[0][row,column]

            RFA_mat[1][row,column] = A[1-shift,0]
            RFA_mat[2][row,column] = A[2-shift,0]
            for pole in range(npoles):
                RFA_mat[3+pole][row,column] = A[3+pole-shift,0]

            ##if RB_Modes_QS == 'y' and (column<6) and (Matrix_type=='qaa'):
            # if RB_Modes_QS == 'y' and (column<6) :
            #     RFA_mat[0][row,column]=aero_matrices_real[0][row,column]
            #     RFA_mat[1][row,column]=A_QS[1,0]
            #     RFA_mat[2][row,column]=A_QS[2,0]
            #     for pole in range(npoles):
            #         RFA_mat[3+pole][row,column]=0.

    return RFA_mat




#@my_timer
def RFA_matrix2(poles,k_matrix,reduced_freq,aero_matrices_real,aero_matrices_imag,RFA_Method,rfa0=0):

    nmodes = np.shape(aero_matrices_real)[1]
    if rfa0 and (RFA_Method == 'R' or RFA_Method == 'r'):
        k_matrix = k_matrix[nmodes:,nmodes:]
        reduced_freq = reduced_freq[1:]
        shift = 1
    else:
        shift=0

    k_matrixr = k_matrix.real
    k_matrixi = k_matrix.imag
    k_matrix_inv=np.linalg.pinv(np.vstack([k_matrixr,k_matrixi]))
    num_reduced_freq = len(reduced_freq)
    npoles = len(poles)

    RFA_mat=[[] for i in range(npoles+3)]

    Aicr = np.vstack(aero_matrices_real[shift:] - shift*aero_matrices_real[0])
    Aici = np.vstack(aero_matrices_imag[shift:] - shift*aero_matrices_imag[0])
    Aics = np.vstack([Aicr,Aici])
    #Evaluate the RFA cofficient
    A=np.dot(k_matrix_inv,Aics)
    ni = np.shape(A)[1]
    #A_QS=np.dot(k_matrix_QS_inv,Qn_QS)
    if RFA_Method == 'E' or RFA_Method == 'e':
            RFA_mat[0] = A[0:ni]
    if RFA_Method == 'R' or RFA_Method == 'r':
        if rfa0:
            RFA_mat[0] = aero_matrices_real[0]
        else:
            RFA_mat[0] = A[0:ni]

    RFA_mat[1]=A[ni-shift*ni:2*ni-shift*ni]
    RFA_mat[2]=A[2*ni-shift*ni:3*ni-shift*ni]
    for pole in range(npoles):
        RFA_mat[3+pole]=A[(3+pole)*ni-shift*ni:(4+pole)*ni-shift*ni]

    return RFA_mat
