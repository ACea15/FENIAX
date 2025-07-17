# from pyNastran.bdf.bdf import BDF
# from pyNastran.op2.op2 import OP2
# import pyNastran.op4.op4 as op4
# from pyNastran.f06 import parse_flutter as flut
import pathlib
import subprocess
# from pyNastran.f06 import parse_flutter as flutter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def flatten_list(lis):
    l = list(lis)
    i = 0
    # while type(max(l)) is list:

    while i < len(l):
        if type(l[i]) is tuple:
            l[i] = list(l[i])
        if type(l[i]) is list:
            if len(l[i]) == 0:
                del l[i]
                continue
            for j in range(len(l[i])):
                l.insert(i + j, l[i + j][j])

            del l[i + j + 1]
        else:
            i = i + 1

    return l

def remove_zeros(Mat,retd=None,rtol=1e-05,atol=1e-08):
    M = copy.copy(Mat)
    dofx = np.shape(M)[0]
    dofy = np.shape(M)[1]
    count=0
    dx=[]
    dy=[]
    for i in range(dofx):
        if np.allclose(Mat[i,:],np.zeros(dofy),rtol=rtol,atol=atol):
             M = np.delete(M,i-count,0)
             count+=1
             dx.append(i)
    count=0
    for i in range(dofy):
        #print M[:,i]
        #print np.zeros(dofx)
        if np.allclose(Mat[:,i],np.zeros(dofx),rtol=rtol,atol=atol):
             M =np.delete(M,i-count,1)
             count+=1
             dy.append(i)
    if retd:
        return M,dx,dy
    else:
        return M


def standard_atmosphere(h, k=0.0065, R=287.05, g=9.806, gamma=1.4, T_0=288.16, rho_0=1.225):
    n = 1 / (1 - k * R / g)
    if h < 11000.0:
        T = T_0 - k * h
        rho = rho_0 * (T / T_0) ** (1 / (n - 1))
        P = rho * R * T
        a = np.sqrt(gamma * R * T)
    elif 11000.0 <= h <= 25000.0:
        h_11k = 11000.0
        T_11k = T_0 - k * h_11k
        rho_11k = rho_0 * (T_11k / T_0) ** (1 / (n - 1))
        P_11k = rho_11k * R * T_11k

        psi = np.exp(-(h - h_11k) * g / (R * T_11k))
        T = T_11k
        rho = rho_11k * psi
        P = P_11k * psi
        a = np.sqrt(gamma * R * T_11k)
    return T, rho, P, a

def dict_difference(d1, d2):
    """
    Remove items in d2 from d1 recursively.
    """
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        return d1

    result = {}
    for key in d1:
        if key not in d2:
            result[key] = d1[key]
        else:
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                nested_diff = dict_difference(d1[key], d2[key])
                if nested_diff:  # Only add non-empty dicts
                    result[key] = nested_diff
            elif d1[key] != d2[key]:
                # If they are different, keep the one from d1
                result[key] = d1[key]

    return result

def dict_merge(d1, d2):
    """
    Recursively merge two dictionaries.
    - Values from d2 overwrite or merge into d1.
    """
    result = dict(d1)  # Make a copy to avoid modifying d1
    for key, val in d2.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(val, dict)
        ):
            result[key] = dict_merge(result[key], val)
        else:
            result[key] = val
    return result

def dict_deletebypath(data, path, sep="."):
    """
    Delete a key from a nested dictionary using a dot-separated path.
    
    Args:
        data (dict): The dictionary to delete from.
        path (str): The dot-separated path to the target key.
        sep (str): The separator used for the path (default: '.').
    """
    keys = path.split(sep)
    current = data
    for i, key in enumerate(keys):
        if i == len(keys) - 1:
            if key in current:
                del current[key]
            else:
                raise KeyError(f"Key '{key}' not found at path: {path}")
        else:
            current = current.get(key)
            if not isinstance(current, dict):
                raise KeyError(f"Path invalid at: {sep.join(keys[:i+1])}")
