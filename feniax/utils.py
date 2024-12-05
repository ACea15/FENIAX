from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import pyNastran.op4.op4 as op4
from pyNastran.f06 import parse_flutter as flut
import pathlib
import subprocess
from pyNastran.f06 import parse_flutter as flutter
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
