import numpy as np
import scipy.interpolate as interpolate
import pathlib

def read_sharpy(file_name):
                    
    X = np.loadtxt(file_name)
    return X

def read_modes_sharpy(file_modes, num_modes):

    modes = []
    coord = read_sharpy(file_modes + ".vertices")
    for i in range(num_modes):
        modes.append(read_sharpy(file_modes + ".mode%s" %(i+1)))

    return coord, modes

def rbef_3Dinterpolators(vertices, displacement,
                         neighbors=None, smoothing=0.0,
                         kernel='thin_plate_spline',
                         epsilon=None, degree=None,
                         **kwargs):

    interpolator_rbfX = interpolate.RBFInterpolator(vertices, displacement[:, 0],
                                                    neighbors=neighbors,
                                                    smoothing=smoothing,
                                                    kernel=kernel,
                                                    epsilon=epsilon,
                                                    degree=degree,
                                                    **kwargs)
    interpolator_rbfY = interpolate.RBFInterpolator(vertices, displacement[:, 1],
                                                    neighbors=neighbors,
                                                    smoothing=smoothing,
                                                    kernel=kernel,
                                                    epsilon=epsilon,
                                                    degree=degree,
                                                    **kwargs)
    interpolator_rbfZ = interpolate.RBFInterpolator(vertices, displacement[:, 2],
                                                    neighbors=neighbors,
                                                    smoothing=smoothing,
                                                    kernel=kernel,
                                                    epsilon=epsilon,
                                                    degree=degree,
                                                    **kwargs)

    return interpolator_rbfX, interpolator_rbfY, interpolator_rbfZ

def calculate_interpolated_modes(Xv, Xm, X, ids=None, filtering: callable=None, **kwargs):

    num_modes = len(Xm)
    interpolated_modal_displacements = []
    interpolated_modal_coord = []
    #X = df_combined.to_numpy()
    X0 = X[:, 0]
    X1 = X[:, 1]
    X2 = X[:, 2]
    #ids = df_combined.index.to_numpy().astype(int)
    for i in range(num_modes):
        interpolator_rbfX, interpolator_rbfY, interpolator_rbfZ = rbef_3Dinterpolators(Xv, Xm[i] - Xv,  **kwargs)
        Ux = interpolator_rbfX(X)
        Uy = interpolator_rbfY(X)
        Uz = interpolator_rbfZ(X)
        if filtering is not None:
            Ux = Ux * filtering(X1)
            Uy = Uy * filtering(X1)
            Uz = Uz * filtering(X1)
        Rx = Ux + X0
        Ry = Uy + X1
        Rz = Uz + X2

        if ids is not None:
            interpolated_modal_displacements.append(np.array([ids, Ux, Uy, Uz]).T)
            interpolated_modal_coord.append(np.array([ids, Rx, Ry, Rz]).T)
        else:
            interpolated_modal_displacements.append(np.array([Ux, Uy, Uz]).T)
            interpolated_modal_coord.append(np.array([Rx, Ry, Rz]).T)

    return interpolated_modal_displacements, interpolated_modal_coord
