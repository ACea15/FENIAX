import numpy as np
import scipy.interpolate as interpolate
import pathlib


def rbf_3Dinterpolators(vertices, displacement,
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

def compute(data_ref, data_def, data_in,
            ids=None, filtering: callable=None, **kwargs):

    data_out_disp = []
    data_out_coord = []
    #X = df_combined.to_numpy()
    X0 = data_in[:, 0]
    X1 = data_in[:, 1]
    X2 = data_in[:, 2]
    #ids = df_combined.index.to_numpy().astype(int)
    #for i in range(num_modes):
    interpolator_rbfX, interpolator_rbfY, interpolator_rbfZ = rbf_3Dinterpolators(data_ref, data_def - data_ref,  **kwargs)
    Ux = interpolator_rbfX(data_in)
    Uy = interpolator_rbfY(data_in)
    Uz = interpolator_rbfZ(data_in)
    if filtering is not None:
        Ux = Ux * filtering(X1)
        Uy = Uy * filtering(X1)
        Uz = Uz * filtering(X1)
    Rx = Ux + X0
    Ry = Uy + X1
    Rz = Uz + X2

    data_out_disp = np.array([Ux, Uy, Uz]).T
    data_out_coord = np.array([Rx, Ry, Rz]).T

    return data_out_disp, data_out_coord
