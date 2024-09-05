from enum import Enum
import jax.numpy as jnp
import numpy as np


class EnumAxis(Enum):
    TIME = 0
    DIM = 1
    NODE = 2


def build_axisindex(shape: tuple, fixaxis: dict):
    axis = [None for i in shape]
    if fixaxis is not None:
        for k, v in fixaxis.items():
            if isinstance(k, str):
                ki = EnumAxis[k.upper()].value
            elif isinstance(k, int):
                ki = k
            else:
                raise ValueError(f"wrong axis input {k}")
            axis[ki] = v
    # axis = [(ind if ind is not None else list(range(shape[i])))  for i, ind in enumerate(axis)]
    for i, ind in enumerate(axis):
        if ind is not None:
            axis[i] = ind
        elif i == len(axis) - 1:
            return axis[:-1]
        else:
            axis[i] = list(range(shape[i]))

    return axis


def pickIntrinsic2D(data1, data2, fixaxis1=None, fixaxis2=None):
    shape1 = data1.shape
    shape2 = data2.shape
    axis1 = build_axisindex(shape1, fixaxis1)
    axis2 = build_axisindex(shape2, fixaxis2)
    x = data1[tuple(axis1)]
    y = data2[tuple(axis2)]
    return x, y


def pickIntrinsic3D(data1, data2, data3, fixaxis1=None, fixaxis2=None, fixaxis3=None):
    shape1 = data1.shape
    shape2 = data2.shape
    shape3 = data3.shape
    axis1 = build_axisindex(axis1, fixaxis1)
    axis2 = build_axisindex(axis2, fixaxis2)
    axis3 = build_axisindex(axis2, fixaxis2)
    x = data1[tuple(axis1)]
    y = data2[tuple(axis2)]
    z = y = data3[tuple(axis3)]
    return x, y, z


class IntrinsicStruct:
    def __init__(self, fem):
        self.fem = fem
        self.nsol = 0
        self.map_ra = dict()
        self.map_mra = dict()
        self.labels_new = None
        self.lines = None
        self._set_initgeo()
        self._set_linetopology()

    def _set_initgeo(self):
        self.X = self.fem.X
        self.Xm = self.fem.Xm.T[1:]
        self.npoints = len(self.X)

    def _set_linetopology(self):
        self.lines = np.vstack(
            [
                2 * np.ones(self.npoints, dtype=int),
                self.fem.prevnodes,
                np.arange(self.npoints, dtype=int),
            ]
        ).T

    def _calculate_midpoints(self, ra):
        mid_points = jnp.matmul(ra, self.fem.Mavg)
        mid_points = mid_points.at[:, 0].set(ra[:, 0])
        return mid_points.T

    def add_solution(self, ra: jnp.array, label=None, label_final=None):
        ra_shape = ra.shape
        self.labels_new = list()
        assert ra_shape[-1] == self.npoints, "ra not the same number of nodes"
        if label_final is not None:
            self.nsol += 1
            self.map_mra[label_final] = self._calculate_midpoints(ra)
            self.map_ra[label_final] = ra.T
            self.labels_new.append(label_final)
        else:
            if len(ra_shape) == 3:  # bunch of solutions
                print("loading solutions")
                for i, ra_i in enumerate(ra):
                    self.nsol += 1
                    if label is None:
                        labeli = self.nsol
                    else:
                        labeli = f"{label}{self.nsol}"
                    self.map_mra[labeli] = self._calculate_midpoints(ra_i)
                    self.map_ra[labeli] = ra_i.T
                    self.labels_new.append(labeli)
            else:
                self.nsol += 1
                if label is None:
                    labeli = self.nsol
                else:
                    labeli = f"{label}{self.nsol}"
                # breakpoint()
                self.map_mra[labeli] = self._calculate_midpoints(ra)
                self.map_ra[labeli] = ra.T
                self.labels_new.append(labeli)


class IntrinsicStructComponent(IntrinsicStruct):
    def __init__(self, fem):
        super().__init__(fem)
        self.map_components = dict()
        self.add_solution(self.X.T, label="ref")

    def _set_linetopology(self):
        self.lines = dict()
        for i, ci in enumerate(self.fem.component_names):
            if i > 0:
                ci_father = self.fem.component_father[ci]
                if ci_father is None:
                    ci_father_node = 0
                else:
                    ci_father_node = self.fem.component_nodes[ci_father][-1]
                self.lines[ci] = [ci_father_node] + self.fem.component_nodes[ci]
            else:
                self.lines[ci] = self.fem.component_nodes[ci]

    def _add_solcomponent(self):
        for labeli in self.labels_new:
            self.map_components[labeli] = list()
            for k, v in self.lines.items():
                self.map_components[labeli].append(self.map_ra[labeli][jnp.array(v)])
            # self.map_components[labeli] = np.array(self.map_components[labeli])

    def add_solution(self, ra: jnp.array, label=None, label_final=None):
        super().add_solution(ra, label, label_final)
        self._add_solcomponent()
