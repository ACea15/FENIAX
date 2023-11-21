import pyvista as pv
from pyvista import examples
import numpy as np
import pathlib
import pdb
import sys
import datetime
import jax.numpy as jnp
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main


def render_wireframe(points, lines, pl: pv.Plotter=None):

    if isinstance(points, jnp.ndarray):
        points = np.array(points)
    if pl is None:
        pl = pv.Plotter()
    mesh = pv.PolyData(points,
                       lines
                       )

    pl.add_mesh(mesh, color='black',line_width=5, style="wireframe",
                render_lines_as_tubes=True)

    pl.add_points(
        points,
        render_points_as_spheres=True,
        #style='points_gaussian',
        #emissive=True,
        #scalars=rgba,
        #rgba=True,
        point_size=7,
        color='red')

    return pl

def render_mesh(points, lines):
    if isinstance(points, jnp.ndarray):
        points = np.array(points)

    mesh = pv.PolyData(points,
                       lines
                       )

    return mesh


class IntrinsicStruct:

    def __init__(self, fem):
        self.fem = fem
        self.nsol = 0
        self.mappoints = dict()
        self.mapmpoints = dict()
        self.lines = None
        self._set_initgeo()
        self._set_linetopology()
        
    def _set_initgeo(self):
        
        self.X = self.fem.X
        self.Xm = self.fem.Xm.T[1:]
        self.npoints = len(self.X)
        
    def _set_linetopology(self):
        
        self.lines = np.vstack([2 * np.ones(self.npoints, dtype=int),
                                self.fem.prevnodes,
                                np.arange(self.npoints, dtype=int)]).T
        
    def _calculate_midpoints(self, ra):
        
        mid_points = jnp.matmul(ra, self.fem.Mavg)
        mid_points = mid_points.at[:,0].set(ra[:, 0])
        return mid_points.T
    
    def add_solution(self, ra: jnp.array, label=None):

        ra_shape = ra.shape
        assert ra_shape[-1] == self.npoints, "ra not the same number of nodes"
        if len(ra_shape) == 3:  # bunch of solutions
            print("loading solutions")
            for i, ra_i in enumerate(ra):
                self.nsol += 1
                if label is None:
                    labeli = self.nsol
                else:
                    labeli = f"{label}{self.nsol}"
                self.mapmpoints[labeli] = self._calculate_midpoints(ra_i)
                self.mappoints[labeli] = ra_i.T
        else:
            self.nsol += 1
            if label is None:
                label = self.nsol
            self.mapmpoints[label] = self._calculate_midpoints(ra_i)
            self.mappoints[label] = ra_i.T
