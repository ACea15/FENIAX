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
