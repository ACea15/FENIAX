import pyvista as pv
from pyvista import examples
import numpy as np
import pathlib
import pdb
import sys
import datetime
import jax.numpy as jnp
import feniax.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main


def render_wireframe(points, lines, pl: pv.Plotter = None):
    if isinstance(points, jnp.ndarray):
        points = np.array(points)
    if pl is None:
        pl = pv.Plotter()
    mesh = pv.PolyData(points, lines)

    pl.add_mesh(mesh, color="black", line_width=5, style="wireframe", render_lines_as_tubes=True)

    pl.add_points(
        points,
        render_points_as_spheres=True,
        # style='points_gaussian',
        # emissive=True,
        # scalars=rgba,
        # rgba=True,
        point_size=7,
        color="red",
    )

    return pl


def render_mesh(points, lines):
    if isinstance(points, jnp.ndarray):
        points = np.array(points)

    mesh = pv.PolyData(points, lines)

    return mesh

import pyvista as pv

points = np.random.rand(100, 3)
mesh = pv.PolyData(points)
mesh.plot(point_size=10, style='points', color='tan')

polydata = pv.PolyData(points)

# Add the vectors as point data
polydata["vectors"] = vectors

# Create glyphs to represent vectors
glyphs = polydata.glyph(orient="vectors", scale=False, factor=0.3)

# Plot the vector field
plotter = pv.Plotter()
plotter.add_mesh(glyphs, color='red')
plotter.add_mesh(polydata, color='blue', point_size=5, render_points_as_spheres=True)
plotter.show()

mesh = pyvista.PolyData(v, cells)
mesh.save(folder_path / f"collocation_{k}.ply", binary=False)
    

X=config.fem.X,
time=range(len(inp.system.t)),
ra=sol.staticsystem_sys1.ra[i],
Rab=sol.staticsystem_sys1.Cab[i],
