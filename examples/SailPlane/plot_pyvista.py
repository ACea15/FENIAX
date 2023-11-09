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

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.eig_type = "inputs"
inp.fem.connectivity = dict(FuselageFront=['RWingInner',
                                           'LWingInner'],
                            FuselageBack=['BottomTail',
                                          'Fin'],
                            RWingInner=['RWingOuter'],
                            RWingOuter=None,
                            LWingInner=['LWingOuter'],
                            LWingOuter=None,
                            BottomTail=['LHorizontalStabilizer',
                                        'RHorizontalStabilizer'],
                            RHorizontalStabilizer=None,
                            LHorizontalStabilizer=None,
                            Fin=None
                            )

inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 50
config = configuration.Config(inp)
points = np.array(config.fem.X)
num_points = len(points)
lines = np.vstack([2*np.ones(num_points, dtype=int),
                   config.fem.prevnodes,
                   np.arange(num_points, dtype=int)]).T
lines_stack = np.hstack(lines[1:])

pl = pv.Plotter()
# mesh = examples.download_dragon()
# mesh['scalars'] = mesh.points[:, 1]
# pl.add_mesh(mesh)
# pl.show()
# mesh.plot(cpos='xy', cmap='plasma', pbr=True, metallic=1.0, roughness=0.6,zoom=1.7)
mesh = pv.PolyData(points,
                   lines
                   #lines=lines_stack
                   )
# mesh.plot(render_lines_as_tubes=True,
#     style='wireframe',
#     line_width=10,
#     cmap='jet',
#     show_scalar_bar=False,
#     background='w')
pl.add_points(
    points,
    render_points_as_spheres=True,
    #style='points_gaussian',
    #emissive=True,
    #scalars=rgba,
    #rgba=True,
    point_size=7,
color='red')
pl.add_mesh(mesh, color='black',line_width=5, style="wireframe",
            render_lines_as_tubes=True)
pl.show()

# mesh = examples.load_airplane()
# mesh.plot()

nodes = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [4.0, 3.0, 0.0],
    [4.0, 0.0, 0.0],
    [0.0, 1.0, 2.0],
    [4.0, 1.0, 2.0],
    [4.0, 3.0, 2.0],
]


edges = np.array(
    [
        [0, 4],
        [1, 4],
        [3, 4],
        [5, 4],
        [6, 4],
        [3, 5],
        [2, 5],
        [5, 6],
        [2, 6],
    ]
)

# We must "pad" the edges to indicate to vtk how many points per edge
padding = np.empty(edges.shape[0], int) * 2
padding[:] = 2
edges_w_padding = np.vstack((padding, edges.T)).T

mesh = pv.PolyData(nodes, edges_w_padding)

colors = range(edges.shape[0])
mesh.plot(
    #scalars=colors,
    render_lines_as_tubes=True,
    style='wireframe',
    line_width=10,
    cmap='jet',
    show_scalar_bar=False,
    background='w',
)


# cube = pv.Cube()
# cube.cell_data['myscalars'] = range(6)

# other_cube = cube.copy()
# other_cube.point_data['myscalars'] = range(8)


# pl = pv.Plotter(shape=(1, 2), border_width=1)
# pl.add_mesh(cube, cmap='coolwarm')
# pl.subplot(0, 1)
# pl.add_mesh(other_cube, cmap='coolwarm')
# pl.show()
# ##############
# pl2 = pv.Plotter()
# mesh2 = pv.PolyData(np.array([[0,0,0],
#                      [1,0,0]]),
#                     lines=[2, 0, 1])

# pl2.add_mesh(mesh2)
# pl2.show()


# pl3 = pv.Plotter()
# points = [[0, 0, 0],
#           [1, 0, 0],
#           [0.5, 0.667, 0],
#           [0.5,-0.6,0]]
# cells = np.hstack([[3, 0, 1, 2], [3, 0, 1, 3]])
# mesh3 = pv.PolyData(points, cells)
# pl3.add_mesh(mesh3)
# pl3.show()

# import numpy as np
# pl3 = pv.Plotter()
# vertices = np.array([[0, 0, 0], [1, 0, 0]])
# lines = np.hstack([[2, 0, 1]])
# mesh = pv.PolyData(vertices, lines=lines)
# pl3.add_mesh(mesh)
# pl3.show()

