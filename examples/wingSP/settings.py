import pathlib
import jax.numpy as jnp
import pdb
import sys
import datetime
import fem4inas.preprocessor.configuration as configuration  # import Config, dump_to_yaml
from fem4inas.preprocessor.inputs import Inputs
import fem4inas.fem4inas_main

inp = Inputs()
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'c1': None}
inp.fem.grid = "structuralGrid"
inp.fem.folder = pathlib.Path('./FEM/')
inp.fem.num_modes = 50
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
# inp.driver.sol_path= pathlib.Path(
#     f"./results_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
inp.driver.sol_path= pathlib.Path(
    "./results_dynamics_m50")

#inp.driver.sol_path=None
inp.simulation.typeof = "single"
inp.systems.sett.s1.solution = "dynamic"
inp.systems.sett.s1.t1 = 15.
inp.systems.sett.s1.tn = 10001

#inp.systems.sett.s1.solver_library = "diffrax"
inp.systems.sett.s1.solver_library = "runge_kutta"
inp.systems.sett.s1.solver_function = "ode"
#inp.systems.sett.s1.solver_settings = dict(solver_name="Dopri5")
inp.systems.sett.s1.solver_settings = dict(solver_name="rk4")
# inp.systems.sett.s1.solver_library = "scipy"
# inp.systems.sett.s1.solver_function = "root"
# inp.systems.sett.s1.solver_settings = dict(method='hybr',#'krylov',
#                                            tolerance=1e-9)
#inp.systems.sett.s1.label = 'dq_101001'
inp.systems.sett.s1.xloads.follower_forces = True
inp.systems.sett.s1.xloads.follower_points = [[23, 0],
                                              [23, 2]]
inp.systems.sett.s1.xloads.x = [0, 4, 4+1e-6, 20]
inp.systems.sett.s1.xloads.follower_interpolation = [[0.05 * -2e5, 1 * -2e5, 0., 0.],
                                                     [0.05 * 6e5, 1 * 6e5,  0., 0.]
                                                     ]
config =  configuration.Config(inp)
sol = fem4inas.fem4inas_main.main(input_obj=config)

























# import plotly.graph_objects as go
# import numpy as np

# N = 1000
# t = np.linspace(0, 10, 100)
# y = np.sin(t)
# fig = go.Figure(data=go.Scatter(x=t, y=y, mode='markers',
#                                 hovertemplate=('<BR><BR>weights:<BR>' +
#                                                '%{customdata[{0}]}%' +
#                                                '<extra></extra>'),
#                                 #hoverlabel=dict(t=t),
#                                 customdata=[t,y]))

# fig.show()

# import plotly.graph_objs as go


# template=[]
# for i in df.index:
#     dados_sim=[]
#     index = f'<BR><b>Index: </b> {i}'
#     dados_sim.append(index)
#     for j in columns:
#         column_name = j
#         if column_name != 'Sharpe Ratio':
#             linha = f'<BR><b>{j}: </b> {round(df.loc[i, j]*100,2)}%'
#         else:
#             linha = f'<BR><b>{j}: </b> {round(df.loc[i, j],2)}'
#         dados_sim.append(linha)
#     dados_sim = ','.join(dados_sim) + f'<extra></extra>'
#     template.append(dados_sim)


# # Sample data
# x_values = [1, 2, 3, 4, 5]
# y_values = [10, 11, 12, 13, 14]
# extra_info1 = ['A', 'B', 'C', 'D', 'E']  # Sample extra information 1 for each point
# extra_info2 = ['Info1', 'Info2', 'Info3', 'Info4', 'Info5']  # Sample extra information 2 for each point
# extra_info3 = ['DataX', 'DataY', 'DataZ', 'DataW', 'DataV']  # Sample extra information 3 for each point

# template=[]
# for i, xi in enumerate(extra_info1):
#     dados_sim=[]
#     index = f'<BR><b>Index: </b> {xi}'
#     dados_sim.append(index)
#     index2 = f'<BR><b>Index2: </b> {extra_info2[i]}'
#     dados_sim.append(index2)
#     index3 = f'<BR><b>Index3: </b> {extra_info3[i]}'
#     dados_sim.append(index3)
#     dados_sim = ','.join(dados_sim) + '<extra></extra>'
#     template.append(dados_sim)


# # Create a custom hovertemplate
# hover_template = (
#     '<b>X</b>: %{x}<br>'
#     '<b>Y</b>: %{y}<br>'
#     '<b>Extra Info 1</b>: %{text}<br>'
#     '<b>Extra Info 2</b>: %{customdata}<br>'
#     '<b>Extra Info 3</b>: %{meta}<extra></extra>'
# )

# # Create a scatter plot
# fig = go.Figure()

# fig.add_trace(go.Scatter(
#     x=x_values,
#     y=y_values,
#     mode='markers',
#     #text=extra_info1,  # Extra information 1 to include in hover
#     customdata=template,  # Extra information 2 to include in hover
#     #meta=extra_info3,  # Extra information 3 to include in hover
#     hovertemplate=template  # Customize hover information with extra data
# ))

# fig.update_layout(
#     title='Scatter Plot with Three Extra Information Vectors in Hover'
# )

# fig.show()


# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import numpy as np
# np.random.seed(0)
# z1, z2, z3 = np.random.random((3, 7, 7))
# customdata = np.dstack((z2, z3))
# fig = make_subplots(1, 2, subplot_titles=['z1', 'z2'])
# fig.add_trace(go.Heatmap(
#     z=z1,
#     customdata=np.dstack((z2, z3)),
#     hovertemplate='<b>z1:%{z:.3f}</b><br>z2:%{customdata[0]:.3f} <br>z3: %{customdata[1]:.3f} ',
#     coloraxis="coloraxis1", name=''),
#     1, 1)
# fig.add_trace(go.Heatmap(
#     z=z2,
#     customdata=np.dstack((z1, z3)),
#     hovertemplate='z1:%{customdata[0]:.3f} <br><b>z2:%{z:.3f}</b><br>z3: %{customdata[1]:.3f} ',
#     coloraxis="coloraxis1", name=''),
#     1, 2)
# fig.update_layout(title_text='Hover to see the value of z1, z2 and z3 together')
# fig.show()


# # Sample data
# x_values = [1, 2, 3, 4, 5]
# y_values = [10, 11, 12, 13, 14]
# extra_info1 = ['A', 'B', 'C', 'D', 'E']  # Sample extra information 1 for each point
# extra_info2 = ['Info1', 'Info2', 'Info3', 'Info4', 'Info5']  # Sample extra information 2 for each point
# extra_info3 = ['DataX', 'DataY', 'DataZ', 'DataW', 'DataV']  # Sample extra information 3 for each point

# template=[]
# for i, xi in enumerate(extra_info1):
#     dados_sim=[]
#     index = f'<BR><b>Index: </b> {xi}'
#     dados_sim.append(index)
#     index2 = f'<BR><b>Index2: </b> {extra_info2[i]}'
#     dados_sim.append(index2)
#     index3 = f'<BR><b>Index3: </b> {extra_info3[i]}'
#     dados_sim.append(index3)
#     dados_sim = ','.join(dados_sim) + '<extra></extra>'
#     template.append(dados_sim)


# # Create a custom hovertemplate
# hover_template = (
#     '<b>X</b>: %{x}<br>'
#     '<b>Y</b>: %{y}<br>'
#     '<b>Extra Info 1</b>: %{text}<br>'
#     '<b>Extra Info 2</b>: %{customdata}<br>'
#     '<b>Extra Info 3</b>: %{meta}<extra></extra>'
# )

# # Create a scatter plot
# fig = go.Figure()

# fig.add_trace(go.Scatter(
#     x=x_values,
#     y=y_values,
#     mode='markers',
#     #text=extra_info1,  # Extra information 1 to include in hover
#     customdata=np.dstack([extra_info1,extra_info2,extra_info3]),  # Extra information 2 to include in hover
#     #meta=extra_info3,  # Extra information 3 to include in hover
#     hovertemplate='z1:%{customdata[0]:.3f} <br><b>z2:%{customdata[1]:.3f}</b><br>z3: %{customdata[2]:.3f} '  # Customize hover information with extra data
# ))

# fig.update_layout(
#     title='Scatter Plot with Three Extra Information Vectors in Hover'
# )

# fig.show()
