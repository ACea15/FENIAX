
import matplotlib.pyplot as plt

import feniax.plotools.utils as putils
import feniax.preprocessor.solution as solution  
import feniax.preprocessor.configuration as configuration  
from feniax.preprocessor.inputs import Inputs
import feniax.feniax_main
import jax.numpy as jnp
import math
import pathlib


# Define rigid-body initial conditions
omega_0 = math.sqrt(12)
T=(2*math.pi)/(math.sqrt(2)*omega_0)
v_x = 0.
v_y = 0.
v_z = 0.
omega_x = 0.
omega_y = omega_0/4.
omega_z = 0.

# Options
gravity_forces = False
gravity_label = "g" if gravity_forces else ""
label = 'm8'
num_modes = 13 # [8, 10]

# 8 modes simulation: 6RB modes  + 2 bending modes (symmetric and antysymmetric)
# 10 modes simulation: 6RB modes + 2 bending modes + 2 roational modes in y (also symmetric and antysymmetric)

ic = "spin" # "bending", "omega"

# Inputs to FENIAX
inp = Inputs()
inp.log.level="error"
inp.engine = "intrinsicmodal"
inp.fem.connectivity = {'rbeam': None, 'lbeam': None}
inp.fem.Ka_name = f"./FEM/Ka_{label}.npy"
inp.fem.Ma_name = f"./FEM/Ma_{label}.npy"
inp.fem.eig_names = [f"./FEM/eigenvals_{label}.npy",
                     f"./FEM/eigenvecs_{label}.npy"]
inp.fem.grid = f"./FEM/structuralGrid_{label}"
inp.fem.eig_type = "inputs"
inp.driver.typeof = "intrinsic"
inp.simulation.typeof = "single"
inp.system.name = "s1"
inp.system.solution = "dynamic"
inp.system.bc1 = 'free'
inp.system.xloads.gravity_forces = gravity_forces
inp.system.t1 = 2*T 
inp.system.tn = 50001 #20000 * 20 + 1
inp.system.solver_library = "runge_kutta" #"diffrax" #
inp.system.solver_function = "ode"
inp.system.solver_settings = dict(solver_name="rk4")

#for ni in enumerate(num_modes):
ni=num_modes
v0=omega_0/4  # max value of vz
if ni==num_modes:
    label_name = label + f"N{ni}" + gravity_label
    inp.fem.num_modes = ni  
    vz = [0, 0.2*v0, 0.4*v0, 0.6*v0, 0.8*v0, v0] # [0., 0.2, 0.3, 0.4, 0.5, 0.6]
    for i, vzi in enumerate(vz):
        label_i = label_name + f"vz{i}"
        inp.driver.sol_path= pathlib.Path(
            f"./results_ant{label_i}{ic}")
        if ic == "bending":
            inp.system.init_states = dict(q1=["nodal_prescribed",
                                              ([[v_x, v_y, v_z, omega_x, (omega_y+vzi), omega_z],
                                                [v_x, v_y, v_z - (omega_y) * 1 - vzi, omega_x, (omega_y-2*vzi), omega_z],
                                                [v_x, v_y, v_z + (omega_y) * 1 + vzi, omega_x, (omega_y-2*vzi), omega_z]]
                                               ,)
                                              ]
                                          )
        elif ic == "omega":
          inp.system.init_states = dict(q1=["nodal_prescribed",
                                            ([[v_x, v_y, v_z, omega_x, omega_y, omega_z],
                                              [v_x, v_y, v_z - omega_y * 1 - vzi, omega_x, omega_y, omega_z],
                                              [v_x, v_y, v_z + omega_y * 1 + vzi, omega_x, omega_y, omega_z]]
                                             ,)
                                            ]
                                        )
        elif ic == "spin":
          inp.system.init_states = dict(q1=["nodal_prescribed",
                                            ([[v_x, v_y, v_z, omega_x, omega_y, omega_z],
                                              [v_x, v_y, v_z - omega_y * 1, omega_x, omega_y, omega_z],
                                              [v_x, v_y, v_z + omega_y * 1, omega_x, omega_y, omega_z]]
                                             ,)
                                            ]
                                        )
            
        config =  configuration.Config(inp)
        sol = feniax.feniax_main.main(input_obj=config) 

############################################################################################
# Postprocessing

# Define plotting function
def plot_multiple_2d(
    x_list,
    y_list,
    labels=None,
    colors=None,
    line_styles=None,
    markers=None,
    title="2D Plot",
    x_label="t/T",
    y_label="Y-Axis",
    xlim=None,
    ylim=None,
    grid=True,
    figsize=(10, 6),
    legend=True,
    filename="myplot"):
    """
    Plot multiple 2D datasets on the same plot with customization.

    Parameters:
        x_list (list of lists/arrays): List of x-data arrays.
        y_list (list of lists/arrays): List of y-data arrays.
        labels (list of str): Labels for each line (for legend).
        colors (list of str): Colors for each line.
        line_styles (list of str): Line styles for each line.
        markers (list of str): Marker styles for each line.
        title (str): Title of the plot.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        grid (bool): Whether to show grid.
        figsize (tuple): Size of the figure.
        legend (bool): Whether to show legend.
        filename (str): If specified, filename to save the plot image.
        show (bool): Whether to display the plot.
    """

    plt.figure(figsize=figsize)

    num_lines = len(x_list)

    for i in range(num_lines):
        x = x_list[i] 
        y = y_list[i]
        label = labels[i] if labels and i < len(labels) else None
        color = colors[i] if colors and i < len(colors) else None
        linestyle = line_styles[i] if line_styles and i < len(line_styles) else '-'
        marker = markers[i] if markers and i < len(markers) else ''

        plt.plot(x, y, label=label, color=color, linestyle=linestyle, marker=marker)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if ylim is not None:
        plt.ylim(*ylim)   
    if grid:
        plt.grid(True)

    if legend and labels:
        plt.legend()

    if filename:
        plt.savefig(filename, bbox_inches='tight')

    plt.close()

def read_result_folders(base_path='.'):
    base = pathlib.Path(base_path)
    return [f.name for f in base.iterdir() if f.is_dir() and f.name.startswith('results_')]

def read_results():
    folders = read_result_folders()
    results = dict()
    for fi in folders:
        name = fi.split("results_")[1]
        print(f"Reading {name} ...")
        results[name] = solution.IntrinsicReader(fi)

    return results

####################################################################
# Load results
results = read_results()


# Plot angular velocity in center node, 8 modes don't capture  antisymmetric oscillations 
x1, y1 = putils.pickIntrinsic2D(results[f"ant{label}N{num_modes}vz1{ic}"].data.dynamicsystem_s1.t,
                                results[f"ant{label}N{num_modes}vz1{ic}"].data.dynamicsystem_s1.X1,
                                fixaxis2=dict(node=0, dim=4))
x2, y2 = putils.pickIntrinsic2D(results[f"ant{label}N{num_modes}vz2{ic}"].data.dynamicsystem_s1.t,
                                results[f"ant{label}N{num_modes}vz2{ic}"].data.dynamicsystem_s1.X1,
                                fixaxis2=dict(node=0, dim=4))
x3, y3 = putils.pickIntrinsic2D(results[f"ant{label}N{num_modes}vz3{ic}"].data.dynamicsystem_s1.t,
                                results[f"ant{label}N{num_modes}vz3{ic}"].data.dynamicsystem_s1.X1,
                                fixaxis2=dict(node=0, dim=4)) 
x4, y4 = putils.pickIntrinsic2D(results[f"ant{label}N{num_modes}vz4{ic}"].data.dynamicsystem_s1.t,
                                results[f"ant{label}N{num_modes}vz4{ic}"].data.dynamicsystem_s1.X1,
                                fixaxis2=dict(node=0, dim=4)) 
x5, y5 = putils.pickIntrinsic2D(results[f"ant{label}N{num_modes}vz5{ic}"].data.dynamicsystem_s1.t,
                                results[f"ant{label}N{num_modes}vz5{ic}"].data.dynamicsystem_s1.X1,
                                fixaxis2=dict(node=0, dim=4)) 

fig = plot_multiple_2d([x1/T,x2/T,x3/T,x4/T,x5/T], 
                       [y1/(omega_0/4), y2/(omega_0/4),y3/(omega_0/4),y4/(omega_0/4),y5/(omega_0/4)], 
                        #ylim=[0.9,2],
                        line_styles=['-','--','--','-'],
                        filename='img/rotvel.png')



# Plot inertial velocity of node 1 in material frame.
x1, y1 = putils.pickIntrinsic2D(results[f"ant{label}N{num_modes}vz1{ic}"].data.dynamicsystem_s1.t,
                                results[f"ant{label}N{num_modes}vz1{ic}"].data.dynamicsystem_s1.X1,
                                fixaxis2=dict(node=1, dim=2)) 
x2, y2 = putils.pickIntrinsic2D(results[f"ant{label}N{num_modes}vz2{ic}"].data.dynamicsystem_s1.t,
                                results[f"ant{label}N{num_modes}vz2{ic}"].data.dynamicsystem_s1.X1,
                                fixaxis2=dict(node=1, dim=2)) 
x3, y3 = putils.pickIntrinsic2D(results[f"ant{label}N{num_modes}vz3{ic}"].data.dynamicsystem_s1.t,
                                results[f"ant{label}N{num_modes}vz3{ic}"].data.dynamicsystem_s1.X1,
                                fixaxis2=dict(node=1, dim=2)) 
x4, y4 = putils.pickIntrinsic2D(results[f"ant{label}N{num_modes}vz4{ic}"].data.dynamicsystem_s1.t,
                                results[f"ant{label}N{num_modes}vz4{ic}"].data.dynamicsystem_s1.X1,
                                fixaxis2=dict(node=1, dim=2))
x5, y5 = putils.pickIntrinsic2D(results[f"ant{label}N{num_modes}vz5{ic}"].data.dynamicsystem_s1.t,
                                results[f"ant{label}N{num_modes}vz5{ic}"].data.dynamicsystem_s1.X1,
                                fixaxis2=dict(node=1, dim=2))
fig = plot_multiple_2d([x1/T,x2/T,x3/T,x4/T,x5/T], 
                       [y1,y2,y3,y4,y5], 
                        ylim=[-2., 2.],
                        line_styles=['-','--','--','-'],
                        filename='img/tipvel.png')
