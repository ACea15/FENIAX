# Imports

# [[file:main.org::*Imports][Imports:1]]
import pdb
import datetime
import os
import shutil
REMOVE_RESULTS = False
#   for root, dirs, files in os.walk('/path/to/folder'):
#       for f in files:
#           os.unlink(os.path.join(root, f))
#       for d in dirs:
#           shutil.rmtree(os.path.join(root, d))
# 
if os.getcwd().split('/')[-1] != 'results':
    if not os.path.isdir("./figs"):
        os.mkdir("./figs")
    if REMOVE_RESULTS:
        if os.path.isdir("./results"):
            shutil.rmtree("./results")
    if not os.path.isdir("./results"):
        print("***** creating results folder ******")
        os.mkdir("./results")
    os.chdir("./results")
# Imports:1 ends here



# #+NAME: PYTHONMODULES

# [[file:main.org::PYTHONMODULES][PYTHONMODULES]]
import pathlib
import plotly.express as px
import plotly.graph_objects as go
import pickle
import jax.numpy as jnp
import pandas as pd
import numpy as np
import feniax.plotools.uplotly as uplotly
import feniax.preprocessor.solution as solution
import feniax.preprocessor.configuration as configuration
from tabulate import tabulate
# PYTHONMODULES ends here

# Common functions

# [[file:main.org::*Common functions][Common functions:1]]
name=[]
figfmt="png"

scale_quality = 6
print(f"Format for figures: {figfmt}")
print(f"Image quality: {scale_quality}")  
def fig_out(name, figformat=figfmt, update_layout=None):
    def inner_decorator(func):
        def inner(*args, **kwargs):
            fig = func(*args, **kwargs)
            if update_layout is not None:
                fig.update_layout(**update_layout)
            fig.show()
            figname = f"figs/{name}.{figformat}"
            fig.write_image(f"../{figname}", scale=scale_quality)
            return fig, figname
        return inner
    return inner_decorator

def fig_background(func):

    def inner(*args, **kwargs):
        fig = func(*args, **kwargs)
        # if fig.data[0].showlegend is None:
        #     showlegend = True
        # else:
        #     showlegend = fig.data[0].showlegend

        fig.update_xaxes(
                       #titlefont=dict(size=20),
                       tickfont = dict(size=20),
                       mirror=True,
                       ticks='outside',
                       showline=True,
                       linecolor='black',
            #zeroline=True,
        #zerolinewidth=2,
            #zerolinecolor='LightPink',
                       gridcolor='lightgrey')
        fig.update_yaxes(tickfont = dict(size=20),
                       #titlefont=dict(size=20),
                       zeroline=True,
                       mirror=True,
                       ticks='outside',
                       showline=True,
                       linecolor='black',
                       gridcolor='lightgrey')
        fig.update_layout(plot_bgcolor='white',
                          yaxis=dict(zerolinecolor='lightgrey'),
                          font=dict(
                              family="Arial",
                              size=18,
                              color="black"
                          ),
                          #showlegend=True, #showlegend,
                          margin=dict(
                              autoexpand=True,
                              l=0,
                              r=0,
                              t=2,
                              b=0
                          ))
        return fig
    return inner

# fig.update_layout(
#     xaxis=dict(
#         title='X AxisTitle',
#         title_font=dict(family='Arial Black', size=22, color='black'),
#         tickfont=dict(family='Arial', size=18, color='black')
#     ),
#     yaxis=dict(
#         title='Y Axis Title',
#         title_font=dict(family='Arial Black', size=22, color='black'),
#         tickfont=dict(family='Arial', size=18, color='black')
#     ),
#     font=dict(
#         family="Arial",
#         size=18,
#         color="black"
#     ),
#     legend=dict(
#         font=dict(size=16),
#         x=0.02,
#         y=0.98,
#         bgcolor='rgba(255,255,255,0)',  # transparent background
#         bordercolor='black',
#         borderwidth=1
#     ),
#     margin=dict(l=80, r=40, t=40, b=80),
#     width=700,
#     height=500
# )
# Common functions:1 ends here

# Plot functions

# [[file:main.org::*Plot functions][Plot functions:1]]
name=[]
figfmt="png"

@fig_background
def plot_jacpdiff(x, yobj, yjac):

    fig = None
    fig = uplotly.lines2d(x, yobj, fig,
                          dict(name="Objective",
                               line=dict(color="black"),
                               marker=dict(symbol="circle")
                               ),
                          dict())
    fig = uplotly.lines2d(x, yjac, fig,
                          dict(name="Jacobian",
                               line=dict(color="blue"),
                               marker=dict(symbol="square")
                               ),
                          dict())

    fig.update_xaxes(type="log",
                     #tickformat= '.0e'
                     exponentformat = 'power'
                     )
    fig.update_yaxes(type="log",
                     #tickformat= '.0e'
                     exponentformat = 'power'
                     )
    fig.update_layout(xaxis_title="Number of paths",
                      yaxis_title=r'$\Large \epsilon$',
                      showlegend=True)

    return fig

@fig_background
def plot_jacediff(x, yjac):

    fig = None
    fig = uplotly.lines2d(x, yjac, fig,
                          dict(#name="Jacobian",
                               line=dict(color="blue"),
                               marker=dict(symbol="square")
                               ),
                          dict())

    fig.update_xaxes(type="log",
                     #tickformat= '.0e'
                     exponentformat = 'power'
                     )
    fig.update_yaxes(type="log",
                     #tickformat= '.0e'
                     exponentformat = 'power'
                     )
    fig.update_layout(xaxis_title=r'$\Large \epsilon$ ',
                      yaxis_title=r'$\Large \epsilon$')
    #fig.update_layout(xaxis_type="log", yaxis_type="log")
    return fig

@fig_background
def plot_jacfem(jac, xlabel="", ylabel=""):

    fig = go.Figure(data=go.Heatmap(
        z=jac, colorscale = 'hot'))

    # fig = px.imshow(jac)
    # fig.update_xaxes(type="log",
    #                  #tickformat= '.0e'
    #                  exponentformat = 'power'
    #                  )
    # fig.update_yaxes(type="log",
    #                  #tickformat= '.0e'
    #                  exponentformat = 'power'
    #                  )
    fig.update_layout(xaxis_title=xlabel,
                      yaxis_title=ylabel)
    #fig.update_layout(xaxis_type="log", yaxis_type="log")
    return fig

@fig_background
def plot_manoeuvretip(aoa, ua, ua_lin):
    fig=None
    colors = ["steelblue", "black"]
    dashes = ["solid", "dash"]
    fig = uplotly.lines2d(aoa, ua, fig,
    dict(name=f"Nonlinear",
    line=dict(color=colors[0],
    dash=dashes[0])
    ))
    fig = uplotly.lines2d(aoa, ua_lin, fig,
    dict(name=f"Linear",
    line=dict(color=colors[1],
    dash=dashes[1])
    ))

    fig.update_yaxes(title=r'$\large \hat{u}_z [\%]$')
    fig.update_xaxes(#range=aoa,
    title=r'$AoA [^o]$')
    return fig

@fig_background
def plot_gustshard(x, y, z, component):

    fig = go.Figure(data =
                    go.Contour(
                        z= z[:,:, component],
                        x=x, # horizontal axis
                        y=y, # vertical axis
                        colorscale='Blues',
                        colorbar=dict(
                            tickfont=dict(size=20)
                        )  # Set tick font size
                    )
                    )
    fig.update_yaxes(title="Gust length [m]")
    fig.update_xaxes(title="Gust intensity [m/s]")
    return fig
# Plot functions:1 ends here

# Computing derivatives of expectations

# [[file:main.org::*Computing derivatives of expectations][Computing derivatives of expectations:1]]
sol_admc1_t = solution.IntrinsicReader("./ADDiscreteMC1_t")
sol_admc1_fem = solution.IntrinsicReader("./ADDiscreteMC1_fem")
jac_t = sol_admc1_t.data.staticsystem_s1.jac['t']
obj_t = sol_admc1_t.data.staticsystem_s1.objective
jac_fem = sol_admc1_fem.data.staticsystem_s1.jac

mc1_jacpaths = [8, 80, 4e2, 8e2] #, 4e3] #[8, 80, 4e2, 8e2, 4e3, 8e3, 2e4]
mc1_eps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
sol_admc1_e = dict()
sol_admc1_j = dict()
mc1_jac = list()
mc1_jobj = list()
mc1_eobj = list()  
mc1_ejac = list()
jac_pdiff = list()
obj_pdiff = list()  
jac_ediff = list() 
for i, _ in enumerate(mc1_jacpaths):
    sol_admc1_j[i] = solution.IntrinsicReader(f"./ADDiscreteMC1_tjac{i}")
    mc1_jobj.append(sol_admc1_j[i].data.staticsystem_s1.objective)
    mc1_jac.append(sol_admc1_j[i].data.staticsystem_s1.jac['t'])
for i, _ in enumerate(mc1_jacpaths): # needing to read all to take last one
    obj_pdiff.append(jnp.linalg.norm(mc1_jobj[i] - mc1_jobj[-1]) /
                     jnp.linalg.norm(mc1_jobj[-1]))
    jac_pdiff.append(jnp.linalg.norm(mc1_jac[i]-mc1_jac[-1]) /
                     jnp.linalg.norm(mc1_jac[-1]))

for i, ei in enumerate(mc1_eps):
    sol_admc1_e[i] = solution.IntrinsicReader(f"./ADDiscreteMC1_te{i}")
    mc1_eobj.append(sol_admc1_e[i].data.staticsystem_s1.objective)
    mc1_ejac.append((mc1_eobj[i] - obj_t) / ei)
    jac_ediff.append(jnp.linalg.norm(mc1_ejac[i]-jac_t) / jnp.linalg.norm(jac_t))
# Computing derivatives of expectations:1 ends here



# Now we set out to calculate the derivatives of the expectations previously computed concurrently via Montecarlo simulations in Sec. [[Uncertainty quantification of nonlinear response]]. While the Montecarlo paths are independent of each other and could therefore be run on different machines, having to do AD on the output statistics gathered via collective operations, forces the entire chain of operations to be within a single program. This makes for an interesting and challenging problem to propagate gradients through concurrent operations. A linear parameter \(\alpha\) is introduced such that the follower forces and torsional moments in Eq. \eqref{eq:normal_loading} are such \(\mu = 10^4 (\frac{\alpha - 1.5}{4-1.5} + 1.5\times\frac{\alpha - 1}{5-1}) \). The selected output is the expectation of a 3-component vector, \(\bm{r}(\alpha)\) of the wing-tip positions at \(\alpha = 4.5\). Fig. [[fig:jac_ediff]] shows a comparison between the derivative \(\partial_{\mathbb{E}_\alpha} \bm{r} = \partial \mathbb{E}[\bm{r}] / \partial \alpha \) using AD and finite differences \( \partial_{\alpha} \bm{r}^{fd}= (\bm{r}(\alpha + \epsilon) - \bm{r}(\alpha))/\epsilon  \). The relative error is calculated as  \(||\partial_{\alpha} \bm{r}^{ad} - \partial_{\alpha} \bm{r}^{fd} || / ||\partial_{\alpha} \bm{r}^{ad} ||  \)

# #+NAME: jac_ediff

# [[file:main.org::jac_ediff][jac_ediff]]
name="jac_ediff"
fig, figname = fig_out(name)(plot_jacediff)(mc1_eps, jac_ediff)
figname
# jac_ediff ends here



# #+NAME: fig:jac_ediff
# #+CAPTION: Verification against finite-differences of 
# #+ATTR_LATEX: :width 0.6\textwidth 
# #+RESULTS: jac_ediff
# [[file:figs/jac_ediff.png]]

# #+NAME: jac_pdiff

# [[file:main.org::jac_pdiff][jac_pdiff]]
name="jac_pdiff"
fig, figname = fig_out(name)(plot_jacpdiff)(mc1_jacpaths, obj_pdiff[:-1], jac_pdiff[:-1])
figname
# jac_pdiff ends here



# #+NAME: fig:jac_pdiff
# #+CAPTION: 
# #+ATTR_LATEX: :width 0.8\textwidth 
# #+RESULTS: jac_pdiff
# [[file:figs/jac_pdiff.png]]


# #+NAME: jac_eigenvecs

# [[file:main.org::jac_eigenvecs][jac_eigenvecs]]
name="jac_eigenvecs"
fig, figname = fig_out(name)(plot_jacfem)(jac_fem['eigenvecs'][0,1,0, 65+50:218, :],
                                          xlabel="Right-Wing Eigenvector components",
                                          ylabel="Mode"
                                          )
figname
# jac_eigenvecs ends here



# #+NAME: fig:jac_eigenvecs
# #+CAPTION: Jacobian of wing tip position with respect to Mass matrix right-wing components
# #+ATTR_LATEX: :width 0.8\textwidth 
# #+RESULTS: jac_eigenvecs
# [[file:figs/jac_eigenvecs.png]]

# #+NAME: jac_Ma

# [[file:main.org::jac_Ma][jac_Ma]]
name="jac_Ma"
fig, figname = fig_out(name)(plot_jacfem)(jac_fem['Ma'][0,1,0, 65+50:218, 65+50:218],
                                          #xlabel="Wing Eigenvectors",
                                          #ylabel="Mode"
                                          )
figname
# jac_Ma ends here

# Steady manoeuvre loads

# [[file:main.org::*Steady manoeuvre loads][Steady manoeuvre loads:1]]
sol_manoeuvre = solution.IntrinsicReader("./manoeuvre1Shard")
config_manoeuvre = configuration.Config.from_file("./manoeuvre1Shard/config.yaml")
t = [1/6*1e-2, 1/6, 1/3, 1/2, 2/3, 5/6, 1]
aoa = [6*ti for ti in t]
ra = sol_manoeuvre.data.staticsystem_s1.ra[-1]
component = 2
node = 35
ra_tip0 = config_manoeuvre.fem.X[node]
ra_tip = ra[:, :, node]
ua = ra_tip - ra_tip0
semispan = ra_tip0[1] 
uatip = ua[:, component] / semispan * 100
uatip_lin = [uatip[0]/t[0]*ti for ti in t]
# Steady manoeuvre loads:1 ends here



# We extend the previous analysis to a static aeroelastic case for varying angles of attack that represent a manoeuvre scenario.  We test the parallelisation by varying the flow density ($\pm 20 \%$ of the reference density 0.41 Kg/ m$^3$) as well and the flow velocity ($\pm 20 \%$ of the reference velocity 209.6 m/s). 16 different points for both density and velocity make a total number of 256 simulations. The Mach number is fixed at 0.7 corresponding to the reference flow condition values.
# Fig. [[fig:BUG_manoeuvre3D]] illustrates the 3D equilibrium of the airframe at the reference flight conditions. 

# #+NAME: fig:BUG_manoeuvre3D
# #+CAPTION: Aeroelastic steady equilibrium for increasing angle of attack manoeuvre
# #+ATTR_LATEX: :width 0.95\textwidth 
# [[file:figs_ext/monoeuvre3D.pdf]]

# In Fig. [[fig:BUG_manoeuvretip]] the tip of the wing in Fig. [[fig:BUG_manoeuvre3D]] is plotted for various angles-of-attach (AoA), normalized with the wing semi-span (\(b= \)). Comparison against linear analysis is carried out and the tip position in the nonlinear analysis falls down the linear counter part as expected. This highlights the potential need for geometrically nonlinear aeroelastic tools in future aircraft configurations under high loading scenarios. 

# #+NAME: Manoeuvretip

# [[file:main.org::Manoeuvretip][Manoeuvretip]]
name="Manoeuvretip"
fig, figname = fig_out(name)(plot_manoeuvretip)(aoa, uatip, uatip_lin)
figname
# Manoeuvretip ends here

# Dynamic loads at large scale

# [[file:main.org::*Dynamic loads at large scale][Dynamic loads at large scale:1]]
sol_gust1shard = solution.IntrinsicReader("./gust1_eaoShard")
node = 13
points = sol_gust1shard.data.shards_s1.points
gust_wn = 11 # 11 intensity points
gust_w = points[:gust_wn,3]
gust_l = points[::gust_wn,2]
gust_ln = 11  # 11 gust lenght points
x2max = jnp.max(jnp.abs(sol_gust1shard.data.dynamicsystem_s1.X2[:,:, :, node]), axis=1) # points,6
#x2min = jnp.min(sol_gust1shard.data.dynamicsystem_s1.X2[:,:, :, node], axis=1)
x2max_mesh = x2max.reshape((gust_ln, gust_wn,6)) # contour: wn is x, ln is y
#x2min_mesh = x2min.reshape((gust_ln, gust_wn,6))
# Dynamic loads at large scale:1 ends here



# In this final example we perform a dynamic aeroelastic analysis to study the response of the aircraft to multiple 1-cos gusts for varying length, intensity and the density of the airflow. The mach number is kept constant at 0.7. In the examples above the aircraft was clamped while the aircraft is free here. A Runge-Kutta solver is employed to march in time the equations with a time step of $10^{-3}$ and the total number of modes used was 100. Note the large size of the aeroelastic ODE system: 2 \times 100 nonlinear equations plus 5 \times 100 linear equations for the aerodynamic states with 5 poles, plus 4 equations for the quaternion tracking the rigid-body motion, for a combined ODE system of 704 equations.  
# In addition, a total of 512 gusts cases are run concurrently for all possible combinations of 8 gust lengths between 50 and 200 meters, 8 gust intensities between 5 and 25 m/s, and 8 airflow densities between 0.34 and 0.48 Kg/m$^3$. This means that $512 \times 704 = 360448$ equations are being marched in time, in this case for 2 seconds which is enough to capture peak loads. Figs. [[fig:gust_bendingout_torsion]], [[fig:gust_bendingout_shear]] and [[fig:gust_bendingin_shear]] show the load diagrams for the wing root at the maximum gust intensity of 20, varying 16 gust lengths, $L$, in the range previously stated and 8 airflow densities,  with the points plotted as $point = L / L_{max} + \rho_{\infty} / \rho_{max}$. Different load pattern emerge which need further analysis but reflect the importance of running multiple of these simulations to assess the critical loads. 


# #+NAME: GustShard_shear

# [[file:main.org::GustShard_shear][GustShard_shear]]
name="GustShard_shear"
fig, figname = fig_out(name)(plot_gustshard)(gust_w, gust_l, x2max_mesh,
                                             component=2)
figname
# GustShard_shear ends here



# #+NAME: fig:GustShard_shear
# #+CAPTION: Wing-root internal loading, shear force
# #+ATTR_LATEX: :width 0.6\textwidth 
# #+RESULTS: GustShard_shear
# [[file:figs/GustShard_shear.png]]

# #+NAME: GustShard_torsion

# [[file:main.org::GustShard_torsion][GustShard_torsion]]
name="GustShard_torsion"
fig, figname = fig_out(name)(plot_gustshard)(gust_w, gust_l, x2max_mesh,
                                             component=3)
figname
# GustShard_torsion ends here



# #+NAME: fig:GustShard_torsion
# #+CAPTION: Wing-root internal loading, shear force
# #+ATTR_LATEX: :width 0.6\textwidth 
# #+RESULTS: GustShard_torsion
# [[file:figs/GustShard_torsion.png]]

# #+NAME: GustShard_bending

# [[file:main.org::GustShard_bending][GustShard_bending]]
name="GustShard_bending"
fig, figname = fig_out(name)(plot_gustshard)(gust_w, gust_l, x2max_mesh,
                                             component=4)
figname
# GustShard_bending ends here

# Load envelope differentiation

# [[file:main.org::*Load envelope differentiation][Load envelope differentiation:1]]
import feniax.intrinsic.objectives as objectives
sol_gust1forager = solution.IntrinsicReader("./gustforager")
load_jacs = True

if load_jacs:
    jac_rho = jnp.load("./gustforager_epsilonrho/jac_rho.npy")
    jac_length = jnp.load("./gustforager_epsilonlength/jac_length.npy")
    jac_intensity = jnp.load("./gustforager_epsilonintensity/jac_intensity.npy")
else:
    # this is not working proprerly with FD, loose of accuracy in saving the data??
    sol_gust1forager_val = solution.IntrinsicReader("./gustforager_validation")
    sol_gust1forager_erho = solution.IntrinsicReader("./gustforager_epsilonrho")
    sol_gust1forager_elength = solution.IntrinsicReader("./gustforager_epsilonlength")
    sol_gust1forager_eintensity = solution.IntrinsicReader("./gustforager_epsilonintensity")  
    node = 13
    components = [2,3,4]
    t_range = jnp.arange(len(sol_gust1forager.data.dynamicsystem_s1.t))
    points = sol_gust1forager.data.shards_s1.points
    filtered_map = sol_gust1forager.data.forager_shard2adgust.filtered_map
    index = list(sol_gust1forager.data.forager_shard2adgust.filtered_indexes)[0]
    epsilon = 1e-4
    jac_rho = (objectives.X2_MAX(sol_gust1forager_erho.data.dynamicsystem_s1.X2,
                                 jnp.array([node]),
                                 jnp.array(components),
                                 t_range) -
                 objectives.X2_MAX(#sol_gust1forager.data.dynamicsystem_s1.X2[index],
                                   sol_gust1forager_val.data.dynamicsystem_s1.X2,
                                   jnp.array([node]),
                                   jnp.array(components),
                                   t_range)
                 ) / epsilon
    epsilon = 1e-4
    jac_length = (objectives.X2_MAX(sol_gust1forager_elength.data.dynamicsystem_s1.X2,
                                 jnp.array([node]),
                                 jnp.array(components),
                                 t_range) -
                 objectives.X2_MAX(#sol_gust1forager.data.dynamicsystem_s1.X2[index],
                     sol_gust1forager_val.data.dynamicsystem_s1.X2,
                                   jnp.array([node]),
                                   jnp.array(components),
                                   t_range)
                 ) / epsilon
    epsilon = 1e-4
    jac_intensity = (objectives.X2_MAX(sol_gust1forager_eintensity.data.dynamicsystem_s1.X2,
                                       jnp.array([node]),
                                       jnp.array(components),
                                       t_range) -
                     objectives.X2_MAX(#sol_gust1forager.data.dynamicsystem_s1.X2[index],
                         sol_gust1forager_val.data.dynamicsystem_s1.X2,
                                       jnp.array([node]),
                                       jnp.array(components),
                                       t_range)
                 ) / epsilon

jacdiff_rho = jnp.hstack((sol_gust1forager.data.dynamicsystem_scatter0.jac['rho_inf'] -
               jac_rho) / jac_rho)
jacdiff_length = jnp.hstack((sol_gust1forager.data.dynamicsystem_scatter0.jac['length'] -
               jac_length) / jac_length)
jacdiff_intensity = jnp.hstack((sol_gust1forager.data.dynamicsystem_scatter0.jac['intensity'] -
               jac_intensity) / jac_intensity)

jac_dict = dict(rho=jnp.hstack(sol_gust1forager.data.dynamicsystem_scatter0.jac['rho_inf']),
                rho_fd=jnp.hstack(jac_rho),
                rho_diff=jacdiff_rho,
                Length=jnp.hstack(sol_gust1forager.data.dynamicsystem_scatter0.jac['length']),
                Length_fd=jnp.hstack(jac_length),
                Length_diff=jacdiff_length, Intensity=jnp.hstack(sol_gust1forager.data.dynamicsystem_scatter0.jac['intensity']),
                Intensity_fd=jnp.hstack(jac_intensity),
                Intensity_diff=jacdiff_intensity
                )
df_jac = pd.DataFrame(jac_dict, index=['Shear', 'Torsion', 'Bending'])
#df_jac = df_jac.rename()
# Load envelope differentiation:1 ends here



# Now that dynamic load envelopes have been shown, the interest is to be able to obtain derivatives at the critical points as described in Sec. [[Differentiable-parallel dynamic loads]]. In opposition to the derivatives of expectations where all the operations are needed in the construction of the computational graph, here only a few of the most problematic cases are required.
# The metrics being tracked are wing-root shear, torsion and out-of plane torsion.
# The parallelisation is set for two gust intensities, two flow densities and 16 gust lengths to cover 1-cos gusts from 50 to 200 m/s with 10 m/s separation between points. Rather than a realistic example, this is set to test the machinery of the forager pattern and verify it can indeed discover critical load cases and automatically compute gradients. The gradient of these critical cases is also calculated with respect to the flow density, gust length and intensity (thus they are not only the parameters for the parallelisation but are also chosen to be the input variables of the gradients, though any other input such as FE matrices could have been chosen). By looking at Figs. [[fig:GustShard_shear]]-[[GustShard_bending]], we can identify maximum loads at around 65, 75, 115 m/s gust lengths for shear, torsion and bending.  
# Of the 64 cases analyzed by the forager program, it picked those with higher intensity and flow density as expected, but only 2 gust lengths of 70 m/s for the shear and torsion, and 110 m/s for the out-of-plane bending. A finer discretization in the parallelization can be setup to be closer to the actual maximum, in which case each output gets a unique critical gust.  
# Once those peaks are found, the algorithms triggers the sensitivity analysis, for which we have shown a verification for the 70 m/s gust with maximum density and intensity in Table. This 

# #+CAPTION: 
# #+ATTR_LATEX: :center t
# #+NAME: table:times_gust




# [[file:main.org::*Load envelope differentiation][Load envelope differentiation:2]]
tabulate(df_jac, headers=df_jac.columns, tablefmt='orgtbl',
         #columns=["\(\rho_{\inf} \)", "\(\rho_{\inf}\) FD", "\(\Delta \)"]
         )
# Load envelope differentiation:2 ends here
