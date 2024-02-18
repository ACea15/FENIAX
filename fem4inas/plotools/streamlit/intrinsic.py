import streamlit as st
import fem4inas.plotools.nastranvtk.bdfdef as bdfdef
import fem4inas.plotools.utils as putils
import fem4inas.plotools.uplotly as uplotly
import fem4inas.plotools.upyvista as upyvista
import jax.numpy as jnp
import pandas as pd
import pyvista as pv
import pathlib
import plotly.express as px
from enum import Enum
import os
from stpyvista import stpyvista

def home():
    #st.write("# Welcome to FEM4INAS ✈️")

    #st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        FEM4INAS is an aeroelastic toolbox written and parallelized in Python,
        which acts as a post-processor of commercial software such as MSC Nastran.
        Arbitrary FE models built for linear aeroelastic analysis are enhanced with
        geometric nonlinear effects, flight dynamics and linearized state-space
        solutions about nonlinear equilibrium.
        
        **👈 Select a tab from the sidebar** to jump to the postprocessing analysis
        ### Want to learn more?
        - Check out our [Repo](https://github.com/ACea15/FEM4INAS)
        ### See more complex demos
        - In the folder [examples](https://github.com/ACea15/FEM4INAS/tree/master/examples)
        """
    )

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames,None)
    if selected_filename is not None:
        return os.path.join(folder_path, selected_filename)
    else:
        return selected_filename

def multifolder_selector(folder_path='.'):
    folder_path = pathlib.Path(folder_path)
    names = [folder_path / pathlib.Path(fi) for fi in os.listdir(folder_path)]
    #breakpoint()
    folder_names = [ni for ni in names if ni.is_dir()]
    selected_foldernames = st.multiselect('Select solution folders', [fi.name for fi in folder_names],None)
    if selected_foldernames is not None:
        folders = [folder_path / pathlib.Path(sfi) for sfi in selected_foldernames]
        return folders
    else:
        return selected_foldernames

# def build_op2modes():
#     bdfdef.vtk_fromop2(bdf_file, op2_file, scale = 100., modes2plot=None)

def show_vtu(vtu_folder, stream_out="streamlit"):
    """
    Reads .vtu files and converts them to pyvista
    """

    folder = pathlib.Path(vtu_folder)
    pl = pv.Plotter()
    reader1 = pv.get_reader(folder / "CQUAD4.vtu")
    mesh1 = reader1.read()
    pl.add_mesh(mesh1)
    reader2 = pv.get_reader(folder / "CBAR.vtu")
    mesh2 = reader2.read()
    pl.add_mesh(mesh2)
    if stream_out == "streamlit":
        ## Pass a key to avoid re-rendering at each time something changes in the page
        stpyvista(pl, key="pv_cube")
    elif stream_out == "pyvista":
        pl.show()

def df_geometry(fem):

    st.header('Geometry and FE Models')
    st.subheader('Condensed model')
    st.table(fem.df_grid)
    
def fe_matrices(fem):
    
    st.subheader('Ka and Ma input matrices')
    st.write("Stiffness matrix")
    fig = px.imshow(fem.Ka, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("See Stiffness matrix"):
        st.table(fem.Ka)
    st.divider()
    st.write("Mass matrix")
    fig2 = px.imshow(fem.Ma, aspect="auto")
    st.plotly_chart(fig2, use_container_width=True)
    with st.expander("See mass matrix"):
        st.table(fem.Ma)

def df_modes(sol, config):
    
    names = [mi for mi in dir(sol.data.modes) if mi[0] != "_"]
    modes_names = [ni for ni in names if ni[0] == "p"]
    col1, col2 = st.columns(2)
    num_modes = len(sol.data.modes.phi1)
    st.subheader("Mode display df")
    mname = col1.selectbox('Select mode type', options=modes_names)
    mnumber = col2.selectbox('Select Mode number', options=range(num_modes))
    mvalue = getattr(sol.data.modes, mname)
    scale = st.sidebar.slider("Select a scaling", min_value=-100., max_value=100.,
                   value=0., step=0.1, format=None, key=None, help=None,
                   on_change=None, args=None, kwargs=None, disabled=False,
                   label_visibility="visible")
    
    df = pd.DataFrame(mvalue[mnumber].T * scale, columns=['x', 'y', 'z', 'rx', 'ry', 'rz'])
    st.table(df)
    st.divider()
    fig = modes_3Dconfiguration(mode=mvalue[mnumber] * scale, config=config, mode_label=mname,settings=None)
    #breakpoint()
    st.plotly_chart(fig, use_container_width=False)

def df_couplings(sol):
    
    names = [mi for mi in dir(sol.data.couplings) if mi[0] != "_"]
    col1, col2 = st.columns(2)
    num_modes = len(sol.data.couplings.gamma1)
    st.subheader("Display df")
    mname = col1.selectbox('Select Coupling', options=names)
    mvalue = getattr(sol.data.couplings, mname)
    if mname in ['gamma1', 'gamma2']:
        mnumber = col2.selectbox('Select Axis', options=range(num_modes))
        df = pd.DataFrame(mvalue[mnumber])
    else:
        df = pd.DataFrame(mvalue)
    st.subheader("Coupling heat map")
    fig = px.imshow(df.abs(), aspect="auto")
    st.plotly_chart(fig, use_container_width=False)
    #st.pyplot(fig)
    with st.expander("See Data Frame"):
        st.table(df)

def sys_states(solsys):

    statei = None
    sinput = None
    q = solsys.q
    if hasattr(solsys, "t"):
        t = solsys.t
        mode = "lines"
    else:
        t = list(range(len(q)))
        mode = "lines+markers"
    statei = st.selectbox(
        "Select a state for plotting",
        range(len(q[0])),
        index=None,
        placeholder="Pick one...",
    )
    if statei is not None:
        #statei = 0#col1s.selectbox('Select state', options=range(len(q[0])))
        fig = uplotly.lines2d(t, q[:, statei], None,
                              dict(name="NMROM",
                                   mode=mode,
                                   line=dict(color="navy")
                                   ),
                              dict(title="Solution state evolution"))
        fig.update_xaxes(title='time [s]')
        fig.update_yaxes(title='q')

        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Select multiple states"):
        sinput = st.text_input("Give a list ")
        if sinput is not None and len(sinput) > 0:
            linput = jnp.array([int(si) for si in sinput.split(",")])
            fig2 = uplotly.lines2d(t, q[:, linput[0]], None,
                                  dict(name=f"State {linput[0]}",
                                       line=dict(color="navy"),
                                       mode=mode
                                       ),
                                  dict(title="qs"))
            for li in linput[1:]:
                fig2 = uplotly.lines2d(t, q[:, li], fig2,
                                       dict(name=f"State {li}",
                                            mode=mode))
                fig2.update_xaxes(title='time [s]')
                fig2.update_yaxes(title='q')

            st.plotly_chart(fig2, use_container_width=True)

def sys_states_comparison(solsys):

    labels = list(solsys.keys())
    syss = list(solsys.values())
    q = [ss.q for ss in syss]
    if hasattr(syss[0], "t"):
        t = [ss.t for ss in syss]
        mode = "lines"
    else:
        t = [list(range(len(qi))) for qi in q]
        mode = "lines+markers"

    statei = None
    statei = st.selectbox(
        "Select a state for plotting",
        range(len(q[0][0])),
        index=None,
        placeholder="Pick one...",
    )
    if statei is not None:
        #statei = 0#col1s.selectbox('Select state', options=range(len(q[0])))
        fig = None
        for i, (ti, qi) in enumerate(zip(t, q)):
            fig = uplotly.lines2d(ti, qi[:, statei], fig,
                                  dict(name=labels[i],
                                       mode=mode,
                                       #line=dict(color="navy")
                                       ),
                                  dict(title="Solution state evolution"))
        fig.update_xaxes(title='time [s]')
        fig.update_yaxes(title='q')

        st.plotly_chart(fig, use_container_width=True)

def sys_X(X, solsys, label='X'):

    nodei = None
    componenti = None
    col1, col2 = st.columns(2)
    ntimes, ncomponents, nnodes = X.shape
    if hasattr(solsys, "t"):
        t = solsys.t
        mode = "lines"
    else:
        t = list(range(ntimes))        
        mode = "lines+markers"

    nodei = col1.selectbox('Select a node', options=range(nnodes))
    componenti = col2.selectbox('Select a component', options=range(ncomponents))

    if nodei is not None:
        fig = uplotly.lines2d(t, X[:, componenti, nodei], None,
                              dict(name="NMROM",
                                   line=dict(color="navy"),
                                   mode=mode
                                   ),
                              dict(title='Time evolution'))
        fig.update_xaxes(title='time [s]',
                         # tickfont = dict(size=16),
                         # titlefont=dict(size=16),
                         # mirror=True,
                         # ticks='outside',
                         # showline=True,
                         # linecolor='black',
                         # gridcolor='lightgrey'
                         )
        fig.update_yaxes(title=label,
                         # tickfont = dict(size=16),
                         # titlefont=dict(size=16),
                         # mirror=True,
                         # ticks='outside',
                         # showline=True,
                         # linecolor='black',
                         # gridcolor='lightgrey'
                         )
        
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Select multiple states"):
        ninput = st.text_input("Give a list of nodes:")
        cinput = st.text_input("Give a list of components")
        if ninput is not None and len(ninput) > 0:
            l_ninput = [int(si) for si in ninput.split(",")]
            if cinput is not None and len(cinput) > 0:
                l_cinput = [int(si) for si in cinput.split(",")]
                fig2 = None
                for ni in l_ninput:
                    for ci in l_cinput:
                        fig2 = uplotly.lines2d(t, X[:, ci, ni],
                                               fig2,
                                               dict(name=f"Node {ni}, component {ci}",
                                                    mode=mode))
                st.plotly_chart(fig2, use_container_width=True)

def sys_X_comparison(X, t, labels, ylabel, mode="lines"):

    fig = None
    # for i, (ti, qi) in enumerate(zip(t, q)):
    nodei = None
    componenti = None
    col1, col2 = st.columns(2)
    ntimes, ncomponents, nnodes = X[0].shape

    nodei = col1.selectbox('Select a node', options=range(nnodes))
    componenti = col2.selectbox('Select a component', options=range(ncomponents))
    #breakpoint()
    if nodei is not None:
        for i, Xi in enumerate(X):
            fig = uplotly.lines2d(t[i], Xi[:, componenti, nodei], fig,
                                  dict(name=labels[i],
                                       #line=dict(color="navy"),
                                       mode=mode
                                       ),
                                  dict(title='Time evolution'))
        fig.update_xaxes(title='time [s]'
                         )
        fig.update_yaxes(title=ylabel
                         )
        st.plotly_chart(fig, use_container_width=True)

def sys_displacements_comp(solsys, config):

    labels = list(solsys.keys())
    Xvs = list(solsys.values())
    if hasattr(Xvs[0], "t"):
        t = [x.t for x in Xvs]
        mode = "lines"
    else:
        t = [list(range(len(qi))) for qi in Xvs[0].q]
        mode = "lines+markers"

    ra = [x.ra - config[labels[i]].fem.X.T for i, x in enumerate(Xvs)]
    sys_X_comparison(ra, t, labels, "Displacements", mode)

def sys_displacements(solsys, config):

    sys_X(solsys.ra - config.fem.X.T, solsys, label='Displacements')

def sys_velocities_comp(solsys):

    Xvs = list(solsys.values())
    if hasattr(Xvs[0], "X1"):
        labels = list(solsys.keys())
        x1 = [x.X1 for x in Xvs]
        t = [x.t for x in Xvs]
        sys_X_comparison(x1, t, labels, "X1")
    else:
        st.text("Static solution!! All velocities are 0")

def sys_velocities(solsys):

    if hasattr(solsys, "X1"):
        sys_X(solsys.X1, solsys, label='X1')
    else:
        st.text("Static solution!! All velocities are 0")
def sys_strains(solsys):

    sys_X(solsys.X3, solsys, label='X3')

def sys_strains_comp(solsys):
    
    labels = list(solsys.keys())
    Xvs = list(solsys.values())
    x3 = [x.X3 for x in Xvs]
    t = []
    for x in Xvs:
        if hasattr(x, "t"):
            t.append(x.t)
            mode = "lines"
        else:
            times, ncomponents, nnodes = x.X3.shape
            t.append(list(range(1, times + 1)))
            mode = "lines+markers"
    sys_X_comparison(x3, t, labels, "X3", mode=mode)

def sys_internalforces(solsys):

    sys_X(solsys.X2, solsys, label='X2')

def sys_internalforces_comp(solsys):

    labels = list(solsys.keys())
    Xvs = list(solsys.values())
    x2 = [x.X2 for x in Xvs]
    t = []
    for x in Xvs:
        if hasattr(x, "t"):
            t.append(x.t)
            mode = "lines"
        else:
            times, ncomponents, nnodes = x.X2.shape
            t.append(list(range(1, times + 1)))
            mode = "lines+markers"
    sys_X_comparison(x2, t, labels, "X2",mode=mode)

def sys_3Dconfiguration(solsys, config, ti=None, labels=None, settings=None):

    icomp = putils.IntrinsicStructComponent(config.fem)
    if labels is not None:
        for li in labels:
            if ti is None:
                icomp.add_solution(solsys.ra, li)
            else:
                icomp.add_solution(solsys.ra[ti], li)
    else:
        if ti is None:
            icomp.add_solution(solsys.ra)
        else:
            icomp.add_solution(solsys.ra[ti])

    if settings is None:
        settings = {}
    fig = uplotly.render3d_multi(icomp,
                                 labels,
                                 **settings)
    fig.update_layout(margin=dict(
        autoexpand=True,
        l=0,
        r=0,
        t=0,
        b=0
    ))

    return fig

def sys_3Dconfiguration_ti(solsys, config, settings=None):

    icomp = putils.IntrinsicStructComponent(config.fem)
    ti = st.sidebar.slider("Select a value", min_value=0, max_value=len(solsys.ra),
                   value=None, step=1, format=None, key=None, help=None,
                   on_change=None, args=None, kwargs=None, disabled=False,
                   label_visibility="visible")
    label = f"{ti}"
    icomp.add_solution(solsys.ra[int(ti)], label_final=label)
    if settings is None:
        settings = {}
    #breakpoint()
    fig = uplotly.render3d_struct(icomp,
                                  label,
                                  **settings)
    fig.update_layout(showlegend=False,width=1000, height=900,
                      margin=dict(
                          autoexpand=True,
                          l=0,
                          r=0,
                          t=0,
                          b=0
                      ))
    fig.update_traces(line=dict(width=1.2, color="navy"),
                      marker=dict(size=1.5))

    return fig

def sys_3Dconfiguration_pv(solsys, config, ti=None, settings=None):
    
    istruct = putils.IntrinsicStruct(config.fem)
    pl = upyvista.render_wireframe(points=config.fem.X, lines=istruct.lines)
    if ti is None:
        istruct.add_solution(solsys.ra)
    else:
        istruct.add_solution(solsys.ra[ti])
    if settings is None:
        settings = {}
    pl = upyvista.render_wireframe(points=config.fem.X, lines=istruct.lines)
    pl.show_grid()
    #pl.view_xy()
    for k, v in istruct.map_ra.items():
        pl = upyvista.render_wireframe(points=v, lines=istruct.lines, pl=pl)

    # ipythreejs does not support scalar bars :(
    pv.global_theme.show_scalar_bar = False

    st.title("A cube")
    st.info("""Code adapted from https://docs.pyvista.org/user-guide/jupyter/pythreejs.html#scalars-support""")

    ## Initialize a plotter object
    plotter = pv.Plotter(window_size=[400,400])

    ## Create a mesh with a cube 
    mesh = pv.Cube(center=(0,0,0))

    ## Add some scalar field associated to the mesh
    mesh['myscalar'] = mesh.points[:, 2]*mesh.points[:, 0]

    ## Add mesh to the plotter
    plotter.add_mesh(mesh, scalars='myscalar', cmap='bwr', line_width=1)

    ## Final touches
    plotter.view_isometric()
    plotter.background_color = 'white'

    ## Send to streamlit
    stpyvista(plotter, key="pv_cube")
    #stpyvista(pl, key="pv_cube")

def sys_3Dconfiguration0(config):
    
    st.subheader('3D reference configuration')
    template=[]
    nnodes = len(config.fem.df_grid)
    x = config.fem.df_grid.x1
    y = config.fem.df_grid.x2
    z = config.fem.df_grid.x3
    componenti = config.fem.df_grid.component[0]
    templatei = []
    for n_i in range(nnodes):
        if config.fem.df_grid.component[n_i] != componenti:
            template.append(templatei)
            componenti = config.fem.df_grid.component[n_i]
            templatei = []
            dados_sim = []
            n_iprev = config.fem.prevnodes[n_i]
            x1 = f'<BR><b>x: </b> {x[n_iprev]}'
            dados_sim.append(x1)
            x2 = f'<BR><b>y: </b> {y[n_iprev]}'
            dados_sim.append(x2)
            x3 = f'<BR><b>z: </b> {z[n_iprev]}'
            dados_sim.append(x3)
            index = f'<BR><b>Index: </b> {n_iprev}'
            dados_sim.append(index)
            fe_orderi = config.fem.df_grid.fe_order[n_iprev]
            fe_index = f'<BR><b>FE_Index: </b> {fe_orderi}'
            dados_sim.append(fe_index)
            Component_i = f'<BR><b>Component: </b> {componenti}'
            dados_sim.append(Component_i)
            dados_sim = ','.join(dados_sim) + '<extra></extra>'
            templatei.append(dados_sim)

        dados_sim = []
        x1 = f'<BR><b>x: </b> {x[n_i]}'
        dados_sim.append(x1)
        x2 = f'<BR><b>y: </b> {y[n_i]}'
        dados_sim.append(x2)
        x3 = f'<BR><b>z: </b> {z[n_i]}'
        dados_sim.append(x3)
        index = f'<BR><b>Index: </b> {n_i}'
        dados_sim.append(index)
        fe_orderi = config.fem.df_grid.fe_order[n_i]
        fe_index = f'<BR><b>FE_Index: </b> {fe_orderi}'
        dados_sim.append(fe_index)
        Component_i = f'<BR><b>Component: </b> {componenti}'
        dados_sim.append(Component_i)
        dados_sim = ','.join(dados_sim) + '<extra></extra>'
        templatei.append(dados_sim)
    template.append(templatei)
    icomp = putils.IntrinsicStructComponent(config.fem)
    #breakpoint()
    fig = uplotly.render3d_struct(icomp,
                                  label="ref1",
                                  scatter_settings=[dict(customdata=ti,
                                                         hovertemplate=ti) for ti in template],
                                  update_traces=dict(line=dict(width=1.2,color="navy"),
                                                     marker=dict(size=1.5)))
    
    # fig = uplotly.render3d_multi(icomp,
    #                              labels=["ref1"],
    #                              scatter_settings=[dict(customdata=ti,
    #                                                     hovertemplate=ti) for ti in template])
    
    fig.update_layout(
        autosize=False,
        margin=dict(
        autoexpand=False,
        l=0,
        r=1.5,
        t=1.5,
        b=0
        ),showlegend=False,width=1000, height=900)
    #fig.update_layout(hoverlabel={"grid": config.fem.df_grid.index})
    st.plotly_chart(fig, use_container_width=False)
    return fig

def systems(sol, config):
    st.header('Systems')
    show = Enum('States', ['STATES', 'DISPLACEMENTS','VELOCITIES',
                            'STRAINS', 'INTERNALFORCES', 'CONFIGURATION3D', 'CONFIGURATION3D_PV'])
    sys_names = [mi for mi in dir(sol.data) if (mi[0] != "_" and
                                            "system" in mi)]
    sys_option = st.selectbox(
        "Select a system for the analysis",
        sys_names,
        index=None,
        placeholder="Select system...",
    )

    st.write('System being analysed:', sys_option)
    if sys_option is not None:
        solsys = getattr(sol.data, sys_option)
        field_option = st.sidebar.radio('Select what you want to display:',
                                        show._member_names_)
        match field_option:
            case show.STATES.name:
                sys_states(solsys)
            case show.DISPLACEMENTS.name:
                sys_displacements(solsys, config)
            case show.VELOCITIES.name:
                sys_velocities(solsys)
            case show.STRAINS.name:
                sys_strains(solsys)
            case show.INTERNALFORCES.name:
                sys_internalforces(solsys)
            case show.CONFIGURATION3D.name:
                # fig = sys_3Dconfiguration(solsys, config)
                #st.plotly_chart(fig, use_container_width=False)
                #st.subheader("Slide")
                fig2 = sys_3Dconfiguration_ti(solsys, config)
                st.plotly_chart(fig2, use_container_width=False)
            case show.CONFIGURATION3D_PV.name:
                sys_3Dconfiguration_pv(solsys, config)

def systems_comparison(sol, config):
    st.header('Comparison')
    show = Enum('States', ['STATES', 'DISPLACEMENTS','VELOCITIES',
                            'STRAINS', 'INTERNALFORCES', 'CONFIGURATION3D', 'CONFIGURATION3D_PV'])
    sys_names = [mi for mi in dir(list(sol.values())[0].data) if (mi[0] != "_" and
                                            "system" in mi)]
    sys_option = st.selectbox(
        "Select a system for the analysis",
        sys_names,
        index=None,
        placeholder="Select system...",
    )

    st.write('System being analysed:', sys_option)
    if sys_option is not None:
        solsys = {k: getattr(solk.data, sys_option) for k, solk in sol.items()}
        field_option = st.sidebar.radio('Select what you want to display:',
                                        show._member_names_)
        #breakpoint()
        match field_option:
            case show.STATES.name:
                sys_states_comparison(solsys)
            case show.DISPLACEMENTS.name:
                sys_displacements_comp(solsys, config)
            case show.VELOCITIES.name:
                sys_velocities_comp(solsys)
            case show.STRAINS.name:
                sys_strains_comp(solsys)
            case show.INTERNALFORCES.name:
                sys_internalforces_comp(solsys)
            case show.CONFIGURATION3D.name:
                # fig = sys_3Dconfiguration(solsys, config)
                #st.plotly_chart(fig, use_container_width=False)
                #st.subheader("Slide")
                fig2 = sys_3Dconfiguration_ti(solsys, config)
                st.plotly_chart(fig2, use_container_width=False)
            case show.CONFIGURATION3D_PV.name:
                sys_3Dconfiguration_pv(solsys, config)

                
def modes_3Dconfiguration(mode, config, mode_label,settings=None):

    icomp = putils.IntrinsicStructComponent(config.fem)
    label = f"{mode_label}"
    if "2" in label:
        mode2plot = (jnp.column_stack([jnp.zeros(3), mode[:3,1:]]) +
                     +config.fem.X.T)
    else:
        mode2plot = mode[:3] + config.fem.X.T
    icomp.add_solution(mode2plot, label_final=label)
    if settings is None:
        settings = {}
    #breakpoint()
    fig = uplotly.render3d_struct(icomp,
                                  label,
                                  **settings)
    fig.update_layout(showlegend=False,width=1000, height=900,
                      margin=dict(
                          autoexpand=True,
                          l=0,
                          r=0,
                          t=0,
                          b=0
                      ))
    fig.update_traces(line=dict(width=1.2, color="navy"),
                      marker=dict(size=1.5))
    #breakpoint()
    return fig

