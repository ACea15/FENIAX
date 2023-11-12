import streamlit as st
import fem4inas.plotools.nastranvtk.bdfdef as bdfdef
import pandas as pd
import pyvista as pv
import pathlib
import plotly.express as px
from enum import Enum

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


def build_op2modes():
    bdfdef.vtk_fromop2(bdf_file, op2_file, scale = 100., modes2plot=None)

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
    
def df_modes(sol):
    st.header('Intrinsic modal data')
    names = [mi for mi in dir(sol.data.modes) if mi[0] != "_"]
    modes_names = [ni for ni in names if ni[0] == "p"]
    col1, col2 = st.columns(2)
    num_modes = len(sol.data.modes.phi1)
    st.subheader("Mode display df")
    mname = col1.selectbox('Select mode type', options=modes_names)
    mnumber = col2.selectbox('Select Mode number', options=range(num_modes))
    mvalue = getattr(sol.data.modes, mname)
    df = pd.DataFrame(mvalue[mnumber].T, columns=['x', 'y', 'z', 'rx', 'ry', 'rz'])
    st.table(df)

def df_couplings(sol):
    st.header('Modal couplings')
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
    st.plotly_chart(fig, use_container_width=True)
    #st.pyplot(fig)
    with st.expander("See Data Frame"):
        st.table(df)

def sys_states(q):

    ...

def sys_positions():
    ...
    
def sys_velocities():
    ...

def sys_strains():
    ...

def sys_internalforces():
    ...

def systems(sol):
    st.header('Systems')
    show = Enum('States', ['STATES', 'POSITIONS','VELOCITIES',
                            'STRAINS', 'INTERNALFORCES'])
    sys_names = [mi for mi in dir(sol.data) if (mi[0] != "_" and
                                            "system" in mi)]
    sys_option = st.selectbox(
        "Select a system for the analysis",
        sys_names,
        index=None,
        placeholder="Select system...",
    )

    st.write('System being analysed:', sys_option)    
    field_option = st.sidebar.radio('Select what you want to display:',
                                    show._member_names_)
    match field_option:
        case show.STATES.name:
            sys_states()
        case show.POSITIONS.name:
            sys_positions()
        case show.VELOCITIES.name:
            sys_velocities()
        case show.STRAINS.name:
            sys_strains()
        case show.INTERNALFORCES.name:
            sys_internalforces()
    
# def df_system_X(sol, ):
#     st.header('Systems')
#     show = Enum('States', ['STATES', 'POSITIONS','VELOCITIES',
#                             'STRAINS', 'INTERNALFORCES'])
#     names = [mi for mi in dir(sol.data) if (mi[0] != "_" and
#                                             "system" in mi)]
#     option = st.sidebar.radio('Select what you want to display:',
#                               State._member_names_)
#     match :
#         case show.STATES:
#             ...
#         case show.POSITIONS:
#             ...
#         case show.VELOCITIES:
#             ...
#         case show.STRAINS:
#             ...
#         case show.INTERNALFORCES:
#             ...


