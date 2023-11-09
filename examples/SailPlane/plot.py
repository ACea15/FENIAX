import seaborn as sns
import pathlib
from fem4inas.preprocessor import solution
#Import the required Libraries
import streamlit as st
import pyvista as pv
from stpyvista import stpyvista
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px

# Apply the default theme
#sns.set_theme()

sol_path =  "./results_2023-11-09_07:47:53"
sol = solution.IntrinsicSolution(sol_path)
sol.load_container("Modes")
sol.load_container("Couplings")
sol.load_container("StaticSystem", label="_s1")

st.set_page_config(layout="wide")

# Functions for each of the pages
def home(uploaded_file):
    if uploaded_file:
        st.header('Begin exploring the data using the menu on the left')
    else:
        st.header('To begin please upload a file')

def modes():
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

def couplings():
    cube = pv.Cube()
    cube.cell_data['myscalars'] = range(6)

    other_cube = cube.copy()
    other_cube.point_data['myscalars'] = range(8)

    pl = pv.Plotter(shape=(1, 2), border_width=1)
    pl.subplot(0, 1)
    pl.add_mesh(cube, cmap='coolwarm')
    pl.view_isometric()
    #pl.subplot(0, 1)
    pl.add_mesh(other_cube, cmap='coolwarm')
    pl.view_isometric()
    
    ## Pass a key to avoid re-rendering at each time something changes in the page
    stpyvista(pl, key="pv_cube")

def data_header():
    st.header('Header of Dataframe')
    st.write(df.head())

def displayplot():
    st.header('Plot of Data')
    
    fig, ax = plt.subplots(1,1)
    ax.scatter(x=df['Depth'], y=df['Magnitude'])
    ax.set_xlabel('Depth')
    ax.set_ylabel('Magnitude')
    
    st.pyplot(fig)

def interactive_plot():
    col1, col2 = st.columns(2)
    
    x_axis_val = col1.selectbox('Select the X-axis', options=df.columns)
    y_axis_val = col2.selectbox('Select the Y-axis', options=df.columns)

    plot = px.scatter(df, x=x_axis_val, y=y_axis_val)
    st.plotly_chart(plot, use_container_width=True)


# Add a title and intro text
st.title('Sail Plane results')
st.text('Hello world')

# Sidebar setup
st.sidebar.title('Sidebar')
upload_file = st.sidebar.file_uploader('Upload a file containing earthquake data')
#Sidebar navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select what you want to display:', ['Home','Modes', 'Couplings', 'Solution'])

# Check if file has been uploaded
if upload_file is not None:
    df = pd.read_csv(upload_file)

# Navigation options
if options == 'Home':
    home(upload_file)
elif options == 'Modes':
    modes()
elif options == 'Couplings':
    couplings()
elif options == 'Solution':
    ...
    #solution()