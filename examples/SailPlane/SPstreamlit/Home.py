#Import the required Libraries
import streamlit as st
from fem4inas.preprocessor import solution
import importlib
import fem4inas.plotools.streamlit.intrinsic as sti
importlib.reload(sti)

st.set_page_config(
    page_title="Home page",
    page_icon="🖥️",
    layout="wide"
    )

sti.home()
if 'sol' not in st.session_state:
    sol_path =  "../results_2023-10-23_08:48:10"
    sol = solution.IntrinsicSolution(sol_path)
    sol.load_container("Modes")
    sol.load_container("Couplings")
    sol.load_container("StaticSystem", label="_s1")
    st.session_state['sol'] = sol
