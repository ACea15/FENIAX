#Import the required Libraries
import streamlit as st
from fem4inas.preprocessor import solution
import fem4inas.preprocessor.configuration as configuration
import importlib
import fem4inas.plotools.streamlit.intrinsic as sti
importlib.reload(sti)

st.set_page_config(
    page_title="Home page",
    page_icon="🖥️",
    layout="wide"
    )

sti.home()
st.markdown("""
### Select a folder with results for posprocessing
""")
solfolder = None
solfolder = sti.file_selector('../')
st.write('Solution Folder `%s`' % solfolder)
if solfolder is not None:
    sol = solution.IntrinsicSolution(solfolder)
    sol.load_container("Modes")
    sol.load_container("Couplings")
    sol.load_container("DynamicSystem", label="_s1")
    config = configuration.Config.from_file(f"{solfolder}/config.yaml")
    st.session_state['sol'] = sol
    st.session_state['config'] = config
