#Import the required Libraries
import streamlit as st
from feniax.preprocessor import solution
import feniax.preprocessor.configuration as configuration
import importlib
import feniax.plotools.streamlit.intrinsic as sti
importlib.reload(sti)
import feniax

st.set_page_config(
    page_title="Home page",
    page_icon="🖥️",
    layout="wide"
    )

sti.home()
st.divider()
st.title("")
# st.markdown("This is a simplified box wing made of composite shells, lumped masses along the load paths and interpolation elements (RBE3s) connecting those to the main structure.")
# wingsp_path = str(feniax.PATH / "../docs/images/wingSP5b.png")
# st.image(wingsp_path, caption="Wing box FE model")


st.text("This is what we can do!!")

st.markdown("""
### Select a folder with results for postprocessing
""")
solfolder = "../results1"
selected_folder = sti.file_selector('../')
if selected_folder is not None:
    solfolder = selected_folder
st.write('Solution Folder `%s`' % solfolder)
if solfolder is not None:
    sol = solution.IntrinsicReader(solfolder)
    config = configuration.Config.from_file(f"{solfolder}/config.yaml")
    st.session_state['sol'] = sol
    st.session_state['config'] = config
