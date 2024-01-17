#Import the required Libraries
import streamlit as st
from fem4inas.preprocessor import solution
import fem4inas.preprocessor.configuration as configuration
import importlib
import fem4inas.plotools.streamlit.intrinsic as sti
importlib.reload(sti)
import fem4inas

st.set_page_config(
    page_title="Home page",
    page_icon="🖥️",
    layout="wide"
    )

sti.home()
st.divider()
st.title("XRF1 plane")
st.markdown("Full aircraft model built to industry standards")
wingsp_path = str(fem4inas.PATH / "../docs/images/xrf1-model3.jpg")
st.image(wingsp_path, caption="Aircraft aeroelastic model")


st.text("This is what we can do!!")
video_file = open("/Users/ac5015//postdoc2/img/xrf1_gust_30frc.mp4", 'rb')
video_bytes = video_file.read()
st.video(video_bytes)
st.divider()

st.markdown("""
### Select a folder with results for postprocessing
""")
solfolder = "../resultsGust_g1i2"
selected_folder = sti.file_selector('../')
if selected_folder is not None:
    solfolder = selected_folder
st.write('Solution Folder `%s`' % solfolder)
if solfolder is not None:
    sol = solution.IntrinsicReader(solfolder)
    config = configuration.Config.from_file(f"{solfolder}/config.yaml")
    st.session_state['sol'] = sol
    st.session_state['config'] = config
