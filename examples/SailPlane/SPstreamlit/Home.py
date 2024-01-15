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


# sti.home()
# video_file = open('/Users/ac5015/postdoc2/Papers/Scitech2024/out4.mp4', 'rb')
# video_bytes = video_file.read()
# st.video(video_bytes)

# st.divider()
# st.image('/Users/ac5015/projects/FEM4INAS/docs/reports/scitech24/figs/classes_architecture.png',
#          caption='Classes architecture')

# st.divider()
# st.image('/Users/ac5015/projects/FEM4INAS/docs/reports/oct23/img/aircraft_process.pdf',
#          caption='Aircraft process')



sti.home()
st.divider()
st.markdown("""
### Select a folder with results for postprocessing
""")
solfolder = "../results_struct"
selected_folder = sti.file_selector('../')
if selected_folder is not None:
    solfolder = selected_folder
st.write('Solution Folder `%s`' % solfolder)
if solfolder is not None:
    sol = solution.IntrinsicReader(solfolder)
    config = configuration.Config.from_file(f"{solfolder}/config.yaml")
    st.session_state['sol'] = sol
    st.session_state['config'] = config
