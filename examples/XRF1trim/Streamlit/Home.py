#Import the required Libraries
import streamlit as st
from fem4inas.preprocessor import solution
import fem4inas.preprocessor.configuration as configuration
import importlib
import fem4inas.plotools.streamlit.intrinsic as sti
importlib.reload(sti)
from streamlit_pdf_viewer import pdf_viewer
import fem4inas

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

st.title("XRF1 Aircraft")
sp_path = str(fem4inas.PATH / "../docs/images/SailPlane2.png")
#pdf_viewer(sp_path)
#st.image(sp_path, caption="Aircraft FE model", use_column_width=False, width=550)
# st.text("This is what we can do!!")
# st.image(str(fem4inas.PATH / "../docs/images/SailPlane3D_front.png"),
#          caption="Static response (Comparison with Nastran Full FE analysis)", use_column_width=True)
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
