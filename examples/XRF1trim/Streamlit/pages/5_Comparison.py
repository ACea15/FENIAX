import fem4inas.plotools.streamlit.intrinsic as sti
from fem4inas.preprocessor import solution
import fem4inas.preprocessor.configuration as configuration

import streamlit as st
import importlib
importlib.reload(sti)

st.set_page_config(
    page_title="Comparison",
    page_icon="🚁",
    layout="wide"
)


Csol = dict()
Cconfig = dict()
st.markdown("""
### Select folders with results for postprocessing
""")
selected_folders = sti.multifolder_selector('../')
if len(selected_folders) > 0:
    #breakpoint()
    for si in selected_folders:
        #st.write('Solution Folder `%s`' % solfolder)
        sol = solution.IntrinsicReader(si)
        config = configuration.Config.from_file(f"{si}/config.yaml")
        Csol[si.name] = sol
        Cconfig[si.name] = config
    st.session_state['Csol'] = Csol
    st.session_state['Cconfig'] = Cconfig
    sti.systems_comparison(st.session_state['Csol'],
                              st.session_state['Cconfig'])
#sti.systems(st.session_state.sol, st.session_state.config)
