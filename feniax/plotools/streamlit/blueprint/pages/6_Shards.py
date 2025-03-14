import feniax.plotools.streamlit.intrinsic as sti
from feniax.preprocessor import solution
import feniax.preprocessor.configuration as configuration

import streamlit as st
import importlib
# importlib.reload(sti)

st.set_page_config(
    page_title="Shards systems",
    page_icon="🎛️",
    layout="wide"
)


Csol = dict()
Cconfig = dict()
Csolshard = dict()
Cconfigshard = dict()
st.markdown("""
### Select folders with results for postprocessing
""")
selected_shardfolders = sti.multifolder_selector('../', key=1)

if len(selected_shardfolders) > 0:
    #breakpoint()
    for si in selected_shardfolders:
        #st.write('Solution Folder `%s`' % solfolder)
        solshard = solution.IntrinsicReader(si)
        configshard = configuration.Config.from_file(f"{si}/config.yaml")
        Csolshard[si.name] = solshard
        Cconfigshard[si.name] = configshard
    st.session_state['Csolshard'] = Csolshard
    st.session_state['Cconfigshard'] = Cconfigshard
    
    selected_folders = sti.multifolder_selector('../')
    if len(selected_folders) > 0:
        for si in selected_folders:
            #st.write('Solution Folder `%s`' % solfolder)
            sol = solution.IntrinsicReader(si)
            config = configuration.Config.from_file(f"{si}/config.yaml")
            Csol[si.name] = sol
            Cconfig[si.name] = config
        st.session_state['Csol'] = Csol
        st.session_state['Cconfig'] = Cconfig
        
        sti.systems_shard(st.session_state['Csolshard'],
                          st.session_state['Cconfigshard'],
                          st.session_state['Csol'],
                          st.session_state['Cconfig'])
    else:
        sti.systems_shard(st.session_state['Csolshard'],
                          st.session_state['Cconfigshard'])        
        
#sti.systems(st.session_state.sol, st.session_state.config)
