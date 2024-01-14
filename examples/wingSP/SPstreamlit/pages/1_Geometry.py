import fem4inas.plotools.streamlit.intrinsic as sti
import streamlit as st

st.set_page_config(
    page_title="Initial model geometry",
    page_icon="🛫",
    layout="wide"
)

sti.df_geometry(st.session_state.config.fem)
sti.sys_3Dconfiguration0(st.session_state.config)
sti.fe_matrices(st.session_state.config.fem)
