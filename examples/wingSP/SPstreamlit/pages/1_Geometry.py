import fem4inas.plotools.streamlit.intrinsic as sti
import streamlit as st
import fem4inas.plotools.streamlit.theory as stt

st.set_page_config(
    page_title="Initial model geometry",
    page_icon="🛫",
    layout="wide"
)

stt.fe_reduction()

st.divider()
sti.df_geometry(st.session_state.config.fem)
sti.sys_3Dconfiguration0(st.session_state.config)
sti.fe_matrices(st.session_state.config.fem)
