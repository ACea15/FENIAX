import feniax.plotools.streamlit.intrinsic as sti
import streamlit as st
import feniax.plotools.streamlit.theory as stt

st.set_page_config(
    page_title="Initial model geometry",
    page_icon="🛫",
    layout="wide"
)

stt.fe_reduction()

st.divider()
sti.df_geometry(st.session_state.config.fem)
st.divider()
sti.sys_3Dconfiguration0(st.session_state.config)
st.divider()
left, = st.columns(1)
if left.button("Click to see FE matrices", use_container_width=True):

    sti.fe_matrices(st.session_state.config.fem)
