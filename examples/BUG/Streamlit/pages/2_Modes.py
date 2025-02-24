import feniax.plotools.streamlit.intrinsic as sti
import feniax.plotools.streamlit.theory as stt

import streamlit as st
import importlib
importlib.reload(sti)

st.set_page_config(
    page_title="Intrinsic modal shapes",
    page_icon="🛫",
    layout="wide"
)

stt.intrinsic_modes()
st.divider()
st.header('Intrinsic modal data')
st.link_button("Code","https://github.com/ACea15/FENIAX/blob/master/feniax/intrinsic/modes.py#L190")

sti.df_modes(st.session_state.sol,
             st.session_state.config)
