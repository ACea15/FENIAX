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
st.link_button("Code","https://github.com/ACea15/FENIAX/blob/0958ca92b55073d799668136ae4d5132687f8969/feniax.intrinsic/modes.py#L192")

sti.df_modes(st.session_state.sol,
             st.session_state.config)
