import fem4inas.plotools.streamlit.intrinsic as sti
import streamlit as st

st.set_page_config(
    page_title="Intrinsic modal shapes",
    page_icon="🛫",
    layout="wide"
)

sti.df_modes(st.session_state.sol)
