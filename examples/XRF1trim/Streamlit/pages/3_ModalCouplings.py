import fem4inas.plotools.streamlit.intrinsic as sti
import streamlit as st
import importlib
importlib.reload(sti)
import fem4inas.plotools.streamlit.theory as stt

st.set_page_config(
    page_title="Intrinsic modal couplings",
    page_icon="🚁",
    layout="wide"
)

stt.intrinsic_couplings()
st.divider()

sti.df_couplings(st.session_state.sol)
