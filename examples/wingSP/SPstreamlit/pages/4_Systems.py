import fem4inas.plotools.streamlit.intrinsic as sti
import streamlit as st
import importlib
importlib.reload(sti)
import fem4inas.plotools.streamlit.theory as stt

st.set_page_config(
    page_title="Systems",
    page_icon="🚁",
    layout="wide"
)

stt.intrinsic_systems()

sti.systems(st.session_state.sol, st.session_state.config)