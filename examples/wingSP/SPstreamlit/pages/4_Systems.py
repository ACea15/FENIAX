import fem4inas.plotools.streamlit.intrinsic as sti
import streamlit as st
import importlib
importlib.reload(sti)

st.set_page_config(
    page_title="Systems",
    page_icon="🚁",
    layout="wide"
)


sti.systems(st.session_state.sol, st.session_state.config)
