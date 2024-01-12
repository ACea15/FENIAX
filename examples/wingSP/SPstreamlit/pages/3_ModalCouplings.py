import fem4inas.plotools.streamlit.intrinsic as sti
import streamlit as st
import importlib
importlib.reload(sti)

st.set_page_config(
    page_title="Intrinsic modal couplings",
    page_icon="🚁",
    layout="wide"
)

sti.df_couplings(st.session_state.sol)
