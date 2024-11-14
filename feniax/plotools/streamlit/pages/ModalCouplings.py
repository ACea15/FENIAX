import feniax.plotools.streamlit.intrinsic as sti
import streamlit as st
import importlib
importlib.reload(sti)
import feniax.plotools.streamlit.theory as stt

st.set_page_config(
    page_title="Intrinsic modal couplings",
    page_icon="🚁",
    layout="wide"
)

stt.intrinsic_couplings()
st.divider()
st.header('Modal couplings')

st.link_button("Code","https://github.com/ACea15/FENIAX/blob/a54b758c10b53e203268a810d6bf813160b34320/fem4inas/intrinsic/couplings.py#L9")


st.divider()
sti.df_couplings(st.session_state.sol)
