import fem4inas.plotools.streamlit.intrinsic as sti
import streamlit as st

st.set_page_config(
    page_title="Intrinsic modal shapes",
    page_icon="🛫",
    layout="wide"
)

st.header('Intrinsic modal data')
st.link_button("Code","https://github.com/ACea15/FEM4INAS/blob/0958ca92b55073d799668136ae4d5132687f8969/fem4inas/intrinsic/modes.py#L192")

sti.df_modes(st.session_state.sol,
             st.session_state.config)
