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


'''
- Next milestones
    * Automatic differentiation of the solution. 
    For instance, how the peak laods due to a nonlinear gust response will vary with design parameters.
    * DLM panels updating and ML exploration

- Points to make a decision
    * An FE model of an aircraft (tail included)
'''
