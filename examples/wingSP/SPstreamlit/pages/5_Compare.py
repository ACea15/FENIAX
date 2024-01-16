#Import the required Libraries
import streamlit as st
from fem4inas.preprocessor import solution
import fem4inas.preprocessor.configuration as configuration
import importlib
import fem4inas.plotools.streamlit.intrinsic as sti
importlib.reload(sti)
import fem4inas

st.set_page_config(
    page_title="Compare solutions",
    page_icon="🖥️",
    layout="wide"
    )
