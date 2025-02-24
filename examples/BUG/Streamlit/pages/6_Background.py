#Import the required Libraries
import streamlit as st
from feniax.preprocessor import solution
import feniax.preprocessor.configuration as configuration
import importlib
import feniax.plotools.streamlit.intrinsic as sti
importlib.reload(sti)
import feniax

st.set_page_config(
    page_title="Background page",
    page_icon="🖥️",
    layout="wide"
    )

st.title("Background theory")

st.header("Solution process")
path = feniax.PATH / "../docs/images"
st.image(str(path / "aircraft_process2.png"),
         caption="Steps for the structural solution")
st.divider()

st.header("Theoretical assumptions of the solution")
st.image(str(path / "reality2NMROM2.png"),
         caption="Underlying assumptions in the solution")
st.divider()

st.header("Implementation in JAX")
st.image(str(path / "jaxlogo.png"),
         caption="JAX capabilities")
st.divider()
