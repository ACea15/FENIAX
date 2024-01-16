import fem4inas.plotools.streamlit.intrinsic as sti
import streamlit as st
import importlib
importlib.reload(sti)

st.set_page_config(
    page_title="Intrinsic modal couplings",
    page_icon="🚁",
    layout="wide"
)

st.header('Modal couplings')

st.link_button("Code","https://github.com/ACea15/FEM4INAS/blob/a54b758c10b53e203268a810d6bf813160b34320/fem4inas/intrinsic/couplings.py#L9")

st.subheader('Tensor definitions')
st.markdown(
        """
        - Alphas must equal the identity matrix
        """
)

st.latex(r"""
\begin{align}
\alpha_{1}^{jl} & = \langle \pmb{\phi}_{1j}, \pmb{\psi}_{1l}\rangle = \delta^{jl} \\
\alpha_{2}^{jl} & = \langle \pmb{\phi}_{2j}, \pmb{\psi}_{2l}\rangle = \delta^{jl}
\end{align}
""")

st.markdown(
        """
        - Gammas give the nonlinear inertia and strain couplings
        """
)

st.latex(r"""
\begin{align}
\Gamma_{1}^{jkl} & = \langle \pmb{\phi}_{1j}, \mathcal{L}_1(\pmb{\phi}_{1k})\pmb{\psi}_{1l}\rangle,  \\
\Gamma_{2}^{jkl} & = \langle \pmb{\phi}_{1j}, \mathcal{L}_2(\pmb{\phi}_{2k})\pmb{\psi}_{2l}\rangle,
\end{align}
""")

st.divider()

sti.df_couplings(st.session_state.sol)
