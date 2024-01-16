import fem4inas.plotools.streamlit.intrinsic as sti
import streamlit as st
import importlib
importlib.reload(sti)

st.set_page_config(
    page_title="Systems",
    page_icon="🚁",
    layout="wide"
)



st.latex(r"""
\begin{equation}
\begin{split}
\dot{q}_{1j} &= \delta^{ji}\omega_{i}q_{2i}-\Gamma^{jik}_{1}q_{1i}q_{1k}-\Gamma^{jik}_{2}q_{2i}q_{2k}+ \eta_{j}  \\
\dot{q}_{2j} &= -\delta^{ji}\omega_{i}q_{1i} + \Gamma_2^{ijk}q_{1i}q_{2k}
\end{split}
\end{equation}
""")


st.latex(r"""
\begin{equation}
\begin{split}
\begin{cases}
 \dot{q}_{1i} &= \hat{\Omega}^{ij} q_{2j}
              - \hat{\Gamma}_{1}^{ijk}q_{1j}q_{1k}
              - \hat{\Gamma}_{2}^{ijk}q_{2j}q_{2k} 
              + \hat{\mathcal{A}}^{ij}_{0}q_{0j}
              + \hat{\mathcal{A}}^{ij}_{1}q_{1j}  \\
            & +\hat{\mathcal{A}}^{is}_{g0}v_{gs}
              + \hat{\mathcal{A}}^{is}_{g1}\dot{v}_{gs}
              + \hat{\mathcal{A}}^{is}_{g2}\ddot{v}_{gs}        
              + \left(\mathcal{M}^{-1}\right)^{ij} \delta^{pp} \lambda_{pj}
              + \hat{\eta}_{gi} + \hat{\eta}_{fi}\\
\dot{q}_{2i} &= -\delta^{ij}\Omega_j q_{1j}+ \Gamma_2^{jik}q_{1j}q_{2k}\\
 \dot{\lambda}_{p,i} &= \hat{\mathcal{A}}^{ij}_{p+2}q_{1j}
                       + \hat{\mathcal{A}}^{is}_{g,p+2}\dot{v}_{gs}
                      -\frac{2U_\infty\gamma_p}{c}\lambda_{p,i} 
\end{cases}
\end{split}
\end{equation}
""")

sti.systems(st.session_state.sol, st.session_state.config)
