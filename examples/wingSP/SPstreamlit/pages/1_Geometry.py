import fem4inas.plotools.streamlit.intrinsic as sti
import streamlit as st

st.set_page_config(
    page_title="Initial model geometry",
    page_icon="🛫",
    layout="wide"
)

'''
# Model condensation

The full FE linear model is splitted into active (ASETs) nodes and ommited nodes such that,
'''

st.latex(r"""
\begin{equation}
\left( \begin{bmatrix}
\bm{K}_{aa} & \bm{K}_{ao} \\ \bm{K}_{oa} & \bm{K}_{oo}
\end{bmatrix} - \omega^2\begin{bmatrix}
\bm{M}_{aa} & \bm{M}_{ao} \\ \bm{M}_{oa} & \bm{M}_{oo}
\end{bmatrix}
\right)
\begin{pmatrix}
\bm{\Phi}_a \\ \bm{\Phi}_o
\end{pmatrix} = 0
\end{equation}
"""
         )
'''
A linear dependency is assumed between the omitted and the active degrees of freedom,
'''

st.latex(r"""
\begin{equation}
\pmb{\Phi}_o =  \pmb{T}_{oa} \pmb{\Phi}_a
\end{equation}
"""
         )

'''
with $\pmb{T}_{oa}$ the transformation matrix between both sets. In general, the condensation is dependent on the frequencies and forms a nonlinear eigenvalue problem where each LNM,  with natural frequency, $\omega_j$, has one transformation matrix,
'''

st.latex(r"""
\begin{equation}
\pmb{T}_{oa}(\omega_j) = (\pmb{K}_{oo}-\omega^2_j \pmb{M}_{oo})^{-1}( \pmb{K}_{oa}- \omega_j^2 \pmb{M}_{oa}) \approx -(\pmb{K}_{oo}^{-1}+\omega^2_j\pmb{K}_{oo}^{-1}\pmb{M}_{oo}\pmb{K}_{oo}^{-1})(\pmb{K}_{oa}-\omega^2_j\pmb{M}_{oa})
\end{equation}
"""
         )
'''
This is the so-called exact-condensation matrix, where Kidder's mode expansion is also introduced. The first-order approximation of this equation is attained by letting $\omega_j =0$, thereby removing inertia effects. This results in a static condensation or Guyan reduction. Note that when the mass model consists only of lumped masses on the active degrees of freedom, $\pmb{M}_{oo} = \pmb{M}_{oa} = \pmb{0}$, Guyan reduction is the exact condensation.

After calculation of $\pmb{T}_{oa}$, the transformation from the active set and the full model is defined as $\pmb{T} =[\pmb{I}_a \; \pmb{T}_{oa}^\top]^\top$, with $\pmb{I}_a$ the identity matrix of dimension $a$. The condensed mass and stiffness matrices are obtained by equating the kinetic energy, $\mathcal{E}_k$ and the potential energy, $\mathcal{E}_p$ in the linear reduced and complete systems; if external loads are applied to the omitted nodes, equating virtual work gives the equivalent loads in the condensed model:

'''

st.latex(r"""
\begin{equation}
\begin{split}
\mathcal{E}_p &= \frac{1}{2}\bm{u}_n^\top\bm{K}\bm{u}_n \cong \frac{1}{2}\bm{u}_a^\top\bm{T}^\top\bm{K}\bm{T}\bm{u}_a = \frac{1}{2}\bm{u}_a^\top\bm{K}_a\bm{u}_a \\
\mathcal{E}_k &= \frac{1}{2}\dot{\bm{u}}_n^\top\bm{M}\dot{\bm{u}}_n \cong \frac{1}{2}\dot{\bm{u}}_a^\top\bm{T}^\top\bm{M}\bm{T}\dot{\bm{u}}_a = \frac{1}{2}\dot{\bm{u}}_a^\top\bm{M}_a\dot{\bm{u}}_a \\
\mathcal{W}_f &=\delta \bm{u}_n^\top \bm{F} \cong \delta \bm{u}_a^\top \bm{T}^\top \bm{F} = \delta \bm{u}_a^\top  \bm{F}_a 
\end{split}
\end{equation}
"""
         )

# so that condensed stiffness and mass matrix are obtained as $\pmb{K}_a = \pmb{T}^\top\pmb{K}\pmb{T}$, $\pmb{M}_a = \pmb{T}^\top\pmb{M}\pmb{T}$, and the external forces $\bm{F}_a = \bm{T}^\top \bm{F}$. The LNMs in the active set are then $\pmb{K}_a\pmb{\Phi}_{a}=\pmb{M}_a\pmb{\Lambda}_a\pmb{\Phi}_{a}$, with $\pmb{\Lambda}_a$ the diagonal matrix of squared natural frequencies.


st.divider()
sti.df_geometry(st.session_state.config.fem)
sti.sys_3Dconfiguration0(st.session_state.config)
sti.fe_matrices(st.session_state.config.fem)
