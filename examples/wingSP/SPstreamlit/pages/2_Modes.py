import fem4inas.plotools.streamlit.intrinsic as sti
import streamlit as st
import importlib
importlib.reload(sti)

st.set_page_config(
    page_title="Intrinsic modal shapes",
    page_icon="🛫",
    layout="wide"
)

'''
# Intrinsic modes

 Let $\pmb{\Phi}_{a}$ be the solution of the eigenvalue problem using the condensed matrices, $\pmb{M}_a$ and $\pmb{K}_a$. $\pmb{\Phi}_{a}$ includes the full set of modes in the condensed system written as displacement and linear rotations at the nodes along the load-paths. Those mode shapes also define velocity and strain distributions. Standard FE solvers yield results in the global reference frame while the intrinsic modes are defined in the initial local configuration (with the convention of the $x$-direction running along the local beam). Therefore, a matrix $\pmb{\Xi}_{0}(s) = [\pmb{R}^{ba}(s,0), \pmb{0} ; \pmb{0} , \pmb{R}^{ba}(s,0)]$ is introduced to rotate the 6-component vectors from the global to the local initial frame, $\pmb{R}^{ba}(s,0)$ calculated from the structural nodes position.

The discrete velocity mode is defined as $\pmb{\Phi}_{1j} = \pmb{\Phi}_{0j}$ and a linear interpolation is sought for the continuous displacement, $\pmb{\phi}_0(s)$, and velocities modes, $\pmb{\phi}_1(s)$:
'''


st.latex(r"""
\begin{equation}
\pmb{\phi}_{0j}(s) = \pmb{\phi}_{1j}(s) =  \pmb{\Xi}_{0}(s_i) \left( \pmb{\Phi}_{0j,i}\frac{s_{i+1}-s}{\Delta s_i} + \pmb{\Phi}_{0j,i+1}\frac{s-s_{i}}{\Delta s_i}\right)
\end{equation}
"""
         )

'''
The corresponding distribution of linear and rotational momenta at the master nodes can be  obtained using the condensed inertia matrix, $\pmb{\Psi}_{1j}  = \pmb{M}_a \pmb{\Phi}_{1j} = \pmb{M}_a \pmb{\Phi}_{0j}$, expressed in their components in the global frame of reference. The introduction of this momentum mode allows the use of arbitrary mass models. Because the mass matrix is already calculated as an integral along the 3D domain and then condensed to a set of master nodes, the continuous momentum mode shapes, $\pmb{\psi}_1$, are considered lumped and defined using Dirac's delta function, $\delta$ as,
'''

st.latex(r"""
\begin{equation}
\pmb{\psi}_{1j}(s) =  \pmb{\Xi}_{0}(s_i) \pmb{\Psi}_{1j,i}\delta(s-s_i)
\end{equation}

"""
         )


'''
Each displacement mode also generates a corresponding internal stress state. This defines discrete force/moment modes, $\pmb{\Phi}_{2}$, which are obtained from the displacement modes and the condensed stiffness matrix using a summation-of-forces approach
'''

st.latex(r"""
\begin{align}
\pmb{\Phi}_{2j,i+\frac{1}{2}}&= \begin{bmatrix}\mathcal{S}(\bm{\mathfrak{f}}_{(j)},s_i)\\  \mathcal{S} \left( \bm{\mathfrak{m}}_{\mathfrak{f}(j)} + (\bm{r}_i-\bm{r}_{i+\frac{1}{2}}) \times \bm{\mathfrak{f}}_{(j)},s_i \right)
\end{bmatrix} 
\end{align}
"""
         )


'''
where $\pmb{r}_i$ is the position vector of the nodes summed by $\mathcal{S}$, and $\pmb{r}_{i+1/2}$ the mid position between nodes $s_i$ and $s_{i+1}$. The first term is the sum of forces due to modal displacements and the second one the sum of moments due to modal rotations and the cross product of the  position vector and the previous calculated force.

The strain modes $\pmb{\psi}_{2}$ are obtained from spatial derivatives of the displacement modes along along the load paths, and interpolated as piece-wise constant too,

'''

st.latex(r"""
\begin{align}
\pmb{\psi}_{2j}(s) = -\frac{\pmb{\phi}_{1j}(s_{i+1})-\pmb{\phi}_{1j}(s_{i})}{\Delta s_{i}}+ \pmb{E}^\top\frac{\pmb{\phi}_{1j}(s_{i+1})+\pmb{\phi}_{1j}(s_{i})}{2} 
\end{align}
"""
         )

st.divider()
st.header('Intrinsic modal data')
st.link_button("Code","https://github.com/ACea15/FEM4INAS/blob/0958ca92b55073d799668136ae4d5132687f8969/fem4inas/intrinsic/modes.py#L192")

sti.df_modes(st.session_state.sol,
             st.session_state.config)
