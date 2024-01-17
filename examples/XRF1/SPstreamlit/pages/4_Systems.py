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

stt.intrinsic_systems()

'''
- Free flying systems
- Multibody systems
- Stability systems
-  Control systems
'''
# st.latex(r"""
# \begin{bmatrix}
# \pmb{\hat{\mathcal{A}}}_1 -\pmb{L}_{\pmb{q}_1}(\pmb{q}_{1}^{\circ}) & \pmb{\hat{\Omega}}-\pmb{L}_{\pmb{q}_{12}}(\pmb{q}_2^\circ) & \pmb{\hat{\mathcal{A}}}_0 & \pmb{\mathcal{M}}^{-1}[\pmb{I}_{N_m}.._{N_p}..\pmb{I}_{N_m}] & \pmb{L}_{\pmb{q}_1\pmb{\zeta}}(\pmb{\zeta}^{\circ})
# \\
# -\pmb{\Omega} + \pmb{L}_{\pmb{q}_{21}}(\pmb{q}_2^\circ) & \pmb{L}_{\pmb{q}_{2}}(\pmb{q}_1^\circ) & \pmb{0} & \pmb{0} & \pmb{0}
# \\
# \pmb{I}_{N_m} & \pmb{0} & \pmb{0} & \pmb{0}& \pmb{0}
# \\
#  \pmb{\hat{\mathcal{A}}}_{P}& \pmb{0} & \pmb{0} &  -\frac{2U_\infty}{c}\pmb{\gamma}_p & \pmb{0}
# \\
# \pmb{L}_{\pmb{\zeta} \pmb{q}_1}(\pmb{\zeta}^{\circ}) & \pmb{0} & \pmb{0} & \pmb{0}& \pmb{L}_{\pmb{\zeta}}(\pmb{q}_{1}^{\circ})
# \end{bmatrix}
# \end{equation}
# """
# )
# st.image(wingsp_path, caption="XRF1 flutter")

sti.systems(st.session_state.sol, st.session_state.config)
