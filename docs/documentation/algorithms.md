# Intrinsic modal solution

## Overall solution process

The tensor structure of the main components in the solution process is illustrated in the figure below in the sequential order they are computed, together with the asymptotic time and space complexities. The discretization of the problem comprises \(N_n\) number of condensed nodes, \(N_m\) modes used in the reduced order model and \(N_t\) time steps in the solution (if the problem is static, \(N_t\) represents a ramping load stepping scheme). The intrinsic modes, \(\Phi, \Psi  \in \mathbb{R}^{N_m \times 6 \times N_n}\) are computed from the eigenvalue solution and the coordinates \(\pmb{x}_a \in \mathbb{R}^{3 \times N_n}\) of the active nodes. 
The nonlinear couplings, \(\pmb{\Gamma} \in \mathbb{R}^{N_m \times N_m \times N_m}\) are calculated next, from which the is assembled and solved to yield the solution states \(\pmb q \in \mathbb{R}^{N_t \times N_s}\) with \(N_s\) the number of states in the system that is proportional to the number of modes. Local velocities, internal forces and strain fields  \(\pmb{x}_{1,2,3} \in \mathbb{R}^{N_t \times 6 \times N_n}\) are computed as a product of the corresponding intrinsic modes and states, and their integration leads to the position tensor, \(r_a\) with similar structure. In some cases, such as when gravity forces are included, the evolution of the rotational matrix, \(\pmb R_{ab}\), needs to be solved for too.

![Intrinsic modal solution](./img/tensors6.png)

For links to the codebase, see the following: 

- [Modes](api/modes.md).
- [Nonlinear couplings](api/couplings.md).
- Solution of equations:
    - Orchestrator to build the [Systems](api/systems.md)
    - Numerical solvers inside [Sollibs](api/sollibs.md): Using [Diffrax](https://docs.kidger.site/diffrax/) or bespoke solvers in JAX.
    - Right-hand-side (RHS) of the system of [equations](api/equations.md) implemented in pure functions to comply with JAX functional programming approach.

## System of equations

## Recovery of deformations

Analytical solutions to the  are obtained when the strain is assumed constant between nodes, using a piecewise constant integration. If a component in the load-path is discretized in $n$+1 points, strain and curvatures are defined in the mid-points of the spatial discretization (n in total). \(\gamma_n\) and \(\kappa_n\) are constant within the segment \(s_{n-1} \leq s \leq s_n\), and the position and rotation matrix after integration are
\begin{equation}
\begin{split}
\textbf{R}_{ab}(s) &= \textbf{R}_{ab}(s_{n-1})\pmb{\mathcal{H}}^0(\pmb\kappa,s) \\
\pmb{r}(s) &= \pmb{r}(s_{n-1}) + \pmb{R}_{ab}(s_{n-1})\pmb{\mathcal{H}}^1(\pmb\kappa, s)\left(\pmb{e}_1+\pmb{\gamma}_n\right) 
\end{split}
\end{equation}
with the operators \(\pmb{\mathcal{H}}^0(\pmb\kappa, s)\) and \(\pmb{\mathcal{H}}^1(\pmb\kappa, s)\) obtained from integration of the exponential function, as in \cite{PALACIOS2010}.
Note that when position and rotations are recovered from strain integration, there is still one point that is either clamped or needs to be tracked from integration of its local velocity.


![Algorithm](./img/algo_deformationsRecovery.png)
