
## Reduction to Load paths:

The full FE linear model is splitted into active (ASETs) nodes and ommited nodes such that,

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
	
A linear dependency is assumed between the omitted and the active degrees of freedom,
\begin{equation}
\pmb{\Phi}_o =  \pmb{T}_{oa} \pmb{\Phi}_a
\end{equation}

with $\pmb{T}_{oa}$ the transformation matrix between both sets. In general, the condensation is dependent on the frequencies and forms a nonlinear eigenvalue problem where each LNM,  with natural frequency, $\omega_j$, has one transformation matrix,
\begin{equation}
\pmb{T}_{oa}(\omega_j) = (\pmb{K}_{oo}-\omega^2_j \pmb{M}_{oo})^{-1}( \pmb{K}_{oa}- \omega_j^2 \pmb{M}_{oa}) \approx -(\pmb{K}_{oo}^{-1}+\omega^2_j\pmb{K}_{oo}^{-1}\pmb{M}_{oo}\pmb{K}_{oo}^{-1})(\pmb{K}_{oa}-\omega^2_j\pmb{M}_{oa})
\end{equation}

This is the so-called exact-condensation matrix, where Kidder's mode expansion is also introduced. The first-order approximation of this equation is attained by letting $\omega_j =0$, thereby removing inertia effects. This results in a static condensation or Guyan reduction. Note that when the mass model consists only of lumped masses on the active degrees of freedom, $\pmb{M}_{oo} = \pmb{M}_{oa} = \pmb{0}$, Guyan reduction is the exact condensation.

After calculation of $\pmb{T}_{oa}$, the transformation from the active set and the full model is defined as $\pmb{T} =[\pmb{I}_a \; \pmb{T}_{oa}^T]^T$, with $\pmb{I}_a$ the identity matrix of dimension $a$. The condensed mass and stiffness matrices are obtained by equating the kinetic energy, $\mathcal{E}_k$ and the potential energy, $\mathcal{E}_p$ in the linear reduced and complete systems; if external loads are applied to the omitted nodes, equating virtual work gives the equivalent loads in the condensed model:

\begin{equation}
\begin{split}
\mathcal{E}_p &= \frac{1}{2}\bm{u}_n^\top\bm{K}\bm{u}_n \cong \frac{1}{2}\bm{u}_a^\top\bm{T}^\top\bm{K}\bm{T}\bm{u}_a = \frac{1}{2}\bm{u}_a^\top\bm{K}_a\bm{u}_a \\
\mathcal{E}_k &= \frac{1}{2}\dot{\bm{u}}_n^\top\bm{M}\dot{\bm{u}}_n \cong \frac{1}{2}\dot{\bm{u}}_a^\top\bm{T}^\top\bm{M}\bm{T}\dot{\bm{u}}_a = \frac{1}{2}\dot{\bm{u}}_a^\top\bm{M}_a\dot{\bm{u}}_a \\
\mathcal{W}_f &=\delta \bm{u}_n^\top \bm{F} \cong \delta \bm{u}_a^\top \bm{T}^\top \bm{F} = \delta \bm{u}_a^\top  \bm{F}_a 
\end{split}
\end{equation}

## Intrinsic modes:
Let $\pmb{\Phi}_{a}$ be the solution of the eigenvalue problem using the condensed matrices, $\pmb{M}_a$ and $\pmb{K}_a$. $\pmb{\Phi}_{a}$ includes the full set of modes in the condensed system written as displacement and linear rotations at the nodes along the load-paths. Those mode shapes also define velocity and strain distributions. Standard FE solvers yield results in the global reference frame while the intrinsic modes are defined in the initial local configuration (with the convention of the $x$-direction running along the local beam). Therefore, a matrix $\pmb{\Xi}_{0}(s) = [\pmb{R}^{ba}(s,0), \pmb{0} ; \pmb{0} , \pmb{R}^{ba}(s,0)]$ is introduced to rotate the 6-component vectors from the global to the local initial frame, $\pmb{R}^{ba}(s,0)$ calculated from the structural nodes position.

The discrete velocity mode is defined as $\pmb{\Phi}_{1j} = \pmb{\Phi}_{0j}$ and a linear interpolation is sought for the continuous displacement, $\pmb{\phi}_0(s)$, and velocities modes, $\pmb{\phi}_1(s)$:

\begin{equation}
\pmb{\phi}_{0j}(s) = \pmb{\phi}_{1j}(s) =  \pmb{\Xi}_{0}(s_i) \left( \pmb{\Phi}_{0j,i}\frac{s_{i+1}-s}{\Delta s_i} + \pmb{\Phi}_{0j,i+1}\frac{s-s_{i}}{\Delta s_i}\right)
\end{equation}

The corresponding distribution of linear and rotational momenta at the master nodes can be  obtained using the condensed inertia matrix, $\pmb{\Psi}_{1j}  = \pmb{M}_a \pmb{\Phi}_{1j} = \pmb{M}_a \pmb{\Phi}_{0j}$, expressed in their components in the global frame of reference. The introduction of this momentum mode allows the use of arbitrary mass models. Because the mass matrix is already calculated as an integral along the 3D domain and then condensed to a set of master nodes, the continuous momentum mode shapes, $\pmb{\psi}_1$, are considered lumped and defined using Dirac's delta function, $\delta$ as,

\begin{equation}
\pmb{\psi}_{1j}(s) =  \pmb{\Xi}_{0}(s_i) \pmb{\Psi}_{1j,i}\delta(s-s_i)
\end{equation}

Each displacement mode also generates a corresponding internal stress state. This defines discrete force/moment modes, $\pmb{\Phi}_{2}$, which are obtained from the displacement modes and the condensed stiffness matrix using a summation-of-forces approach
\begin{align}
\pmb{\Phi}_{2j,i+\frac{1}{2}}&= \begin{bmatrix}\mathcal{S}(\bm{\mathfrak{f}}_{(j)},s_i)\\  \mathcal{S} \left( \bm{\mathfrak{m}}_{\mathfrak{f}(j)} + (\bm{r}_i-\bm{r}_{i+\frac{1}{2}}) \times \bm{\mathfrak{f}}_{(j)},s_i \right)
\end{bmatrix} 
\end{align}
where $\pmb{r}_i$ is the position vector of the nodes summed by $\mathcal{S}$, and $\pmb{r}_{i+1/2}$ the mid position between nodes $s_i$ and $s_{i+1}$. The first term is the sum of forces due to modal displacements and the second one the sum of moments due to modal rotations and the cross product of the  position vector and the previous calculated force.
The strain modes $\pmb{\psi}_{2}$ are obtained from spatial derivatives of the displacement modes along along the load paths, and interpolated as piece-wise constant too,

\begin{align}
\pmb{\psi}_{2j}(s) = -\frac{\pmb{\phi}_{1j}(s_{i+1})-\pmb{\phi}_{1j}(s_{i})}{\Delta s_{i}}+ \pmb{E}^\top\frac{\pmb{\phi}_{1j}(s_{i+1})+\pmb{\phi}_{1j}(s_{i})}{2} 
\end{align}

## Nonlinear couplings:
After a Galerkin projection of the equations, the following tensors need to be approximated:
- Alphas must equal the identity matrix
\begin{align}
\alpha_{1}^{jl} & = \langle \pmb{\phi}_{1j}, \pmb{\psi}_{1l}\rangle = \delta^{jl} \\
\alpha_{2}^{jl} & = \langle \pmb{\phi}_{2j}, \pmb{\psi}_{2l}\rangle = \delta^{jl}
\end{align}

- Gammas give the nonlinear inertia and strain couplings

\begin{align}
\Gamma_{1}^{jkl} & = \langle \pmb{\phi}_{1j}, \mathcal{L}_1(\pmb{\phi}_{1k})\pmb{\psi}_{1l}\rangle,  \\
\Gamma_{2}^{jkl} & = \langle \pmb{\phi}_{1j}, \mathcal{L}_2(\pmb{\phi}_{2k})\pmb{\psi}_{2l}\rangle,
\end{align}

## Aeroelastic system:

Systems of equations
Different systems of equations are assembled depending on options: 

- Structural dynamic:
  \begin{equation}
		\begin{split}
		\dot{q}_{1j} &= \delta^{ji}\omega_{i}q_{2i}-\Gamma^{jik}_{1}q_{1i}q_{1k}-\Gamma^{jik}_{2}q_{2i}q_{2k}+ \eta_{j}  \\
		\dot{q}_{2j} &= -\delta^{ji}\omega_{i}q_{1i} + \Gamma_2^{ijk}q_{1i}q_{2k}
		\end{split}
  \end{equation}
- (Clamped) Aeroelastic systems:
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
