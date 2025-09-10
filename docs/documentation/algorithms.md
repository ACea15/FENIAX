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

## System based solutions

An internal function identifier (Sol name) is build such that unique functions are mapped to the input settings in the system of equations, thereby avoiding the use of if-conditionals, 0-only additional vectors inside the solvers (running iteratively).
The options are as follows:

| Type        | Target | Gravity    | BC1 [prime=2] | ModalAero [prime=3] | SteadyAero [prime=5] | UnsteadyAero [prime=7] | Point loads [prime=11] | q0 approx | Rigid-body           | Nonlinearities          | residualised |
|-------------|--------|------------|---------------|---------------------|----------------------|------------------------|------------------------|-----------|----------------------|-------------------------|--------------|
| 1 static    | Level  | False: "g" | Clamped       | None                | None                 | None                   | None                   | via q2    | 1-quaternion+strains | All -\> ""              | None -\> ""  |
| 2 Dynamic   | TRIM1  | True: "G"  | Free          | Rogers              | qalpha               | gust                   | follower               | via q1    | All-quaternions      | Linear sys -\> "l"      | True -\> "r" |
| 3 Stability | TRIM2  |            | Prescribed    | Loewner             | qx (control)         | controls               | dead                   |           |                      | Linear sys+disp -\> "L" |              |
| 4 Control   |        |            |               |                     |                      |                        |                        |           |                      | only gamma1 -\> "g1"    |              |

And some of the implemented RHS solutions (see for instance [equations](api/equations.md)):

| Sol name |                                                 | label               | Imp |
|----------|-------------------------------------------------|---------------------|-----|
| 10G1     | Structural static under Gravity                 | [1,0,G]             | Y   |
| 10g11    | Structural static with follower point forces    | [1,0,g,0,0,0,0,1]   | Y   |
| 10g121   | Structural static with dead point forces        | [1,0,g,0,0,0,0,2]   | Y   |
| 10g1331  | Structural static with follower+dead forces     | [1,0,g,0,0,0,0,3]   | N   |
| 10g15    | Manoeuvre under qalpha                          | [1,0,g,0,1,1]       | Y   |
| 10G15    | Manoeuvre under qalpha and Gravity              | [1,0,G,0,1,1]       | N   |
| 10g75    | Manoeuvre under qalpha and controls             | [1,0,g,0,1,2]       | N   |
| 10G75    | Manoeuvre under qalpha+controls+Gravity         | [1,0,G,0,1,2]       | N   |
| 20g1     | Clamped Structural dynamics, free vibrations    | [2,0,g]             | Y   |
| 20G2     | Free Structural dynamic with gravity forces     | [2,0,G,1]           | Y   |
| 20g2     | Free Structural dynamic                         | [2,0,g,1]           | Y   |
| 20g11    | Structural dynamic follower point forces        | [2,0,g,0,0,0,0,1]   | Y   |
| 20g121   | Structural dynamic dead point forces            | [2,0,g,0,0,0,0,2]   | Y   |
| 20g22    | Free Structural dynamic follower point forces   | [2,0,g,1,0,0,0,1]   | Y   |
| 20g242   | Free Structural dynamic dead point forces       | [2,0,g,1,0,0,0,2]   | Y   |
| 11G6     | Static trimmed State (elevator-qalpha,          | [1,1,G,1,1]         | Y   |
|          | no gravity updating)                            |                     | Y   |
| 12G2     | Static trimmed State (elevator-qalpha,          | [1,2,G,1]           | Y   |
|          | gravity updating)                               |                     |     |
| 21G150   | Dynamic trimmed State                           | [2,1,G,1,1,2]       | N   |
| 20g21    | Gust response                                   | [2,0,g,0,1,0,1]     | Y   |
| 20g273   | Gust response, q0 obtained via integrator q1    | [2,0,g,0,1,0,1,0,1] | Y   |
| 20g105   | Gust response with steady qalpha                | [2,0,g,0,1,1,1]     | N   |
| 20g42    | Gust response Free-flight                       | [2,0,g,1,1,0,1]     | Y   |
| 20G42    | Gust response Free-flight and gravity (X error) | [2,0,G,1,1,0,1]     | N   |
| 20G1050  | Gust response Free-flight, gravity, controls    | [2,0,G,1,1,2,1]     | N   |
|          |                                                 |                     |     |

Sol name is based on the input options obtained as the multiplication of prime numbers, which can be obtained and inspected as:

``` python
import feniax.intrinsic.functions as functions
label = functions.label_generator([2,0,'g',0,1,0,1,0,1])
print(label)
```

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
