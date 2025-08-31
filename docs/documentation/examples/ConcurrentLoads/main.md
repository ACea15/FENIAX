
<div class="abstract" id="orgb69ed49">
<p>
This paper demonstrates a new solution for large scale simulations of aircraft-sizing loads. A geometrically nonlinear aeroelastic description has been extended with concurrent capabilities to tackle efficiently the very demanding loading scenarios in the design process. The software can be deployed on modern hardware architectures and has been benchmarked on CPUs and GPUs. While keeping computational times in the order of seconds, the presented approach enables the construction of load envelopes corresponding to static and dynamic cases, seamless integration with full-aircraft models, and uncertainty quantification analysis via massively-parallel Montecarlo simulations, among others. Furthermore, differentiation of both concurrent computations and aeroelastic loads is achieved via JAX library automatic differentiation engine and collective primitives. 
We explore a high-aspect-ratio-wing aircraft configuration, showing the ability of the solvers to capture nonlinear effects in a wide range of scenarios: discrete, manoeuvre and gust loads are assessed for multiple flow and parametric conditions running concurrently. Sensitivities of maximum wing-root stresses and tip displacements are computed and verified against finite differences.  
</p>

</div>


# Introduction

Aeroelastic analysis is expected to become increasingly important from the very early phases of the wing design process: while the focus was historically on the post-design stages to ensure in-flight integrity, it now becomes paramount to capture the cross-couplings between disciplines for optimal performance.
This is specially important for high-aspect-ratio wings (HARWs) <cite:&PATIL2004;&MODARESS-AVAL2019> that improve aerodynamic efficiency but leads to an increase in flexibility, which brings new challenges both in terms of design and modelling capabilities. As highlighted in <cite:&LIVNE2018>, formulations that include nonlinear effects should be developed that not only enhance current modelling techniques  but that also allow rapid data turnaround for the industry. Real-time, hardware-in-the-loop flight simulators would also benefit of actively controlled, deformable airplane models. This leads to a more nonlinear landscape, where the overall aerodynamic performance needs to be calculated around a flight shape with large deformations <cite:&JONSSON2023>.
A more holistic approach to the design also increases the complexity of the processes exponentially, and the trade-offs and cost-benefit analysis may not be possible until robust computational tools are in-place to simulate the different assumptions.
Certification of new air vehicles is another important aspect that requires 100,000s of load cases simulations <cite:&KIER2017>, as it considers manoeuvres and gust loads at different velocities and altitudes, and for a range of mass cases and configurations. This poses another challenge for new methods that aim to include new physics since they normally incur in prohibitively expensive computational times. 
Lastly, the mathematical representation of the airframe, embodied in the complex Finite-Element Models (FEMs) built by organizations, encompasses a level of knowledge that is to be preserved when including the new physics mentioned above <cite:&KENNEDY2014a>. 
Leveraging on the numerical library JAX <cite:&BRADBURY2018>, these considerations set the goals for the baseline work in <cite:&CEA2025>: 1) to be able to perform geometrically nonlinear aeroelastic analysis, 2) to work with existing generic FEMs in a non-intrusive manner, and 3) to achieve a computational efficiency that is equivalent to present linear methods (if not faster), 4) compute derivatives of the aeroelastic response via Algorithm Differentiation (AD). Those were extended in to include rigid-body dynamics, trimmed flight, and architectural benchmarks of GPU versus CPU <cite:&CEA2025b>.
In this work we explore the latest advances on accelerator's parallelisation, and how to integrate them into our solution process to enable intensive aeroelastic simulations under geometrically nonlinear assumptions.
Specifically, we set out to characterise the dynamics of highly flexible aircraft in response to the large envelope of simulations required to capture in-flight loads encountered in the certification process &#x2013;while introducing new physics that account for the large displacements and rotations ultra-high-aspect-ratio wings are expected to undergo.
For this, a Single Program Multiple Data (SPMD) paradigm is employed with the main computation spanning as many devices as available in the cluster and performing collective operations to communicate between devices.
By addressing in one program a substantial part of scenarios during flight (manoeuvres and gust responses at different velocities and altitudes, and for a range of mass cases and configurations), we are able to produce the critical loading characteristics of the aircraft in very short simulation times. Moreover, we can differentiate the boundaries of the critical cases using the already demonstrated capabilities of AD within JAX <cite:&CEA2025>, but now extending them to produce derivatives across those design envelopes. This implies computing gradients across concurrent simulations and collective operations, which we show it is well managed by the chosen library, that is actively developed to solved similar problems in the realm of machine learning.   
This is expected to be highly applicable in providing designers with additional insights about sensitivities and in extending gradient-based optimization analysis <cite:&BROOKS2019> with load-sizing constraints. In fact, having shown our formulation works well with high fidelity structural models of industrial complexity <cite:&CEA2023>, a further step is to increase the accuracy of the currently used Doublet-Lattice-Method (DLM) solution.
By coupling with differentiable Computational-Fluid-Dynamics (CFD) solvers such as those of  <cite:&MUELLER2018>,  <cite:&HE2020>,  <cite:&ECONOMON2016a>, a fully-differentiated high-fidelity aeroelastic solver could be achieved. This is a very feasible option within our workflow where the matrices of Generalised Aerodynamic Forces (GAFs) from a DLM solution are replaced by CFD GAFs, which are a standard part of the process in flutter analysis <cite:&SILVA2004;&SIMIRIOTIS2023>. 
Another area we begin to explore herein is the development of tools for optimisation under uncertainty. While various methods have been envisioned to introduce Uncertainty Quantification (UQ) in simulations <cite:&SCHAEFER2020>, Montecarlo methods remain a compelling option for generic problems. They can suffer from slow convergence, however they cater well for embarrassingly parallel algorithms as each path in the simulation is independent of the others. The accelerated simulations may open new ways for UQ of aeroelastic problems without resorting to surrogates <cite:&FELDSTEIN2020>.
Therefore the three major contributions of this work are summarized are as follows:

-   Large-scale parallelisation of static and dynamic aeroelastic cases to produce the load envelopes used in aircraft-sizing design loops.
-   Computation of gradients via AD across the critical cases previously calculated concurrently.
-   Differentiation of any moment of a distribution (expectations of the powers of a random variable) produced as the output of an aeroelastic quantity of interest to an stochastic input.

It is also worth remarking the whole suite of capabilities outlined are entirely physics-based, thereby not relying on metamodels of any kind, but achieving similar computational performance during evaluation thanks to a physical condensation of the model and a very efficient implementation. 
The paper is organised as follows: Sec. [2](#org611af80) gives and overview of the theoretical and computational developments that underpin this work with a focus on the new parallelisation capabilities, computation of derivatives across concurrent simulations, and a new pattern to build sensitivities of large load envelopes. In sec. [3](#org95182e6), a representative configuration of an ultra-high-aspect-ratio aircraft is studied under various loading scenarios that have been parallelised; namely structural static loads, manoeuvre cases for varying flow conditions and dynamic loads with multiple gusts running concurrently. To our knowledge, this application of modern hardware architectures to aircraft nonlinear load analysis has not been explored so far and has a huge potential to be introduced inside current industrial processes. We conclude in Sec. [4](#org0893bee) with a summary of the main advances and potential future work to consolidate the methods into a versatile, production-ready tool.


# Theoretical and computational background

The main aspects of the aeroelastic framework we have developed are presented in this section. 
The approach is built on a non-intrusive reduction order process combined with a nonlinear description of the dominant dimension for slender structures. It achieves a nonlinear representation of aeroelastic models of arbitrary complexity in a very efficient manner and without losing the characteristics of the linear model. We target the calculation of flight loads in this work, but it can also be applied to the computation of aeroelastic stability phenomena such as flutter or divergence <cite:&CEA2023>.The key features of the formulation have been presented in previous work:

-   Geometrically nonlinear aeroelastic analysis using complex GFEMs: achieved via a three step process in which a condensed model is first produced, the dynamics of this reduced model are described by a system on nonlinear equations <cite:&HODGES2003> written in material velocities and stresses, and a modal expansion of those variables is the final key step in seamlessly mapping the global FEM into the nonlinear description <cite:&PALACIOS2011>. The overall process can be found in <cite:&CEA2021b>.
-   Maximum performance: as a combination of the numerical library JAX with its JIT compiler and accelerator capabilities  driving the calculations, a highly optimised and vectorized codebase with the main algorithms described in <cite:&CEA2025b>, and the newly  added parallelisation capabilities.
-   Differentiation and sensitivity analysis: using JAX algorithmic differentiation toolbox, the entire process, from inputs to aeroelastic outputs can be differentiated as shown in <cite:&CEA2024a>.

In addition to this, we show below how to leverage on modern hardware architectures and a parallelisation across devices to a) build load envelopes of static and dynamic aeroelastic simulations in seconds; b) differentiate across the concurrent simulations to obtain sensitivities of dynamic loads; c) quantify the uncertainties in the nonlinear response given a non-deterministic loading field and obtain derivatives of the expectations.


## Nonlinear aeroelastic system

Given a general GFEM, a reduced model is obtained from a static or dynamic condensation into nodes along load paths that captures well the stiffness and inertia properties in the condensed matrices, $\pmb{K}_a$ and $\pmb{M}_a$. The eigenvalue solution of the FEM yields the modal shapes, $\pmb \Phi_0$, and frequencies $\pmb \omega$. A projection of the state variables, velocities $\pmb{x}_1 = \pmb{\Phi}_1\pmb{q}_1$ and stresses $\pmb{x}_2 = \pmb{\Phi}_2\pmb{q}_2$, and a Galerkin projection of the equations of motion leads to the system of ODEs that is solved in time domain. 
Aerodynamic forces are obtained via Generalised Aerodynamic Forces (GAFs) using a panel-based DLM solver and Roger's rational function approximation <cite:&ROGER1977> to bring the forces to the time domain, resulting in a modal force component given as:

\begin{equation}\label{eq3:eta_full}
\begin{split}
\bm{\eta}_a = Q_\infty & \left(\vphantom{\sum_{p=1}^{N_p}} \pmb{\mathcal{A}}_0\bm{q}_0 +b\pmb{\mathcal{A}}_1 \bm{q}_1 +b^2 \pmb{\mathcal{A}}_2\dot{\bm{q}}_1    + \pmb{\mathcal{A}}_{g0}\bm{v}_g +b\pmb{\mathcal{A}}_{g1} \dot{\bm{v}}_g +b^2 \pmb{\mathcal{A}}_{g2}\ddot{\bm{v}}_g +  \sum_{p=1}^{N_p} \pmb{\lambda}_p  \right) 
\end{split}
\end{equation}

where the $\pmb{\mathcal{A}}_is$ are real matrices, $b=\frac{c}{2U_\infty}$ with $c$ the reference chord, $Q_\infty = \tfrac12\rho_\infty U_\infty^2$ the dynamic pressure, $\pmb{\lambda}_p$ the aerodynamic states and $N_p$ the number of lags. Note these forces naturally follow the structure since the formulation is written in the material frame of reference. 
The coupling of the structure and aerodynamic equations combined with the aerodynamic lags, gravity forces, $\bm{\eta}_g$, and gust disturbances, $\bm{v}_g$, gives the final ODE system in compact form: 

\begin{equation}
\label{eq2:sol_qs}
\begin{split}
\dot{\pmb{q}}_{1} &=  \hat{\pmb{\Omega}}  \pmb{q}_{2} - \hat{\pmb{\Gamma}}_{1} \pmb{:} \left(\pmb{q}_{1} \otimes \pmb{q}_{1} \right) - \hat{\pmb{\Gamma}}_{2} \pmb{:} \left( \pmb{q}_{2} \otimes  \pmb{q}_{2} \right) + \hat{\bm{\eta}}  \\
\dot{\pmb{q}}_{2} &= -\pmb{\omega} \odot \pmb{q}_{1} + \pmb{\Gamma}_{2}^{\top} \pmb{:} \left( \pmb{q}_{2} \otimes  \pmb{q}_{1} \right) \\
\dot{\bm{\lambda}}_{p} &= Q_{\infty}\bm{\mathcal{A}}_{p+2}\pmb{q}_{1}
                       + Q_{\infty}\bm{\mathcal{A}}_{p+2}\dot{\pmb{v}}_g
                       -\frac{\gamma_p}{b}\bm{\lambda}_{p}
\end{split}
\end{equation}

where $\odot$ is the  Hadamard product (element-wise multiplication), $\otimes$ is the tensor product operation and $\pmb{:}$ is the double dot product.
In this system the aerodynamic added-mass effect has been moved to the left hand side such that $\bm{\mathrm{A}}_2 = (\pmb{I} - \frac{\rho c^2}{8}\pmb{\mathcal{A}}_2)^{-1}$, and it couples all DoF in $\pmb q_1$. Thus the natural frequency terms become $\hat{\pmb{\Omega}} = \bm{\mathrm{A}}_2 \textup{diag}(\pmb{\omega})$ and the nonlinear terms $\hat{\pmb{\Gamma}} = \bm{\mathrm{A}}_2 \bm{\Gamma}$. The effect of all external forces, aero, $\bm{\eta}_a$, gravity, $\bm{\eta}_g$, and others, $\bm{\eta}_f$, are combined in such that $\hat{\bm{\eta}} = \bm{\mathrm{A}}_2 \left( \left( \bm{\eta}_a - \frac{\rho c^2}{8} \pmb{\mathcal{A}}_2\dot{\bm{q}}_1 \right) +  \bm{\eta}_g + \bm{\eta}_f \right)$.
The aerodynamic matrices $\hat{\bm{\mathcal{A}}}_{p+2}$ have also been scaled accordingly.
 The nonlinearities in the system are encapsulated in the modal couplings of the third-order tensors $\pmb{\Gamma}_1$ and $\pmb{\Gamma}_2$  (the former introduces the gyroscopic terms in the dynamics and the latter introduces the strain-force nonlinear relation).
Once the nonlinear solution of the condensed model is computed, the corresponding full 3D state can be calculated as a postprocessing step if necessary <cite:&CEA2025>.
Note unsteady aero is linear. Here it is obtained about undeformed geometry since we use the DLM, but it could be updated to the trim shape if needed (particularly to CFD aerodynamics). 


## High performance implementation

The formulation described above has been made into the codebase FENIAX (Finite Element models for Nonlinear Intrinsic Aeroelastics in JAX) <sup><a id="fnr.1" class="footref" href="#fn.1" role="doc-backlink">1</a></sup>. It has been thoroughly tested with currently 12 different models that amount to over 200 tests that run in minutes and are part of Continuous-Integration/Development (CI/CD) workflow. Moreover, a flexible software architecture allows for automatic analysis of generic models from standard input files, which can integrated with other computational tools.
The Python library JAX has been used as the numerical engine for calculations and it also manages the parallelisation, therefore some details on the library are worth describing.
JAX is designed for high-performance numerical computing with focus on machine learning activities <cite:&BRADBURY2018>. It relies on XLA (Accelerated Linear Algebra), a domain-specific compiler for linear algebra that optimizes computations for both CPUs and GPUs. In fact XLA is platform-agnostic and achieves optimised performance on the target architecture orchestrating a complex process that encompassing a series of optimizations and transformations: the source code is first converted into HLO (High-Level Optimizer) code, an specialized language derived from a graph representation of the computations; XLA performs optimisations on the HLO code (geared towards high-level mathematical operations, particularly those in linear algebra and machine learning models), and are independent of the hardware architecture, such as operation fusion. It then carries optimisations for the particular architecture in use. From there the LLVM toolkit is leveraged to produce and Intermediate Representation (IR) that the LLVM compiler can understand, perform further optimisations and finally output the machine code. 
When it comes to leveraging the computational power of (NVIDIA) GPUs, the link between XLA and CUDA kernels is critical. On the one hand, JAX utilises CUDA libraries such as cuBLAS for dense linear algebra; on the other hand, it is capable of generating custom CUDA kernels for operations that are not efficiently covered by standard libraries. 
In order to transform the high level Python to low level optimised code, the source code has to comply with various constraints and feature functional programming characteristics.
With regards to the parallelisation, JAX follows a Single-Program Multi-Data (SPMD) parallelism, whereby a single program operates on multiple data sets in parallel. This means the same computation graph is compiled and executed across different devices. Inter-device communication and synchronization are managed internally by the library.
Implementation wise, the \texttt{pmap} function maps a function across multiple input sets, distributing the workload across available GPUs. Thus being the parallel equivalent to the \texttt{vmap} function.
The new standard for parallelisation is based on data sharding, either done automatically using the \texttt{shard\_map} function or by sharding the data and passing it to a \texttt{jitted} function specifying input and output shape of the data to be partitioned. Inside the function, the compiler determines the necessary partitions of the data, synchronization, and communication. Collective operations like broadcasts and reductions are available within the \texttt{jax.lax} module. 
Internally JAX uses NVIDIA Collective Communications Library (NCCL) for low level communication across devices.  
The overall solution process and a description of the parallelisation strategy follow next. 


### Overall solution process

Algorithm [3](#org5f9fbc5) shows the main components in the solution, highlighting the time and space complexities, $O(time, space)$, of the data generated along the process. One single program is being run, for instance a dynamic simulation computing the response to multiple gusts that will be run in parallel for a total number of $N_c$ cases. $N_t$ time-steps are used in the integration scheme with a resolution of $N_m$ modal shapes. The FE model has been condensed to $N_n$ number of nodes. 
The intrinsic modes, $\bm{\phi}$, $\bm{\psi}$, are computed from the condensed FE nodal positions and matrices; subsequently, the nonlinear terms, $\bm \Gamma$, are obtained as the integral along the reduced domain of the modal couplings; the nonlinear system of equations is built and time-marched in time to yield the solution in modal coordinates, $\bm q$;
the intrinsic variables of the solution (velocities, $\bm{X}_1$,  internal forces, $\bm{X}_{2}$ and strains, $\bm{X}_{3}$) are recovered from the modal coordinates and the intrinsic modes; finally the positional and rotational field, $\bm{r}_a$, $\bm{R}_{a}$, of the reduced model are computed via integration of the strain field. 

\begin{algorithm}[h!]
\label{org5f9fbc5}
\DontPrintSemicolon
\SetKwInOut{Input}{input}
\SetKwInOut{Output}{output}
\Input{Input file: settings.yaml; FE model: $\bm{K}_a$, $\bm{M}_a$, $\bm{X}_a$; Aerodynamic matrices: $\bm{\mathcal{A}}$}
\Output{Nonlinear aeroealastic solutioxn}
\Begin{
 \BlankLine
$\bm{\phi}$, $\bm{\psi}$  $\longleftarrow$ modes($\bm{K}_a$, $\bm{M}_a$, $\bm{X}_a$) \Comment{Intrinsic modes: O($N_n^2 \times N_m$; $N_n \times N_m$)}  \;
$\bm{\Gamma}$  $\longleftarrow$ couplings($\bm{\phi}$, $\bm{\psi}$) \Comment{Nonlinear couplings O($N_n \times N_m^3$; $N_m^3$)} \;
$\bm{q}$  $\longleftarrow$ system($\bm{\Gamma}$, $\bm{\mathcal{A}}$, $\bm{\phi}$, $\bm{X}_a$) \Comment{Modal coordinates: O($\frac{N_c}{N_d} \times N_t \times N_m^3$; $N_c \times N_t \times N_m$)}  \;
$\bm{X}_1$, $\bm{X}_{2}$, $\bm{X}_{3}$   $\longleftarrow$ ivars($\bm{q}$, $\bm{\phi}$, $\bm{\psi}$) \Comment{velocity/strain fields: O($\frac{N_c}{N_d} \times N_t \times N_n \times N_m$; $N_c \times N_t \times N_n$)} \;
$\bm{r}_a$, $\bm{R}_{a}$   $\longleftarrow$ integration($\bm{X}_{3}$, $\bm{X}_a$) \Comment{Positional/rotational fields: O($\frac{N_c}{N_d} \times N_t \times N_n \times N_m$; $N_c \times N_t \times N_n$)}  \;
\BlankLine
}
\caption{Main components in solution process}
\end{algorithm}


### Two-level parallelisation

Various parallelism models have been developed in the context of deep learning, for which JAX has been particularly designed, and we try to adapt here those methods to  a large system of nonlinear equations solved in parallel for multiple external forces, i.e. right hand side of the equations. Data Parallel (DP) strategies consist of splitting a large batching into chunks, each fed to a single device, hence scaling to very large data batches. In Large Language Models (LLMs), the number of parameters can exceed that of input data, and often do not fit in a single device. In this case a tensor parallelism (TP) strategy is employed by which the tensor of weights that are to be optimised is sharded with synchronisation at the end of each step. Hybrid strategies that mix the two are employed in production. In the engineering applications of interest here, the number of designs variables would usually be between the tens to the few thousands, so tensor parallelism becomes less relevant. However, the number of simulations for different inputs, and the size of each one of them, can be very large. 
Therefore we opt for a DP strategy in which our batch of data becomes the multiple inputs that are used to build the external forces for which we want to compute the response.
The strategy implemented first splits the input data along the leading axis according to the total number of devices available using a data sharding approach. Each device receives the subset of inputs, a closure function that is "jitted" is called with the respective inputs, and inside the closure the high level function that computes the response (solution of the static response or time marching of the dynamic equations) is "vmapped" with respect to the subset of inputs. This last \texttt{vmap} makes the inputs that go into each device, or CPU cores, to run in parallel. Note the parallelisation happens at the final solution of the system of equations, meaning previous steps such as computation of intrinsic modes or nonlinear couplings is only carried out once before the concurrent simulations.
Algorithm [4](#org1db1514) illustrates this process with pseudo code.
The process by which inputs are split and sent to each device is presented in Fig. [18](#org77ab921), which shows the two-level parallelisation.

![img](figs_ext/parallelGPU.pdf "Input distribution example for multi-GPU runs")

The inputs to the parallel simulation are tensors of arbitrary shape with the only condition that the first axis being the one over which to run the parallelisation. For the manoeuvre and gust cases below, for instance, the tensor of inputs is a matrix with the second axis being a vector with the combination of flow conditions and gust parameters.   
In the figure we can see each GPU has a global memory and L2 cache, and in addition cores in the GPU are packed into the so-called streaming processors, each with its own registers and L1 caches. The strength of these chips is in the large number of cores, in the thousands, that can run in parallel, thus after the inputs are initially divided, many computations can run in parallel even within each GPU.

\begin{algorithm}[h!]
\label{org1db1514}
\DontPrintSemicolon
\Begin{
 \BlankLine
%
\SetKwFunction{Fy}{y\_aeroelastic}
\SetKwFunction{Fyy}{y}
 \SetKwProg{Fn}{Function}{:}{}
  \Fn{\Fy {\texttt{inputs}}}{
   \Fn{\Fyy {\texttt{input}}}{
   ... \;
(nonlinear aeroelastic computation)
\BlankLine
\KwRet \texttt{q, X1, X2, X3, ra, Rab} \;
}
  \texttt{
  yvmap = jax.vmap(y) \;
  q\_multi, X1\_multi, X2\_multi, X3\_multi, ra\_multi, Rab\_multi $\longleftarrow$ yvmap(inputs) \;
  \KwRet dict(q=q\_multi, X1=X1\_multi, X2=X2\_multi, X3=X3\_multi, ra=ra\_multi, Rab=Rab\_multi) \;
  }}

\texttt{
num\_devices $\longleftarrow$ jax.device\_count() \;
mesh $\longleftarrow$  jax.sharding.Mesh( \;
devices=jax.experimental.mesh\_utils.create\_device\_mesh( \;
(num\_devices,)), axis\_names=('x')) \;
inputs = jax.device\_put(inputs, jax.sharding.NamedSharding(mesh, \;  jax.sharding.PartitionSpec('x'))) \;
y\_aeroelastic $\longleftarrow$ jax.jit(y\_aeroelastic) \;
sol $\longleftarrow$ y\_aeroelastic(inputs)
}
    }
\caption{Parallelisation multiple load cases}
\end{algorithm}

Algorithm [4](#org1db1514) has been also implemented using a \texttt{pmap} function with similar results but slightly different design.


## Gradients of concurrent load-cases

Differentiable Programming encompasses AD, adjoint solutions and various concepts to produce full-fledged programs where derivatives can be taken through complex control flow, loops, and even other libraries embedded in the program such as MPI. It is the backbone of both gradient-based optimisation and Machine Learning frameworks, hence a very active field of research. New libraries for AD are either based on source-code transformation <cite:&MOSES2021> or tape-based methods <cite:&SAGEBAUM2019>.
JAX tape-based approach showcases unique features to do AD in heterogeneous devices and across concurrent operations that are explored herein with two important applications: obtaining derivatives of statistical moments of a distribution generated as the output of a Montecarlo simulation, and a strategy to differentiate the boundaries of load envelopes representing critical aircraft loads in the certification process.
The JAX-based numerical library Diffrax <cite:&KIDGER2021> is used as it ships with robust Newton-Raphson and ODE solvers with various adjoints options. It has been found that when computing derivatives of parallel simulations, the solvers in this library are not compatible with the new sharding features in JAX, however, it is able to differentiate across concurrent simulations if those we produced via a \texttt{pmap} strategy. This means that \texttt{pmap} was used for the parallelisation in the differentiation of Expectations in Sec [3.2](#orgcbb00b3).
Most likely both strategies will be supported for differentiation in the future, therefore giving more flexibility and aligning the AD to the various parallelisation approaches.


### Montecarlo analysis for UQ

Montecarlo analysis utilises random sampling to evaluate the effect of input uncertainties on model outputs.
It allows here to propagate uncertainties in the nonlinear aerolelastic formulation outlined in Sec. [2.1](#org9fcebb9). Since the samples are independent, a very simple parallelisation is possible on this algorithm: for each sample load case (right-hand side of the equations), launch a simulation, then collect outputs and perform a collective operation such as the mean across devices to calculate expectations. We can represent the system of equations in Eq. \eqref{eq3:eta_full} as follows,

\begin{equation}\label{random_system}
\dot{\bm{q}} = \bm{Q}(\bm{q}, \bm{\alpha}_1) + \bm{L}(\bm{X}, \bm{\alpha}_2)
\end{equation}

where $\alpha_1$ and $\alpha_2$ are the input parameters we want to obtain the gradients with respect to, the former for internal states in $\bm{Q}$, the latter for external loads in $\bm{L}$. $\bm{X}$ is the random variable or vector of variables to sample from. In Sec. [3.2](#orgcbb00b3) this will be demonstrated for a nonlinear static equilibrium with $\bm{\alpha}_1$ representing the FE matrices and corresponding eigenvectors, and $\bm{\alpha}_2$ a single parameter controlling the amount of external loading in the structure.
After marching the system in time or solving the nonlinear equations for a static problem, the random variable $\bm{Y}$ emerges as some (unknown) function $\bm{F}$ of the final state $\bm{q}(t_f)$:

\begin{equation}\label{random_system}
\bm{Y} = \bm{F}(\bm{q}(t_f), \bm{X}, \bm{\alpha}_1, \bm{\alpha}_2)
\end{equation}

The objective is to calculate derivatives of arbitrary moments $n$ of $\bm{Y}$, $ \frac{\partial \mathbb{E}[Y^n]}{\partial \bm{\alpha}}$. They can be calculated via Montecarlo by launching the concurrent simulations for each $\bm{X}_i$ and performing a collective mean on the output of interest $\bm{Y}_i$ after solving the aeroelastic system. Since all the operations in the process are being tracked in JAX, derivatives with respect to $\bm{\alpha}$ can be recovered. The slow convergence is one of the key limitations of plain Monte Carlo methods as the error goes with number of samples $ N $: $\text{Error} \sim O \left( \frac{1}{\sqrt{N}} \right) $.
Quasi Montecarlo methods using low discrepancy sequences such as Sobol numbers can be used for better convergence and multilevel methods to reduce computational cost <cite:&GILES2008>.
While the method itself is not new, the combination of modern-hardware architectures and concurrent subroutines that are AD-capable can unlock optimisation problems in Engineering written in terms of expectations instead of deterministic quantities.


### Differentiable-parallel dynamic loads

Once a parallel system is in place to compute hundreds of load cases, the next step is to obtain the derivatives of the critical loads coming from the parallel analysis. Since those are calculated using AD, all the operations need to be available in memory. While the Montecarlo analysis of the previous section made use of all the cases to compute the output expectation, in this case we are only interested on those that induce the largest loads.
We encountered two major issues: the memory required for the gust cases was already in the limit of a single device (over 60 GB of RAM), to which the AD normally duplicates the requirement. As the software can now be run on multiple devices, each with its own memory, this is not a completely restrictive factor. The second issue was simply a lack of implementation of the needed collective operations in JAX. This is the case for instance with the maximum function, that can usually be differentiated but not if collective operations are involved (it is also not practical as most of the data generated by such a maximum are zeros not needed anyway). The solution found has been named the Forager Pattern and is depicted in Fig. [26](#org967af6c). The code launches many simulations concurrently with the predefined load-cases. The solutions of all these simulations are collected (hundreds of cases, hundreds of nodes, thousands of time steps make for a single field of interest like the stress to have a size of the order of $10^7-10^8$). A filtering step consists of a selection of monitoring points of interest (nodes in the FEM), and then a double reduction operation in both time and load cases, for example the maximum of the selected field in time and across cases; the output is a selection of the most problematic load cases according to the predefined metric in the input file. For these critical points the program builds the inputs for the cases previously run in parallel but now with AD and on a much smaller basis, and finally more FENIAX process are spawn for the AD computations. In this way we have created a meta-program that can automatically create programs based on the results &#x2013;with limitations on the implemented possibilities.

![img](figs_ext/forager.pdf "Forager pattern for differentiable-parallel simulations")


# Results

In this section we show the main strengths of our solvers. We run a representative aircraft model undergoing very large nonlinear displacements, the University of Bristol Ultra-Green (BUG) aircraft model <cite:&STODIECK2018> is the chosen platform as it is not based on proprietary data and it showcases high-aspect ratio wings and a GFEM based on beam and shell elements in MSC Nastran. The main components of the aeroelastic model have been presented in <cite:&CEA2025a>.
Leveraging on modern hardware architectures and a parallelisation across devices, to unlock problems such as quantifying the uncertainties in the nonlinear response given a non-deterministic loading field.
We build load envelopes of static and dynamic aeroelastic simulations and differentiate across the concurrent simulations to obtain sensitivities of dynamic loads as well as moment statistics.
Structural and aeroelastic static simulations follow, solved via a Newton-Raphson solver with tolerance of $10^{-6}$, as well as an assessment of the aircraft dynamics in response to various gust profiles.
A high modal resolution of 100 modes is employed in all the results, more than what is necessary for most of the examples.
Calculations are carried out on a CPU Intel Xeon Silver 4108 with 1.80GHz speed, 6 cores and a total 12 threads, as well as on an Nvidia GPU A100 80GB SXM.


## Uncertainty quantification in nonlinear simulations

In this section uncertainty quantification is performed for both linear and nonlinear responses to a loading field that is non-deterministic. Thousands of simulations are employed in Monte Carlo type of analysis to resolve for the statistics, for which parallelisation of the independent simulations become critical.
The example resembles the workflow of flight loads and wing stress analysis in an industrial setup. There will always be an element of uncertainty around computed loads, and what we show here is how for large displacements, the statistics need to be computed for every distinct loading. And for this, having a parallisation strategy as the one presented could potentially allow the computation of complex correlations and averages via  Montecarlo analysis.
Considering this, a static loading field is prescribed along the wings consisting of follower forces in the normal (out-of-plane) direction, as well as torsional moments (to mimic the added aerodynamic forces on an airfoil), with the characteristic that the force follows a normal distribution:

\begin{subequations}
\begin{align}\label{eq:normal_loading}
N&(\mu=1.5 \times 10^4 \mu_0, \sigma=0.15 \mu) \; \text{[forces]}  \\
N&(\mu=3 \times 10^4 \mu_0, \sigma=0.15 \mu)   \; \text{[moments]}
\end{align}
\end{subequations}

Three scenarios are studied: one in which very large nonlinear deformations are induced with $\mu_0 = 1$, and two small loading with  $\mu_0 = 10^{-2}$ and $\mu_0 = 10^{-3}$.
The distribution of displacements is characterised by means of Montecarlo simulations that run in parallel for a total of 1600 simulations.
Fig. [30](#org84fee40) shows the equilibrium at high loading ($\mu_0 = 1$) for two of the random cases (first and last of the 1600 computed). 

![img](figs_ext/MC1.png "Static equilibrium for two cases of the random excitation ($\mu_0=1$)")

Table [1](#orgaa17d30) shows the normalized statistics gathered from the response, in this case the expectation of tip of the wing displacements in the normal (out-of-plane) direction, $\mu_u $, and the corresponding standard deviation,  $ \sigma_u  $.

<table id="orgaa17d30" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 1:</span> Vertical wing-tip displacement statistics</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Case</th>
<th scope="col" class="org-right">\(\mu_u / \mu_0 \)</th>
<th scope="col" class="org-right">\( \sigma_u / \mu_0 \)</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">(\(\mu_0 = 1\))</td>
<td class="org-right">11.57</td>
<td class="org-right">1.35</td>
</tr>

<tr>
<td class="org-left">(\(\mu_0 = 0.01\))</td>
<td class="org-right">14.8</td>
<td class="org-right">2.4</td>
</tr>

<tr>
<td class="org-left">(\(\mu_0 = 0.001\))</td>
<td class="org-right">14.9</td>
<td class="org-right">2.3</td>
</tr>
</tbody>
</table>

We can see the statistics of the linear response ($\mu_0 = 0.001$) are fully captured by one single Montecarlo analysis, that is, output magnitudes such as equilibrium displacements correlate with the average input load. Whereas in cases with nonlinear deformations ($\mu_0 = 1$), the whole Montecarlo analysis would need to be carried out. This is akin to deterministic linear versus nonlinear analysis. 
Expanding this data at the tip to the entire right wing, Fig. [33](#orgf21d677) shows the mean normal displacement along the wing, $\mu_u$, and its standard deviation,  $\sigma_u$.
Note how despite the standard deviation of the input forces is the same along the wing, the uncertainty in the displacement output grows towards the tip as expected &#x2013;the aircraft being clamped at the root will only showcase 0 displacements there regardless of the input forces.

<>

Table [2](#orgad9d532) shows the times taken for the nonlinear case in both CPU and GPU. The computation of 1600 independent simulations in just over a minute, involving deformations of over 40% the wing semi-span as shown in Fig. [30](#org84fee40), highlights the potential of this methodology in more complex uncertainty quantification problems. Note that at this level of nonlinearity, our solvers are already two orders of magnitude faster than commercial solvers such as MSC Nastran even for a single simulation as demonstrated in <cite:&CEA2025>. The extension to thousands of cases with parallelisation on modern architectures is a key feature of this work with far-reaching applications in aircraft loads analysis.   

<table id="orgad9d532" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 2:</span> Static loading UQ computational times for 1600 paths</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Device</th>
<th scope="col" class="org-right">Time [sec.]</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">CPU (single)</td>
<td class="org-right">16.8 &times; 1600 = 26880</td>
</tr>

<tr>
<td class="org-left">CPU (parallel)</td>
<td class="org-right">317.4</td>
</tr>

<tr>
<td class="org-left">GPU</td>
<td class="org-right">67.6</td>
</tr>
</tbody>
</table>


## Computing derivatives of expectations

Now we set out to calculate the derivatives of the expectations previously computed concurrently in Sec. [2.3.1](#org490f785). While the Montecarlo paths are independent of each other and could therefore be run on different machines, having to do AD on the output statistics -gathered via collective operations-, forces the entire chain of operations to be within a single program. This makes for an interesting and challenging problem to propagate gradients through concurrent operations. A linear parameter $\alpha$ is introduced such that the follower forces and torsional moments in Eq. \eqref{eq:normal_loading} are $\mu = 10^4 (\frac{\alpha - 1.5}{4-1.5} + 1.5\times\frac{\alpha - 1}{5-1}) $. The selected output is the expectation of a 3-component vector, $\bm{r}(\alpha)$ of the wing-tip positions at $\alpha = 4.5$. Fig. [36](#orgc117c2e) shows a comparison between the derivative $\partial_{\alpha} \bm{r}^a = \partial \mathbb{E}[\bm{r}] / \partial \alpha $ using AD, and finite differences $ \partial_{\alpha} \bm{r}^{f}= (\bm{r}(\alpha + \epsilon) - \bm{r}(\alpha))/\epsilon  $. The relative error is calculated as  $||\partial_{\alpha} \bm{r}^{a} - \partial_{\alpha} \bm{r}^{f} || / ||\partial_{\alpha} \bm{r}^{a} ||  $, using the $l_2 $ norm.  

![img](figs/jac_ediff.png "Relative error for the derivative of expectations of the wing-tip position with respect to $\alpha $ (finite-differences VS AD).")

Another convergence  metric to investigate is the number of paths in the Montecarlo analysis. We take $[8, 80, 400, 800]$ paths and using the same error metric as for the finite differences calculate its evolution taking 4000 paths as the reference. The plot in [38](#orgbcefadc) shows that the convergence of the value function (expectation of the wing tip position) is faster than its derivative with respect to the loading parameter $\alpha $. This may or not be an issue  in optimization studies with expectations as sometimes high accuracy in the gradients is not that important as having a good direction.

![img](figs/jac_pdiff.png "Convergence with number of paths of expected wing-tip position and its Jacobian with respect to $\alpha $.")

While changing the input loading is good for verification studies, more realistic examples are solved in Fig. [40](#org9a9b826)  where the sensitivity of the expectations of wing root loads are calculated with respect to the FE matrices and eigenvectors (only indexes corresponding to the right wing nodes are shown in the heat map as the other are 0 because the model is clamped).
The derivatives in Fig. [38](#orgbcefadc) were calculated in forward mode, but the condensed FE matrices of the BUG model contain 99 condensed nodes, each with 6 components, so $(99 \times 6)^2 = 352836 $ entries, to what all the components in eigenvectors are also added, therefore only backward mode AD is possible in this case. The sensitivity of the loads with respect to the mass matrix increase towards the nodes at tip of the wing, which makes sense if we think a unit mass at the tip will produce much larger loads than the same unit mass close to the root where the aircraft is clamped.
The wall-time to calculate the gradient of wing-root loads expectations with respect to the mass matrix using 800 paths in the Montecarlo analysis was 48.0 and 297.9 in the GPU and CPU respectively.

![img](figs/jac_Ma4.png "Weight-normalized Jacobian of wing-root bending moment with respect to Mass matrix right-wing subcomponents")

![img](figs/jac_Ma42.png "Weight-normalized Jacobian of wing-root shear force with respect to Mass matrix right-wing subcomponents")


## Steady manoeuvre loads

We extend the previous analysis to a static aeroelastic case for varying angles of attack that represent a manoeuvre scenario.  We test the parallelisation by varying the flow density ($\pm 20 \%$ of the reference density 0.41 Kg/ m$^3$) as well and the flow velocity ($\pm 20 \%$ of the reference velocity 209.6 m/s). 16 different points for both density and velocity make a total number of 256 simulations. The Mach number is fixed at 0.7 corresponding to the reference flow condition values.
Fig. [43](#orgabb14aa) illustrates the 3D equilibrium of the airframe at the reference flight conditions. 

![img](figs_ext/monoeuvre3D.pdf "Aeroelastic steady equilibrium for increasing angle of attack manoeuvre")

In Fig. [45](#org686d2a8) the tip of the wing in Fig. [43](#orgabb14aa) is plotted for various angles-of-attach (AoA), normalized with the wing semi-span ($b= 25.9$) m. Comparison against linear analysis is carried out and the tip position in the nonlinear analysis falls down the linear counter part as expected. The flow velocities and density are selected at the maximum of the load envelope at 251.6 m/s and 0.5 kg/m$^3$ respectively. This highlights the potential need for geometrically nonlinear aeroelastic tools in future aircraft configurations under high loading scenarios. 

![img](figs/Manoeuvretip.png "Wing tip position for increasing angle of attack")

Table [3](#org70f48da) shows the computational times to run these simulations, which shows near no overhead in adding a few hundred of static calculations when moving from the single load case in the CPU to the GPU (nearly 8 seconds to 14 seconds, which amounts for 6 seconds cost when adding an extra 255 cases).

<table id="org70f48da" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 3:</span> Computational times for the multiple manoeuvre problem</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Device</th>
<th scope="col" class="org-right">Time [sec.]</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">CPU (single)</td>
<td class="org-right">7.71 &times; 256 = 1973.8</td>
</tr>

<tr>
<td class="org-left">CPU (parallel)</td>
<td class="org-right">52.8</td>
</tr>

<tr>
<td class="org-left">GPU</td>
<td class="org-right">14.4</td>
</tr>
</tbody>
</table>


## Dynamic loads at large scale

In this final example we perform a dynamic aeroelastic analysis to study the response of the aircraft to multiple 1-cos gusts for varying length, intensity and the density of the airflow. The mach number is kept constant at 0.7. In the examples above the aircraft was clamped while the aircraft is free here. A Runge-Kutta solver is employed to march in time the equations with a time step of $10^{-3}$ s and the total number of modes used was 100. Note the large size of the aeroelastic ODE system: 2 &times; 100 nonlinear equations plus 5 &times; 100 linear equations for the aerodynamic states with 5 poles, plus 4 equations for the quaternion tracking the rigid-body motion, for a combined ODE system of 704 equations.  
In addition, a total of 512 gusts cases are run concurrently for all possible combinations of 8 gust lengths between 50 and 200 meters, 8 gust intensities between 5 and 25 m/s, and 8 airflow densities between 0.34 and 0.48 Kg/m$^3$. This means that $512 \times 704 = 360448$ equations are being marched in time, in this case for 2 seconds which is enough to capture peak loads.
We have verified the concurrent implementation by satisfactory comparing single-point simulations to the same points within the parallel results.   
Table [4](#org42fd719) contains the simulation times of the calculation, which shows one order of magnitude increase in performance when running in parallel in the CPU versus a complete single simulation running sequentially, and another order of magnitude when moving from the CPU to a modern GPU. This exemplifies the power of modern hardware for scientific computation.

<table id="org42fd719" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 4:</span> Computational times multiple gust problem</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Device</th>
<th scope="col" class="org-right">Time [sec.]</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">CPU (single)</td>
<td class="org-right">27.8 &times; 512 = 14233.6</td>
</tr>

<tr>
<td class="org-left">CPU (parallel)</td>
<td class="org-right">922.6</td>
</tr>

<tr>
<td class="org-left">GPU</td>
<td class="org-right">38.2</td>
</tr>
</tbody>
</table>

In Fig. [49](#org933d013) the 3D reconstructed flight shape of the airframe is depicted for a gust of 150 m length, intensity of 20 m/s and flow density of 0.41 Kg/m$^3$

![img](figs_ext/Gust3D_3.png "Full aircraft Dynamic response to 1-cos gust excitation")

Figs. [51](#org13ec83e), [52](#org51dbe67) and [53](#org8279fa3) show the load diagrams for the maximum shear, torsion and out-of-pane bending at the wing root during the gust encounter, normalized with the aircraft weight and wing semi-span. They reflect the importance of running multiple of these simulations to assess the critical loads: maximum loads occur at different gust lengths of 65, 75, 115 m/s. This analysis would be extended to include various mass cases, flying altitudes etc. in an industrial environment, and it would be straight forward to extend our tools for this.   

![img](figs/GustShard_shear.png "Normalized max. wing-root internal loading, shear force")

![img](figs/GustShard_torsion.png "Normalized max. wing-root internal loading, torsional moment")

![img](figs/GustShard_bending.png "Normalized max. wing-root internal loading, bending moment")


### Load envelope differentiation

Since dynamic load envelopes have been constructed, the interest is to be able to obtain derivatives at the critical points as described in Sec. [2.3.2](#orge6de701). In opposition to the derivatives of expectations where all the operations are needed in the construction of the computational graph, here only a few of the most problematic cases are automatically identifies and its derivatives computed.
The metrics being tracked are wing-root shear, torsion and out-of plane bending moments.
The parallelisation is set for two gust intensities, two flow densities and 16 gust lengths to cover 1-cos gusts from 50 to 200 m/s with 10 m/s separation between points. Rather than a realistic example, this is set to test the machinery of the forager pattern and verify it can indeed discover critical load cases and automatically compute gradients. The gradient of these critical cases is also calculated with respect to the flow density, gust length and intensity (thus they are not only the parameters for the parallelisation but are also chosen to be the input variables of the sensitivity analysis, though any other input such as FE matrices could have been chosen). By looking at Figs. [51](#org13ec83e) -  [53](#org8279fa3), we can identify maximum loads at around 65, 75, 115 m/s gust lengths for shear, torsion and bending. Of the 64 cases analyzed, the forager step picked those with higher intensity and flow density as expected, but only 2 gust lengths of 70 m/s for the shear and torsion, and 110 m/s for the out-of-plane bending. The finer the discretization in the input of the parallelization, the closer to the actual maximum, but also the more computations that need to be run. It is therefore a balance whose best compromise is to be found on an actual optimization study.
Once those peaks are found, the algorithms triggers the sensitivity analysis, for which we have shown a verification for the 70 m/s gust with maximum density and intensity in Table [5](#org426984d). Finite differences are computed with an $\epsilon = 10^{-4} $ and the absolute relative difference with AD is shown as $ \Delta $ for each of the input parameters ($ \partial  \rho_{\inf} $ for the flow density derivative, $ \partial L $ for the gust length, and $ \partial w $ for the gust intensity). The computational wall-time for entire forager loop (concurrent loads, filtering of critical cases and sensitivities on them) was 60.1 and 359.5 seconds on the GPU and CPU respectively.  

<table id="org426984d" border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">
<caption class="t-above"><span class="table-number">Table 5:</span> Comparison of peak gust loads at the critical points, AD versus FD</caption>

<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-left" />

<col  class="org-right" />

<col  class="org-left" />

<col  class="org-right" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">&#xa0;</th>
<th scope="col" class="org-right">objective</th>
<th scope="col" class="org-right">\( \partial  \rho_{\inf} \)</th>
<th scope="col" class="org-left">\( \Delta \)</th>
<th scope="col" class="org-right">\( \partial L \)</th>
<th scope="col" class="org-left">\( \Delta \)</th>
<th scope="col" class="org-right">\( \partial w \)</th>
<th scope="col" class="org-left">\( \Delta \)</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">Shear</td>
<td class="org-right">0.897</td>
<td class="org-right">1.670</td>
<td class="org-left">\( 1.6 \times 10^{-5} \)</td>
<td class="org-right">-0.0026</td>
<td class="org-left">\( 4 \times 10^{-6} \)</td>
<td class="org-right">0.035</td>
<td class="org-left">\( 1. \times 10^{-7} \)</td>
</tr>

<tr>
<td class="org-left">Torsion</td>
<td class="org-right">0.025</td>
<td class="org-right">-0.018</td>
<td class="org-left">\( 4.2 \times 10^{-4} \)</td>
<td class="org-right">-0.0007</td>
<td class="org-left">\( 3.9 \times 10^{-7} \)</td>
<td class="org-right">0.0018</td>
<td class="org-left">\( 1.2 \times 10^{-6} \)</td>
</tr>

<tr>
<td class="org-left">Bending</td>
<td class="org-right">0.300</td>
<td class="org-right">0.329</td>
<td class="org-left">\( 2.9 \times 10^{-5} \)</td>
<td class="org-right">0.00003</td>
<td class="org-left">\( 1.7 \times 10^{-4} \)</td>
<td class="org-right">0.012</td>
<td class="org-left">\( 9.2 \times 10^{-8} \)</td>
</tr>

<tr>
<td class="org-left">&#xa0;</td>
<td class="org-right">&#xa0;</td>
<td class="org-right">&#xa0;</td>
<td class="org-left">&#xa0;</td>
<td class="org-right">&#xa0;</td>
<td class="org-left">&#xa0;</td>
<td class="org-right">&#xa0;</td>
<td class="org-left">&#xa0;</td>
</tr>
</tbody>
</table>


# Conclusions

A modal-based, geometrically nonlinear formulation has been implemented with concurrent capabilities for multiple load-cases in modern hardware architectures. We have applied state-of-the-art techniques and tools to the computation of the sizing aeroelastic loads encountered during the design of commercial aircraft. These can span thousands of simulations in multiple scenarios and the efficiency of our approach has been highlighted while accounting for nonlinear effects and rigid body dynamics. Moreover, sensitivity of the response to model inputs using automatic differentiation has been demonstrated across parallel simulations.    
Computational times of under a minute are achieved for 256 manoeuvres varying flow conditions and for 512 dynamic gust responses undergoing large displacements.
This performance makes the solution suitable for two different applications: uncertainty quantification of the nonlinear aircraft response to a non-deterministic loading and integration of the software in larger multidisciplinary optimisation studies.
The former has been presented on a problem where a field of forces with an stochastic component induces very large deformations. Montecarlo analysis then allows to build statistics of the response and we have shown how we can not only run multiple paths concurrently to build a distribution of the outputs, but also differentiate expectations of those with respect to any input in the aeroelastic analysis.   
For the sensitivity analysis of load envelopes, a strategy named The Forager Pattern has been introduced whereby the multiple cases are launched in parallel, critical cases are filtered out, and new simulations for AD are spawn to obtain gradients on the most demanding loads. Since the examples shown are already in the memory boundaries of a single GPU or CPU, further scale-up requires the use of multiple devices, which we have already implemented. 
Introduction of various mass cases, as it is done in industrial scenarios, is also a feasible target. The resulting framework thus combines prediction of sizing aeroelastic loads in a nonlinear fashion with the computation of their gradients with respect to design variables, and therefore it is intended to be part of multidisciplinary design optimization studies that include the large set of loading requirements in the certification process.
Despite dynamic loads and aeroelastic instabilities being still obtained using medium fidelity tools, a future enhancement with high fidelity CFD aerodynamics is the one major addition left to achieve a versatile, production-ready framework. 

\section*{Acknowledgements}
This work has received funds from Innovate UK, under project 10002372, managed by the UK
Aerospace Technology Institute.
We acknowledge computational resources and support provided by the Imperial College Research Computing Service (<http://doi.org/10.14469/hpc/2232>).
The benchmark examples for the GPU computations were carried out at the Imperial College cluster. Code and examples are open-source.


# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> Both implementation and examples can be found at \url{https://github.com/ACea15/FENIAX}.
