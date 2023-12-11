attr_latex: :width 0.8\\textwidth
author: Alvaro Cea and Rafael Palacios
bibliography: /home/acea/Documents/Engineering.bib
caption: Low intensity gust, x-component
header-includes:
`\synctex=1`{=latex}; `\usepackage[margin=1in]{geometry}`{=latex}; `\usepackage{graphicx}`{=latex}; `\usepackage{amsmath,bm}`{=latex}; `\usepackage[version=4]{mhchem}`{=latex}; `\usepackage{siunitx}`{=latex}; `\usepackage{longtable,tabularx}`{=latex}; `\usepackage{booktabs}`{=latex}; `\usepackage{tabularx,longtable,multirow,subfigure,caption}`{=latex}; `\setlength\LTleft{0pt}`{=latex}; `\usepackage{mathrsfs}`{=latex}; `\usepackage{amsfonts}`{=latex}; `\usepackage{enumitem}`{=latex}; `\usepackage{mathalpha}`{=latex}; `\renewcommand{\figurename}{\bf \small Figure}`{=latex}; `\renewcommand{\tablename}{\bf \small Table}`{=latex}; `\newcommand{\de}{\delta}`{=latex}; `\newcommand{\ve}{\text{v}}`{=latex}; `\newcommand{\lo}{\mathcal{L}}`{=latex}; `\newcommand{\vt}{\overline{\delta\bm{\theta}}}`{=latex}; `\newcommand{\vu}{\overline{\delta\bm{u}}}`{=latex}; `\newcommand{\e}{\bm{\mathfrak{e}}}`{=latex}; `\newcommand{\E}{\bm{\mathbb{E}}}`{=latex}; `\newcommand{\T}{\bm{\mathcal{T}}}`{=latex}; `\newcommand{\fra}{(\mathtt{1})}`{=latex}; `\newcommand{\frb}{(\mathtt{2})}`{=latex}; `\newcommand{\fri}{(\mathfrak{i})}`{=latex}; `\newcommand{\bs}[1]{\boldsymbol{#1}}`{=latex}; `\newcommand{\rhoinf}{\rho}`{=latex}; `\newcommand{\Vinf}{U}`{=latex}; `\newcommand{\Cl}[1]{c_{l_{#1}}}`{=latex}; `\newcommand{\barCl}[1]{\bar{c}_{l_{#1}}}`{=latex}; `\newcommand{\Cm}[1]{c_{m_{#1}}}`{=latex}; `\newcommand{\barCm}[1]{\bar{c}_{m_{#1}}}`{=latex}; `\newcommand{\AIC}{\bs{\mathcal{A}}}`{=latex}
name: fig:gust001
title: A Nearly-Real Time Nonlinear Aeroelastic Simulation Architecture
  Based on JAX

`\synctex=1`{=latex}

`\usepackage[margin=1in]{geometry}`{=latex}

`\usepackage{graphicx}`{=latex}

`\usepackage{amsmath,bm}`{=latex}

`\usepackage[version=4]{mhchem}`{=latex}

`\usepackage{siunitx}`{=latex}

`\usepackage{longtable,tabularx}`{=latex}

`\usepackage{booktabs}`{=latex}

`\usepackage{tabularx,longtable,multirow,subfigure,caption}`{=latex}

`\setlength\LTleft{0pt}`{=latex}

`\usepackage{mathrsfs}`{=latex}

`\usepackage{amsfonts}`{=latex}

`\usepackage{enumitem}`{=latex}

`\usepackage{mathalpha}`{=latex}

`\renewcommand{\figurename}{\bf \small Figure}`{=latex}

`\renewcommand{\tablename}{\bf \small Table}`{=latex}

`\newcommand{\de}{\delta}`{=latex}

`\newcommand{\ve}{\text{v}}`{=latex}

`\newcommand{\lo}{\mathcal{L}}`{=latex}

`\newcommand{\vt}{\overline{\delta\bm{\theta}}}`{=latex}

`\newcommand{\vu}{\overline{\delta\bm{u}}}`{=latex}

`\newcommand{\e}{\bm{\mathfrak{e}}}`{=latex}

`\newcommand{\E}{\bm{\mathbb{E}}}`{=latex}

`\newcommand{\T}{\bm{\mathcal{T}}}`{=latex}

`\newcommand{\fra}{(\mathtt{1})}`{=latex}

`\newcommand{\frb}{(\mathtt{2})}`{=latex}

`\newcommand{\fri}{(\mathfrak{i})}`{=latex}

`\newcommand{\bs}[1]{\boldsymbol{#1}}`{=latex}

`\newcommand{\rhoinf}{\rho}`{=latex}

`\newcommand{\Vinf}{U}`{=latex}

`\newcommand{\Cl}[1]{c_{l_{#1}}}`{=latex}

`\newcommand{\barCl}[1]{\bar{c}_{l_{#1}}}`{=latex}

`\newcommand{\Cm}[1]{c_{m_{#1}}}`{=latex}

`\newcommand{\barCm}[1]{\bar{c}_{m_{#1}}}`{=latex}

`\newcommand{\AIC}{\bs{\mathcal{A}}}`{=latex}

<div class="LATEX_PROPERTIES drawer" markdown="1">

</div>

<div class="abstract" markdown="1">

This paper presents a novel aeroelastic framework that has been rebuilt
for performance and robustness. Leveraging on the numerical library JAX,
a highly vectorised codebase is written that is capable of nonlinear,
time-domain computations and achieves two orders of magnitude
accelerations compare to its predecessor. This brings the calculations
to run close to if not in real-time, thus opening new possibilities for
aircraft aeroelastic analysis which have traditionally been constrained
to either linear, frequency domain solutions, or to their nonlinear
counterparts but with a narrower scope. Moreover, the approach
seamlessly integrates with conventional aeroelastic load packages which
facilitates the analysis of complex aircraft configurations. An
extensive verification has been carried out and is presented herein,
starting from canonical beam cases up to a full industrial FE model that
is assessed for dynamic aeroelastic loads.

</div>

Introduction
============

The ever need for performance and operating costs reduction, together
with the more recent major push for a cleaner aviation, are driving new
aircraft designs outside the conventional envelop with an emphasis for
high aspect ratio wings to minimise induced drag. Furthermore,
discoveries in advanced lighter materials may well bring down the weight
of the overall vehicle but potentially also increase the flexibility of
the main structures. In this scenario, **aeroelastic analysis** are
expected to become critical in the very early phases of the wing design
process: while the field was more important in post-design stages to
ensure in-flight integrity, it can become paramount to capture the
cross-couplings between disciplines now. In this more nonlinear
landscape, the overall aerodynamic performance needs to be calculated
around a flight shape with large deformations
[cite:&GRAY2021](cite:&GRAY2021); the input for efficient control laws
account for the steady state and nonlinear couplings
[cite:&Artola2021](cite:&Artola2021); and the loads ultimately sizing
the wings are atmospheric disturbances computed in the time-domain
[cite:&CESNIK2014a](cite:&CESNIK2014a). A more holistic approach to the
design also increases the complexity of the processes exponentially, and
the trade-offs and cost-benefit analysis may not be possible until
robust computational tools are in-place to simulate the different
assumptions. The **certification** of new air vehicles is another
important aspect that requires 100,000s of load cases simulations
[cite:&Kier2017](cite:&Kier2017), as it considers manoeuvres and gust
loads at different velocities and altitudes, and for a range of mass
cases and configurations. This poses another challenge for new methods
that aim to include new physics since they normally incur in prohibitly
expensive computational times. Lastly, the mathematical representation
of the airframe, embodied in the complex **Finite-Element Models**
(FEMs) built by organizations, encompasses a level of knowledge that is
to be preserved when including the new physics mentioned above. These 3
facts set the goals for the current enterprise: 1) to be able to perform
geometrically nonlinear aeroelastic analysis, 2) to work with generic FE
models in a non-intrusive manner, and 3) to achieve a computational
efficiency that is equivalent to present linear methods if not faster.
Grounded on previous developments where the first two points where
demonstrated [cite:&PALACIOS2019](cite:&PALACIOS2019), cite:&CEA2021,
[cite:&CEA2023](cite:&CEA2023) we tackle the third point herein with a
new implementation that achieves remarkable computational performance.
The numerical library JAX [cite:&jax2018github](cite:&jax2018github) was
leveraged to produce highly vectorised, automatically differentiated
routines that are managed by a modular, object-oriented approach in
Python. The power of JAX for scientific computation has been proved
recently in fluid dynamics [cite:&BEZGIN2023](cite:&BEZGIN2023) and
solid mechanics [cite:&XUE2023](cite:&XUE2023) applications. We add to
those an aeroelastic solution to enhance already built models for linear
loads analysis. This aligns with current efforts to build robust methods
that incorporate nonlinear effects to complex 3-D FEMs, via stick models
[cite:&RISO2023](cite:&RISO2023) or other modal-based methods
[cite:&DRACHINSKY2022](cite:&DRACHINSKY2022).  
The proposed solution procedure can be divided into the five stages
shown in Fig. [1](#aircraft_process2): 1) A linear (arbitrarily complex)
model is the input for the analysis. 2) Model condensation is employed
to derive a skeleton-like substructure, along the main load path,
containing the main features of the full 3D model. 3) The modes of the
reduced structure are evaluated in intrinsic variables (velocities and
strains) and used as a basis of a Galerkin-projection of the
geometrically-nonlinear intrinsic beam equations. 4) The projected
equations are solved in time-domain under given forces: aerodynamic
influence coefficient matrices are obtained here from DLM and a rational
function approximation (RFA) [cite:&ROGER1975](cite:&ROGER1975) is used
to transform to the time domain. We have also presented a more efficient
data-driven approach that circumvents the additional states added by the
RFA in [cite:&PALACIOS2024](cite:&PALACIOS2024) and the approach would
also be amendable more accurate Computational Fluids Aerodynamics (CFD).
5) The full 3D solution using the nonlinear 1D solution, the reduced
order transformations and interpolation. Therefore
geometrically-nonlinear behaviour is captured along the principal
skeleton and the linear response of the cross-sections (in the form of
ribs and fuselage reinforcements) is also represented –if nonlinear
deformations also occur in the cross-sections, there is no reliable
analysis other than high-fidelity solutions of the full model. The
overall procedure has been implemented in what we have named as
*Nonlinear Modal Reduced Order Model* (NMROM).

[file:./figs/aircraft\_process2.pdf](./figs/aircraft_process2.pdf)

The structure of the rest of the paper is as follows. Sec. [2](#Theory)
presents a summary of the mathematical description that conforms the
backbone behind the computational implementation of `FEM`~4~`INAS`
(Finite-Element-Models for Intrinsic Nonlinear Aeroelastic Simulations),
the high performance software for aeroelasticity outlined in Sec.
[3](#Computational implementation). Sec. [4](#Examples) shows the
verification examples that cover the static and dynamic structural
response of canonical cases and of a simplified aircraft model, and the
aeroelastic response to a gust of a full aircraft configuration. The
improvements in performance are highlighted in all of the examples.
Lastly, sec. [5](#Conclusions) summarises the the achievements and
further developments planned for future work.

Theory
======

In this section we briefly describe the backbone theory of the proposed
methods for nonlinear aeroelasticity modelling. For further details, see
cite:&CEA2021, cite:&CEA2023.

Airframe idealisation
---------------------

Our starting point is an airframe that we want to approximate
mathematically. Starting from solid mechanics, geometric considerations
lead to a (nonlinear) 1D approximation on one hand, while stiffness
considerations result in linear elastic response on the other;
discretisation of the linear equations and applying the appropriate
boundary conditions produce the FE models typically used in design, and
which may be enhanced with experimental datasets to account for the
missing details in mathematical process. The combination of the 1D
description with a condensed version of the FE model along the main load
paths allow us the construction of the NMROM.  
The 3D equations and the process by which the 1D kinematics are imposed
on the deformation tensor have been described in
`\cite{CEA2021a}`{=latex}. The main assumption is that cross-sectional
deformations of the solid body in the reference configuration are not
coupled to the main dimension as moving through configurations in time.
As a result, distributed internal stresses act only through the normal
of the cross-sections in the undeformed configuration. Applying the
appropriate integration over the cross sectional reference area of the
of the distributed traction forces, a Cosserat rod model is built, where
the deformed state on the full domain is approximated by a deformable
space curve $\Gamma$ – identified with the aircraft major load-paths.

The primary variables are the local inertial (linear and angular)
velocities, $\pmb{\ve}(s,t)$ and $\pmb{\omega}(s,t)$, and the local
internal forces and moments $\pmb{f}(s,t)$ and $\pmb{m}(s,t)$, all along
$\Gamma$. They are function of the spanwise coordinate along, $s$, and
time, $t$. They are defined as 3-dimensional column matrices whose
coefficients are the components of the corresponding vector quantity in
the local material frame. As an example, the three components of
$\pmb{m}$ are the torsional, out-of-plane bending and in-plane moments
resulting for the cross-sectional integration of the stress field.

For simplicity, $\pmb{\ve}$ and $\pmb{\omega}$ will be grouped into the
(unknown) velocity state variable, $\bm{x}_1$ and the internal force and
moments, $\pmb{f}$ and $\pmb{m}$ will be combined into the (unknown)
force state $\bm{x}_2$, namely

```{=latex}
\begin{align}\label{eq2:x1}
\pmb{x}_1= \begin{bmatrix}
 \pmb{\ve} \\ \pmb{\omega}
\end{bmatrix} ,
\hspace{1cm} \pmb{x}_2 =  \begin{bmatrix}
\pmb{f} \\ \pmb{m}
\end{bmatrix}
\end{align}
```
Adding to this the compatibility relations of the field variables, the
equations are written here in compact form as in
`\cite{PALACIOS2019}`{=latex}:

```{=latex}
\begin{subequations}\label{eq2:intrinsic_eqs}
\begin{align}
\mathcal{M}\dot{\pmb{x}}_1-\pmb{x}_2'-\pmb{\mathsf{E}}\bm{x}_2+ \lo_1(\pmb{x}_1)\mathcal{M}\pmb{x}_1 + \lo_2(\pmb{x}_2)\mathcal{C}\pmb{x}_2 & = \pmb{f}_1  \\
\mathcal{C}\dot{\pmb{x}_2}-\pmb{x}_1' + \pmb{\mathsf{E}}^\top\pmb{x}_1- \lo_1^\top(\pmb{x}_1)\mathcal{C}\pmb{x}_2 & = \pmb{0}
\end{align}
\end{subequations}
```
where $\lo_1$ and $\lo_2$ are linear operators and $\pmb{\mathsf{E}}$ a
constant matrix.

The applied forces and moments per unit length, $\bm{f}_1$, are
$\pmb{f}_1= [ \pmb{f}_e, \pmb{m}_e]$, where $\pmb{f}_e(s,t)$ and
$\pmb{m}_e(s,t)$ are follower forces and moments per unit span length,
respectively. As before, they are expressed in terms of their components
in the deformed material frame. Note also that displacements and
rotations do not appear explicitly in equations. The above description
is geometrically-exact with quadratic nonlinearities only.

The material properties introduce the final set of relations, named the
constitutive equations, which under linear assumptions are written as,

```{=latex}
\begin{equation}\label{eq2:costitutive1}
\begin{bmatrix} \bm{\gamma}(s,t) \\ \bm{\kappa}(s,t)
\end{bmatrix}  =
\bm{\mathcal{C}}(s) \begin{bmatrix}
\bm{f}(s,t) \\ \bm{m}(s,t)
\end{bmatrix}  \hspace{1cm} ; \hspace{1cm}
\begin{bmatrix} \bm{p}(s,t) \\ \bm{h}(s,t)
\end{bmatrix}  =
\bm{\mathcal{M}}(s) \begin{bmatrix}
\pmb{\ve}(s,t) \\ \bm{\omega}(s,t)
\end{bmatrix} 
\end{equation}
```
the compliance matrix $\bm{\mathcal{C}}$ relates sectional forces and
moments to strains and curvatures – in problems such as plasticity or
fracture mechanics, where this relation cannot be assumed linear, the
constitutive connection would have to be updated in the simulation loop.
This matrix is difficult to obtain for complex structures with composite
materials and usually homogenization or asymptotic methods are utilized
to predict it `\cite{Dizy2013}`{=latex}. The mass matrix,
$\bm{\mathcal{M}}$, links velocities and momenta, and is not trivial to
obtain for structures with distributed inertia either. This work
circumvents having to calculate explicit expressions of
$\bm{\mathcal{C}}$ and $\bm{\mathcal{M}}$ by solving the equations in
modal space and linking them to the modal shapes and their derivatives;
furthermore, they are time-independent due to the material formulation,
which greatly facilities the nonlinear computations.  
Using the intrinsic modes and the projection of the state variables, a
Galerkin projection is performed on Eqs.
`\eqref{eq2:intrinsic_eqs}`{=latex} such that
$\pmb{x}_1 = \pmb{\phi}_1\pmb{q}_1$ and
$\pmb{x}_2 = \pmb{\phi}_2\pmb{q}_2$. Taking the inner product in the 1D
domain,
$\langle \pmb{u},\pmb{v}  \rangle = \int_\Gamma \pmb{u}^\top \pmb{v} ds$,
for any $\pmb{u}\in\mathbb{R}^6$ and $\pmb{v}\in\mathbb{R}^6$, the final
form of the equations is `\cite[Ch. 8]{PALACIOS2023}`{=latex}

```{=latex}
\begin{equation}
\label{eq2:sol_qs}
\begin{split}
\dot{q}_{1j} &= \delta^{ji}\omega_{i}q_{2i}-\Gamma^{jik}_{1}q_{1i}q_{1k}-\Gamma^{jik}_{2}q_{2i}q_{2k}+ \eta_{j}  \\
\dot{q}_{2j} &= -\delta^{ji}\omega_{i}q_{1i} + \Gamma_2^{ijk}q_{1i}q_{2k}
\end{split}
\end{equation}
```
where we have used implicit summation over repeated indices, with
$\delta^{ij}$ the Kronecker delta. The coefficients $\pmb{\Gamma}_{1}$
and $\pmb{\Gamma}_{2}$ are third-order tensors that encapsulate the
nonlinear modal couplings in the response (the former introduces the
gyroscopic terms in the dynamics and the latter introduces the
strain-force nonlinear relation), and $\pmb{\eta}$ is the modal
projection of the external forcing terms. They can be written as:

```{=latex}
\begin{align}\label{eq2:gammas12}
\Gamma_{1}^{jkl} & = \langle \pmb{\phi}_{1j}, \lo_1(\pmb{\phi}_{1k})\pmb{\psi}_{1l}\rangle, \nonumber \\
\Gamma_{2}^{jkl} & = \langle \pmb{\phi}_{1j}, \lo_2(\pmb{\phi}_{2k})\pmb{\psi}_{2l}\rangle,  \\
\eta_{j} & = \langle \pmb{\phi}_{1j}, \pmb{f}_1\rangle  \nonumber
\end{align}
```
with $\pmb{\psi}_1 = \bm{\mathcal{M}}\pmb{\phi}_1$ and
$\pmb{\psi}_2 = \bm{\mathcal{C}}\pmb{\phi}_2$ also cast as momentum and
strain mode shapes. In other words, each natural vibration mode can be
uniquely expressed in terms of velocity, force/moment, momentum, or
strain variables. While those would be redundant in a conventional
linear vibration analysis, they will enable to identify all the
coefficients in the geometrically-nonlinear equations
`\eqref{eq2:sol_qs}`{=latex}. Furthermore, they can all be directly
obtained from a condensation of a general built-up finite-element model
along load paths, as outlined next.

Nonlinear aeroelastic system
----------------------------

The full aeroelastic solution is described extending Eq.
`\eqref{eq2:sol_qs}`{=latex} with gravity forces, $\bm{\eta}_g$,
aerodynamic forces and gust disturbances, $\bm{w}_g$. Control states can
also be included [cite:&CEA2021a](cite:&CEA2021a), but they are not
necessary for this work. For a set of reduced frequencies and a given
Mach number, the DLM (or a higher fidelity aerodynamic method) yields
the modal forces in the frequency domain. The current implementation
uses Roger's rational function approximation to those GAFs, which
results in the follower modal forces

```{=latex}
\begin{equation}\label{eq3:eta_full}
\begin{split}
\bm{\eta}_a = \tfrac12\rho_\infty U_\infty^2 & \left(\vphantom{\sum_{p=1}^{N_p}} \pmb{\mathcal{A}}_0\bm{q}_0 +\frac{c}{2U_\infty}\pmb{\mathcal{A}}_1 \bm{q}_1 +\left(\frac{c}{2U_\infty}\right)^2 \pmb{\mathcal{A}}_2\dot{\bm{q}}_1   \right.  \\
& \left. + \pmb{\mathcal{A}}_{g0}\bm{v}_g +\frac{c}{2U_\infty}\pmb{\mathcal{A}}_{g1} \dot{\bm{v}}_g +\left(\frac{c}{2U_\infty}\right)^2 \pmb{\mathcal{A}}_{g2}\ddot{\bm{v}}_g +  \sum_{p=1}^{N_p} \pmb{\lambda}_p  \right) 
\end{split}
\end{equation}
after rescaling the matrices above, the subsequent nonlinear aeroelastic system is written as \cite{CEA2023}
\begin{equation}
\label{eq3:intrinsic_full_aeroelastic}
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
```
Computational implementation
============================

The main contribution of this work is a new computational implementation
with two clear goals: on the one hand to make the code more efficient
and suitable for a large number of load cases, including
high-performance computing (HPC); on the other hand, to introduce a
flexible architecture for aeroelastic solutions, easily extensible and
user friendly. In addition, computation of gradients of the code via
algorithmic differentiation have been taken into consideration as a
crucial part for design optimisation and will be exemplified in future
work. The resulting software is completely new, performs over two orders
of magnitude faster than its predecessor and is highly modular with an
implementation based on software design patterns. The key elements of
the development are highlighted next. \\subsection{Nonlinear aeroelastic
framework for computations on modern hardware architectures} Numerical
computations have been written in JAX, a Python library designed for
high-performance numerical computing with focus on machine learning
activities `\cite{jax2018github}`{=latex}. This library is developed and
maintained by Google research team and used by DeepMind among others. It
combines XLA (accelerated linear algebra) and Autograd, the former being
a compiler that optimises models for different hardware platforms, the
latter is an Automatic Differentiation (AD) tool in Python. Moreover,
its extensible system of composable function transformations provides a
set of important features for Computational Science as illustrated in
Fig. `\ref{fig:JAX-overview}`{=latex}. For instance, the vmap function
allows for complex **vectorisation** operations and the pmap function
for Single-Program Multiple-Data (SPMD) **parallelisation**. Both
forward and reverse mode **automatic differentiation** are supported.
Finally the **just-in-time compilation** (jit) relies on the XLA engine
to compile and execute functions on CPUs but also on accelerators such
as GPUs and TPUs, offering a versatile solution for seamlessly
connecting the software to various types of hardware without requiring
extra CUDA code, or a Domain Specific Language (DSL).

```{=latex}
\begin{figure}[htbp]
\centering
\includegraphics[width=0.35\textwidth]{./figs/jaxlogo2.pdf}
\caption{\label{fig:JAX-overview} JAX capabilities for modern scientific computing}
\end{figure}
```
The new capabilities come at the expense of a higher restriction in the
way the code is written. Compilation and transformations in JAX only
work for functionally pure programs, which pushes the software to comply
with a nonconventional functional paradigm. Some of these
characteristics are **pure functions**, i.e. functions that have no side
effects, **input/output** stream management needs to be placed outside
the numerical algorithms, **inmutability** of arrays, and **function
composition**, or the ability to create functions by chaining other
callables.  
These very constraints are the enabler to achieve the capabilities
describe above via the many abstractions implemented internally in the
library. The challenge after the algorithms have been implemented
appropriately is to manage a general aeroelastic code that can deal with
arbitrary configurations, solutions that may range from purely
structural to aeroelastic simulations with multibody components, and
even workflows that may involve various simulations in serial or in
parallel. A good example is the calculation of the nonlinear trimmed
flight state on which various gust profiles are to be assessed. A mixed
approach has been employed for this whereby numerical algorithms are
written using functional programming but the flow of execution is
managed using an object oriented approach that focus on modularity and
extensibility. This will be outline in the following section.

Software design
---------------

*"Supporting state-of-the-art AI research means balancing rapid prototyping and quick iteration with the ability to deploy experiments at a scale traditionally associated with production systems."*.
Jax target inside DeepMind would also be desirable in a scientific
research environment. It however entails a good amount of labour and
expertise into the field of software design, whose payoffs are only
realisable in the long term.

Fig. [1](#components_architecture) shows a high-level view of this first
version of the software in terms of components. A Configuration
component builds the necessary settings for the simulation, including
geometric coordinates, load-paths information. The Configuration is
injected into the Driver component that initialises the Simulation
component, the Systems and the Solution component, after which it
triggers the simulation. The Systems are run as managed by the
Simulation component and encapsulate the various equations to be solved
(time marching, nonlinear static equilibrium or stability for instance).
The solution component acts as a memory-efficient container of the new
data to be kept as the solution process advances, and it is responsible
for loading (from a previous simulations) and writing solution data too.
It is thus passed to every System.

![Components architecture
diagram](figs/components_architecture.png "components_architecture")

Fig. [2](#classes_architecture) shows a lower view of the abstractions,
interfaces between classes and how they interact via their public
methods. The inputs to the program may be given via a .yaml file or a
python dictionary in memory. The starting point in the main file is the
initialisation of the driver corresponding to the mathematical
description to be solved (so far only the intrinsic modal is available,
Eqs. `\eqref{eq3:intrinsic_full_aeroelastic}`{=latex}). The intrinsic
driver computes (or loads), as a pre-simulation step, the intrinsic
modal shapes and nonlinear tensors from the linear stiffness and mass
matrices and the nodal coordinates; then it runs the cases by triggering
the simulation class. This class is responsible for managing how the
systems are being run (in serial, in parallel, or even in a coupling
process between systems). From the configuration settings, the intrinsic
system loads the equations (dqs), the external loads in Eqs.
`\eqref{eq2:sol_qs}`{=latex}, such as point-forces, gravity or modal
aerodynamic GAFs. Various libraries can be chosen to either solve the
static equations or march in time if the solution is dynamic;
importantly, the JAX-based Diffrax library has been integrated and
supports ordinary, stochastic and controlled equations, with many
solvers and multiple adjoint methods which could be used in an
optimization framework. This initial layout of the software is expected
to evolve and to be consolidated as the software matures.

![Class architecture UML
diagram](figs/classes_architecture.png "classes_architecture")

`\newpage`{=latex}

Examples
========

All the cases presented are part of a Test suite that has been built as
a critical step for long term software management. They serve as a
demonstration of the approach's ability to deal with geometric
nonlinearities, the accuracy of the solvers when compared to full FE
simulations, and the computational gains that can be achieved. Table
`\ref{table:benchmarks}`{=latex} introduces the cases that are discussed
below with the improvements in performance from the new implementation.
All computations are carried out on a single core of the same CPU, an
i7-6700 with 3.4 GHz clock speed. The old code based on Python was not
optimised and made heavy use of for-loops instead of vectorised
operations. These results convey the potential improvements in
scientific software when paying attention to the implementation solely.
Besides of this, it is also worth remarking the very short times in the
solutions, which is also largely due to a formulation in modal space
that naturally leads to reduced order models and easily caters for
vectorised operations. Six examples are presented, first three are
static cases and the other three are dynamic cases with the last one
being a time domain aeroelastic response to a gust. The model complexity
is also augmenting starting with beams to then move to a representative
aircraft of medium complexity, the so-called Sail Plane, and finally
considering an industrial-scale aircraft, the XRF1 model. Note the
longer dynamic simulation of the Sail Plane wing compared to the XRF1
gust response: despite i.e. more operations in the solution time step,
driven the largest eigenvalue in the solution, was much smaller in the
Sail Plane results

```{=latex}
\begin{table}[h!]
  \begin{center}
    \caption{Simulation times for cases part of the test suite}
    \label{table:benchmarks}
    \begin{tabular}{lrll}
      \toprule
      Model & Time [s] & Time (old) & Speed-up\\[0pt]
      \midrule
      ArgyrisCantilever (7 load-cases) & 7.8 & 9m:44s & $\times$74.9\\[0pt]
      Simo45Beam (11 load cases) & 7.1 & 1m:45s & $\times$14.8\\[0pt]
      SailPlane (6 load cases) & 8.1 & 56.3s & $\times$6.95\\[0pt]
      ShellBeam (Dynamic, 20 sec. with 85 modes) & 34.3 & 6h:16m:53s & $\times$659.3\\[0pt]
      SailPlaneWing (Dynamic, 15 sec. with 50 modes) & 10.88 & 2h:18min:35s & $\times$764.2\\[0pt]
      XRF1-Gust (Dynamic Aeroelastic, 15 sec. with 70 modes) & 17.4 & 1h:38min:28s & $\times$339.5\\[0pt]
      \bottomrule
            &  &  & \\[0pt]
    \end{tabular}
  \end{center}
\end{table}
```
Canonical cases
---------------

Structural static and dynamic cases of simple models undergoing very
large deformations are shown in this section. Even though the models are
simple, the complexity here is found in the more challenging physics
than a normal airplane undergoes in terms of geometric nonlinearities.

### Beams static response

Initially we consider a series of simple beam models that have been
standard for the verification of geometrically nonlinear theories. First
a 2D problem of a straight cantilever under a follower tip force is
shown in Fig. [3](#ArgyrisBeamPlot). The structure is deformed into a
hook undergoing very large deformations. The beam properties are 100 cm
length, cross-sectional area of 20 cm$^2$, $I=3/2$ cm$^4$,
$E=2.1\times 10^7$ N/cm$^2$. The example first appeared in
`\cite{Argyris1981}`{=latex} but a finer discretisation with 25 nodes is
used here, which explains the small differences. This case, which
consists of 7 different load increments, runs in the new implementation
with the full set of modes (150) in 7.8 seconds and it used to run in
almost 10 minutes, for a 75 times speed-up.

![Beam under follower tip load pointing
down](figs/ArgyrisBeam.png "ArgyrisBeamPlot")

  
Next, a curved cantilever under static follower loads is shown in Fig.
`\ref{fig:simo45}`{=latex}. It was first analysed by Bathe and Bolourchi
`\cite{Bathe1979}`{=latex} and it has extensively been used to validate
nonlinear structural implementations
`\cite{Simo1986a,Werter2016}`{=latex}. It increments the complexity with
respect to the previous example by undergoing very large deformations in
3D space. The geometry consists of a 45-degree bend circle of 100 m
radius, 1 m square cross section, Young’s modulus $E = 107$ Pa, and
negligible Poisson ratio. A discretisation of 15 nodes with the full set
of modes (90) are employed in the solution. This case runs in 7 seconds
as opposed to the 2 minutes taken in the previous implementation. The
difference is substantial but not as large as the previous case, where
less modes were used in the solution. Also note previously Jacobians of
the equations were being provided to the Newton-Raphson solvers, which
are not (yet) implemented in the new solver.

![45-deg bend cantilever deformations](figs/s45follower.png "simo45")

![45-deg bend cantilever
deformations](figs/Simoverificationfollower.png "simo45")

### Free vibrations of thin-walled cantilever

Next, we study the dynamic behaviour of a cantilever previously studied
in `\cite{PALACIOS2019}`{=latex} and shown in Fig.
`\ref{fig:cantilever}`{=latex}. Three equivalent models are built: 1)
with beam elements and lumped masses, 2) shell elements with lumped
inertia, and 3) shell elements with distributed inertia. The material
properties are $E = 106$ N/m$^2$, $v = 0.3$ and $\rho = 1$ Kg/m$^3$. MSC
Nastran 4-noded elements (CQUADs) are employed; mass properties are
given either as density in the material cards or as discrete mass
elements (CONM2s) representing sectional inertia; and interpolation
elements (RBE3s) which link the nodes in full and reduced models. The
differences between the shell and beam models, are shown as well as how
those are captured by the current methodology in dynamical problems.

NMROMs are built from three linear models in MSC Nastran and a 30-node
spanwise discretisation along the main load path is used for the model
condensation using Guyan reduction. This was found to provide converged
solution for the nonlinear response studied –which surpassed 30$\%$ of
displacements with respect to the cantilever length. The free-vibrations
of the system are investigated by imposing an initial parabolic velocity
distribution along the undeformed cantilever. A small excitation results
in a linear response as shown in the displacements in y and z directions
in Fig. `\ref{fig:cantilever_sollinA}`{=latex}. Axial displacements are
exactly zero. The lumped shell and beam models show identical response
while the distributed mass model is slightly shifted with respect to
them. Geometrically-nonlinear effects become relevant as the amplitude
of the initial velocity is increased. Displacements over 35$\%$ are
obtained as presented in the time history of the free-end displacements
in Fig. `\ref{fig:cantilever_sollinB}`{=latex}. Converged simulations
are obtained with 85 modes and a time step of $\Delta t = 0.002$.
**The difference between previous and current implementations is nearly three orders of magnitude: 34.3 seconds versus 6 hours, 16 minutes and 53 seconds.**

Structural verification of a representative configuration
---------------------------------------------------------

A representative FE model for aeroelastic analysis of a full aircraft
without engines is used to demonstrate the capabilities of the current
methodology on large finite-element models. The aircraft’s main wing is
composed of wing surfaces, rear and front spars, wing box and ribs.
Flexible tail and rear stabiliser are rigidly attached to the wing.
Structural and aerodynamic models are shown in Fig.
`\ref{fig:SailPlane}`{=latex}. This is a good test case as it is not
very complex yet has all the features for detailed aeroelastic analysis
and it is available open source.

```{=latex}
\begin{figure}[h!]
\centering
\includegraphics[width=0.9\textwidth]{./figs/SailPlane2}
\caption{Sail Plane structural and aerodynamic models}\label{fig:SailPlane}
\end{figure}
```
### Geometrically nonlinear static response

The static equilibrium of the aircraft under prescribed loads is first
studied with a NMROM built with the first 50 modes. Follower loads
normal to the wing are applied at the tip of each wing. The response
under loads of 200, 300, 400, 480 and 530 KN is shown in Fig.
`\ref{fig:sp_static}`{=latex}. Nonlinear static simulations on the
original full model (before condensation) are also carried out in MSC
Nastran and are included in the figure. The interpolation elements in
Nastran are used to output the displacements at the condensation nodes
for direct comparison with the NMROM results. To quantify the difference
between both sets of results, tip displacements, in global coordinates,
for the 530 KN load and the full model Nastran calculations are
$u_x = -0.217$ m $u_y = -1.348$ m, $u_z = 7.236$ m; while NMROM
calculations yield $u_x = -0.219$ m $u_y = -1.352$ m, $u_z = 7.249$ m.
This represents an error of 0.19$\%$ for a 25.6$\%$ tip deformation of
the wing semi-span, $b = 28.8$ m.

```{=latex}
\begin{figure}[h!]
\centering
\includegraphics[width=0.99\textwidth]{./figs/sp_static3}
\caption{Aircraft static response under wing-tip follower loads}\label{fig:sp_static}
\end{figure}
```
Geometric nonlinearities are better illustrated by representing a
sectional view of the wing as in Fig. `\ref{fig:sp_axial}`{=latex}.
Deformations in the z-direction versus the metric $\sqrt{x^2+y^2}$ are
shown in Fig. `\ref{fig:sp_axial}`{=latex}(a) where MSC Nastran linear
solutions are also introduced. This allows appreciating more clearly the
shortening effect in nonlinear computations. On the other hand, the
length of the main wing after reduction to the 1D domain is computed
before and after deformations ($L_w = \int_{\Gamma_{w}} ds$). Because
the resultant axial stiffness is much higher than bending or torsional
stiffness, the structure is nearly inextensible. This effect, however,
is not captured by linear approximations. Fig.
`\ref{fig:sp_axial}`{=latex}(b) shows the percentage change in the total
length of the main wings with the driving set of forces.

```{=latex}
\begin{figure}[h!]
 \centering
 \subfigure[Nonlinear shortening effects]{\includegraphics[width=0.53\textwidth]{./figs/sp_axial2}}
\subfigure[Elongation of the main wing]{\includegraphics[width=0.46\textwidth]{./figs/sp_axial}}%\label{fig:sp_dis}
\caption{Static geometrically-nonlinear effects on the aircraft main wing}\label{fig:sp_axial}
\end{figure}
```
Excellent agreement is obtained between the nonlinear static
calculations from MSC Nastran and those of the proposed approach. In
terms of computational efficiency,
**the new implementation has taken 8 seconds, while the full model in Nastran takes 5 minutes and 47 seconds.**  
A first attempt into recovering the 3D structural response has been put
in place with a very good agreement between the obtained configuration
and Nastran 400 solution. A mapping between the aerodynamic model
(surface of panels) and the condensed model is built using the positions
and rotations from the solution; then an RBF kernel is placed at every
corner of each of the panels, which deforms the 3D structural mesh. Fig.
`\ref{fig:sailplane3Dstatic}`{=latex} shows the overlap in the Nastran
solution (in blue) and the NMROM (in red) for the 2000 and 5300 KN
loadings. A similar but more realistic strategy is to use the nodes
connected by the RBE3s elements instead of the aerodynamic panels as the
interpolation points. Furthermore, the cross-sectional information from
the 3D modal shapes could also be added if necessary. The accuracy of
those will be assessed in future work but the current results already
show the potential of this method to reproduce full 3D nonlinear
solutions and, for instance, be able to couple the solvers to a CFD
aerodynamic model for more accurate results on the aerodynamic side.

![](./figs/3Dstatic-frontview.png) ![](./figs/3Dstatic-sideview.png)

### Large-amplitude nonlinear dynamics

This test case demonstrates the accuracy of the NMROM approach for
dynamic geometrically-nonlinear calculations and was first introduced in
[cite:&CEA2021](cite:&CEA2021). The right wing of Fig.
`\ref{fig:SailPlane}`{=latex} is considered and dynamic nonlinear
simulations are carried out and compared to MSC Nastran linear and
nonlinear analysis (SOL 109 and 400, respectively) on the full FE model.
A force is applied at the wing tip with a triangular loading profile,
followed by a sudden release of the applied force to heavily excite the
wing.

The dynamic response is presented in Fig.
`\ref{fig:sp_results}`{=latex}, where results have been normalised with
the wing semi-span. As expected, linear analysis over predicts vertical
displacements and does not capture displacements in the $x$ and $y$
directions. NMROMs were built with 15 and 50 modes. An impressive
reduction of computational time is achieved in the new implementation
that for the 50 modes case and with a small time step of 0.001 seconds
takes **10.9 seconds** while the nonlinear response of the full model in
**Nastran took 1 hour 22 minutes.**

![Span-normalised tip displacements in the dynamic simulation,
z-component](figs/wingSP_z.png "wingSP_z")

![Span-normalised tip displacements in the dynamic simulation,
x-component](figs/wingSP_x.png)

![Span-normalised tip displacements in the dynamic simulation,
y-component](figs/wingSP_y.png)

Dynamic loads on an industrial configuration
--------------------------------------------

The studies presented in this section are based on a reference
configuration developed to industry standards by Airbus as part of the
eXternal Research Forum (XRF), from which the aircraft takes its name,
XRF1. The aircraft represents a long-range wide-body transport airplane
and has been used as a research platform for collaboration between the
company, universities and research institutions. Fig.
`\ref{fig8:xrf1-model}`{=latex} shows the full aeroelastic model split
up into the structural, mass and aerodynamic components. The FE model
contains a total of around 177400 nodes, which are condensed into 176
active nodes along the reference load axes through interpolation
elements. A Guyan or static condensation approach is used for the
reduction. One of the strengths of the present methodology is that The
aerodynamic model contains $\sim 1,500$ aerodynamic panels.

```{=latex}
\begin{figure}[th!]
\centering
\includegraphics[width=1.\textwidth]{./file:figs/xrf1-model.pdf}
\caption{Modified XRF1 aeroelastic subcomponents}\label{fig8:xrf1-model}
\end{figure}
```
![Residual $\alpha_2$ in current
implementation](figs/XRF1Plot_alpha2.png "xrf1_alphas")

![Residual $\alpha_2$ in previous
implementation](figs/XRF1Plot_alpha2old.png)

![Residual $\alpha_1$ current implementation](figs/XRF1Plot_alpha1.png)

![Residual $\alpha_1$ in previous
implementation](figs/XRF1Plot_alpha1old.png)

Fig. `\ref{fig:gust001}`{=latex} shows the normalised tip response to a
low intensity 1-cos gust shape for a 0.81 Mach flow. A very good
agreement is found with NASTRAN calculations based on the full FE model
for this case with very small displacements, i.e. linear. On the other
hand, a high intensity gust in Fig. `\ref{fig:gust2}`{=latex} induces
large deformations whose effects are only captured by the nonlinear
solver.

![Low intensity gust, y-component](figs/Gust3Plot_y.png)

![Low intensity gust, z-component](figs/Gust3Plot_z.png)

![High intensity gust, x-component](figs/Gust4Plot_x.png "gust2")

![High intensity gust, y-component](figs/Gust4Plot_y.png)

![High intensity gust, z-component](figs/Gust4Plot_z.png)

An important remark about these computations is that the gusts have been
input in the reference configuration. Undergoing updates in the
implementation aim to update the gust intensity at each panel with its
normal component. This will account for the added nonlinearity of the
changing in downwash.

Conclusions
===========

This paper has presented a modal-based description that incorporates
geometrically nonlinear effects due to structural slenderness onto
generic FE models initially built for linear analysis. While the
underlying theory had already been introduced, a new implementation has
been put in-place for both high-performance and software modularity,
with the numerical library JAX as the engine powering the computations.
Furthermore, a relevant amount of test cases accompany the software, of
which a subset has been presented to illustrate the main capabilities
that may range from a canonical beam undergoing extremely large
deformations to a full-vehicle nonlinear aeroelastic response. A major
highlight are the computational accelerations experimented which reach
two orders of magnitude in dynamic analysis. This is due to the heavy
use of vectorisation and just-in-time compilation. The ability to
recover the full 3D state from the NMROM was also demonstrated and
compared to the solution in NASTRAN.  
In the immediate future two objectives are foreseen with this
implementation: first, a further assessment of the computational gains
by running the examples presented here on GPUs; second and more
important, the propagation of derivatives in the solution process via
the Algorithmic Differentiation tool embedded in JAX. This will complete
a fully differentiated aeroelastic framework that can run very efficient
in modern software architectures while enhancing traditional FE models
that can be very complex by construction but lack the physics of
geometrically nonlinear effects. After that, increasing the fidelity in
the load calculations to consider CFD-based aerodynamics would be an
additional necessary step in order to achieve a more accurate nonlinear
aeroelastic methodology.

bibliography:\~/Documents/Engineering.bib
