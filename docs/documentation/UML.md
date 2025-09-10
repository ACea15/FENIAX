# Code design and software architecture

The figure below shows a high-level view of this first version of the software with,


-   feniax main: entry point of the simulation
-   Config object with input settings.
-   Driver class: initialises all relevant objects in the computation
    such as the simulation, and the systems objects.
-   Simulation class: responsible for running the various systems
    appropriately, including setting the initial conditions and passing
    data from one system of equations to another.
-   System class: sets the computations to solve the corresponding
    system of equations, including the solver library that should be
    called, the system of equations and the arguments to the solvers.
-   The Solution class is a memory-efficient container loading and writing solution data to disk.

![Abstract workflow](./img/abstract_classes.png)


A Configuration component builds the necessary settings for the simulation, including geometric coordinates, load-paths information. Input settings can be seen in [Inputs](api/inputs.md), and include:

- Condensed stiffness and mass matrices along load paths (should be suitable for eigenvalue analysis).
- Load paths connection: interpolation elements to connect to other FE nodes; aerodynamic forces applied along these paths.
- Aerodynamic model via GAFs. Preliminary DLM model. Automatically built from corner coordinates. Steady loads: Corrections may be needed. 

The Configuration is injected into the Driver component that initialises the Simulation component, the Systems and the Solution component, after which it triggers the simulation. The Systems are run as managed by the Simulation component and encapsulate the various equations to be solved (time marching, nonlinear static equilibrium or stability for instance). 
The Solution component acts as a memory-efficient container of the new data to be kept as the solution process advances, and it is responsible for loading (from a previous simulations) and writing solution data too. It is thus passed to every System. 
The intrinsic modal shapes and nonlinear tensors are computed from the linear stiffness and mass matrices and the nodal coordinates in the driver in a pre-simulation step ([Modes](api/modes.md) and [Couplings](api/couplings.md)). Then it runs the cases by triggering the simulation class. This class is responsible for managing how the systems are being run (in serial, in parallel, or even in a coupling process between systems). From the configuration settings, the intrinsic system loads the equations and the external loads such as point-forces, gravity or modal aerodynamic GAFs. Various libraries can be chosen to either solve the static equations or march in time if the solution is dynamic; importantly, the JAX-based Diffrax library has been integrated and supports ordinary, stochastic and controlled equations, with many solvers and multiple adjoint methods which can be used in an optimization framework. This initial layout of the software is expected to evolve and to be consolidated as the software matures. 

## Config

See [Config](api/config.md) in codebase:

-   Builds configuration settings of the simulation
-   Static object in jit functions
-   The config object encapsulates a set of containers that are data
    classes with the simulation variables
-   The object is serialised and saved into a .yaml file.
-   This config .yaml file is needed if postprocessing in Streamlit is
    to be deployed

### Containers

- Config object build from these containers: [Inputs](api/inputs.md) 
- Branchless programming: critical optimisation technique. Avoidconditional statements in core computational subroutines: <https://en.algorithmica.org/hpc/pipelining/branchless/>.
- In FENIAX most conditionals happen at this level.

### Inheritance model
The inheritance model below shows how the systems are being composed. 

![Inheritance structure](./img/inheritance_classes.png)

There are various ways to run the code, with a chain of requirements between the various options: 
- Normal mode: workflow of simulations happening sequentially major functions for the systems to be solved can be found in dq\_….py. Computation of intrinsic modes, couplings, etc. happen prior to the system solution.

- Fast: entire solution within one function such that memory copies to cuda devices are avoided. Computation of intrinsic modes, modal couplings, aerodynamic matrices happen within a single function, from within the solution of the system of equations is also called. Importantly, the functions employed in normal mode are used for the solution, thereby avoiding code duplication and promoting modular design.

- AD: the entire solution within one function as well, but needs inputs/outputs for the differentiation to take place. It uses Fast solvers. 

- Shard flexible: workflow as in flexible but with inputs over which to build solutions in parallel.

- Shard fast: Similarly, everything happens within a function.

- Shard AD: shard the inputs, AD on collective quantities.

