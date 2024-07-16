<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
<script type="text/x-mathjax-config">
 MathJax.Hub.Config({
 tex2jax: {
 inlineMath: [['$', '$'] ],
 displayMath: [ ['$$','$$'], ["\\[","\\]"] ]
 }
 });
</script>

# FENIAX sensitivity analysis design doc

mermaid
flowchart LR
    A[Design variables d] --> B(NastranHandler)
    A --> C(SensitivityNastran)
    B --> |p| D(FEM4INAS)
    C --> |$$\partial p\partial d$$| E[$$\partial r/\partial d$$]
    D --> |$$\partial r/\partial p$$|E


mermaid
classDiagram
NastranDesignModel : Array d
NastranDesignModel : return_bdf()
NastranHandler <|-- SensitivityNastran
NastranHandler : NastranDesignModel model
NastranHandler : _run_natran()
NastranHandler : get_rom()
SensitivityNastran : _run_nastran_parallel()
SensitivityNastran : get_rom_sensitivity()
SensitivityFeniax : NastranDesignModel model
SensitivityFeniax : _run_feminas()
SensitivityFeniax : get_design_sensitivity()
SensitivityFeniax -- NastranHandler
SensitivityFeniax -- SensitivityNastran


## Global variables

### NASTRAN_LOC

location of Nastran execution file

### Parameter (thickness)

Thickness ratios are interpolated using control point values

params : values at control points
coord : coordinates at control points
order : order of polynomial interpolation

## Input in dictionary form:

File names, directory structure: fixed

### Structural
P_PSHELL : float (nt,)\
　Normalized parameters for PSHELL entry

C_PSHELL : class\
　C_THICK.get_val takes P_THICK as input, returns DICT_PSHELL as output
> DICT_PSHELL : dict ('PID' : Array, 'T' : Array,  . . .)\
　PSHELL entry values

### Aero
P_CAERO1 : float (na,)\
　Normalized parameters for CAERO1 entry

C_CAERO1 : class\
　C_CAERO1.get_val takes P_CAERO1 as input, returns DICT_CAERO1 as output
> DICT_CAERO1 : dict ('EID' : Array, 'LCHORD' : Array, . . .)\
　CAERO1 entry values

### Mass
P_CONM2 : float (ncm,)\
　Normalized parameters for CONM2 entry

C_CONM2 : class\
　C_CONM2.get_val takes P_CONM2 as input, returns DICT_CONM2 as output
> DICT_CONM2 : dict ('G' : Array, . . .)\
　CONM2 entry values

### Material
P_MAT2 : float (nm,)\
　Normalized parameters for MAT2 entry

C_MAT2 : class\
　C_MAT2.get_val takes P_MAT2 as input, returns DICT_MAT2 as output
> DICT_MAT2 : dict ('MID' : Array, . . .)\
　MAT2 entry values
