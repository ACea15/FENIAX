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

```mermaid
flowchart LR
    A[Design variables d] --> B(NastranHandler)
    A --> C(SensitivityNastran)
    B --> |p| D(FEM4INAS)
    C --> |$$\partial p\partial d$$| E[$$\partial r/\partial d$$]
    D --> |$$\partial r/\partial p$$|E
```

```mermaid
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
```

## Global variables

### NASTRAN_LOC

location of Nastran execution file

### Parameter (thickness)

Thickness ratios are interpolated using control point values

params : values at control points
coord : coordinates at control points
order : order of polynomial interpolation
