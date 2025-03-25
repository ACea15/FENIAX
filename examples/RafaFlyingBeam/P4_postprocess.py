# [[file:modelgen.org::*Postprocess][Postprocess:1]]
import feniax.plotools.utils as putils
import feniax.plotools.uplotly as uplotly
import feniax.preprocessor.solution as solution  
sol0 = solution.IntrinsicReader("./results_m1")
x, y = putils.pickIntrinsic2D(sol0.data.dynamicsystem_s1.t,
                              sol0.data.dynamicsystem_s1.X1,
                              fixaxis2=dict(node=1, dim=2)) # given 2 data
fig = uplotly.lines2d(x, y)
fig.show()
# Postprocess:1 ends here

# [[file:modelgen.org::*Symmetric velocities][Symmetric velocities:1]]
fig = None
for i in range(6):
    sol_as = solution.IntrinsicReader(f"./results_symm1vz{i}")
    x, y = putils.pickIntrinsic2D(sol_as.data.dynamicsystem_s1.t,
                                  sol_as.data.dynamicsystem_s1.X1,
                                  fixaxis1=None,
                                  fixaxis2=dict(node=0, dim=4)) # given 2 data
    fig = uplotly.lines2d(x, y, fig)
fig.show()
# Symmetric velocities:1 ends here

# [[file:modelgen.org::*Antisymmetric velocities][Antisymmetric velocities:1]]
fig = None
for i in range(6):
    sol_as = solution.IntrinsicReader(f"./results_antisymm1vz{i}")
    x, y = putils.pickIntrinsic2D(sol_as.data.dynamicsystem_s1.t,
                                  sol_as.data.dynamicsystem_s1.X1,
                                  fixaxis1=None,
                                  fixaxis2=dict(node=0, dim=4)) # given 2 data
    fig = uplotly.lines2d(x, y, fig)
fig.show()
# Antisymmetric velocities:1 ends here
