from pyNastran.op2.op2 import OP2
import pandas as pd
import plotly.express as px
import fem4inas.preprocessor.solution as solution

# op2 = OP2()
# op2.set_additional_matrices_to_read({b'OPHP':False})
# op2.read_op2("./NASTRAN/SOL146/run_cao.op2")



# data = op2.displacements[1].data
# df = pd.DataFrame(dict(#time=op2.displacements[1].dts,
#                   z=data[:,19,2]))

# fig = px.line(df,  title='Life expectancy in Canada')
#fig.show()


sol = solution.IntrinsicReader("resultsGust")


df = pd.DataFrame(dict(#time=op2.displacements[1].dts,
                  z=gust.data.dynamicsystem_s1.ra[:,0, 25]))




fig = px.line(df,  title='Life expectancy in Canada')
fig.show()

