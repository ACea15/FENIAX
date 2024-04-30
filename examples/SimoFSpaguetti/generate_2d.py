import fem4inas.plotools.reconstruction as rec
import fem4inas.preprocessor.solution as solution
import fem4inas.preprocessor.configuration as configuration

results = "results_try"
config = configuration.Config.from_file(f"./{results}/config.yaml")
sol = solution.IntrinsicReader(f"./{results}")

r, u = rec.rbf_based("./NASTRAN/Shell_na50.bdf",
                     config.fem.X,
                     sol.data.dynamicsystem_s1.t,
                     sol.data.dynamicsystem_s1.ra,
                     sol.data.dynamicsystem_s1.Cab,
                     sol.data.modes.C0ab,
                     vtkpath="./vtk2d/conf",
                     plot_timeinterval=10,
                     tolerance=1e-3,
                     size_cards=8)
