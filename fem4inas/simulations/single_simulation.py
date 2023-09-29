from fem4inas.simulations.simulation import Simulation

class SingleSimulation(Simulation, cls_name="single"):
        
    def trigger(self):
        # Implement trigger for SerialSimulation
        self._run_systems()

    def _run_systems(self):
        # Implement _run for SerialSimulation

        for k, sys in self.systems.items(): # only one item in the loop
            sys.set_generator()
            sys.set_solver()
            sys.set_xloading()
            sys.set_states()
            sys.set_ic(q0=None)
            sys.solve()
            sys.build_solution()
            #self._post_run(k, sol_obj)

    def _post_run(self, sys_name, sol_obj):
        # Implement _post_run for SerialSimulation

        if self.settings.save_objs:
            self.sol.add_dict('dsys_sol', sys_name, sol_obj)

