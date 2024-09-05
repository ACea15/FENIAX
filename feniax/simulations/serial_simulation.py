from feniax.simulations.simulation import Simulation
import jax.numpy as jnp


class SerialSimulation(Simulation, cls_name="serial"):
    def trigger(self):
        # Implement trigger for SerialSimulation
        self._run_systems()

    def _run_systems(self):
        # Implement _run for SerialSimulation

        sys0 = None
        eta0 = None
        for k, sys in self.systems.items():  # only one item in the loop
            sys.set_system()
            sys.set_solver()
            sys.set_xloading()
            sys.set_states()
            if sys0 is not None:
                q0 = self._compute_q0(sys, sys0)
                # q0=None
                sys.set_eta0(eta0)
            else:
                q0 = None
                sys.set_eta0()
            sys.set_ic(q0)
            sys.solve()
            if sys0 is None:
                eta0 = sys.build_connection_eta()
            sys.build_solution()
            sys0 = sys
            # self._post_run(k, sol_obj)

    def _post_run(self, sys_name, sol_obj):
        # Implement _post_run for SerialSimulation

        if self.settings.save_objs:
            self.sol.add_dict("dsys_sol", sys_name, sol_obj)

    def _compute_q0(self, sys, sys0):
        states = sys.settings.states
        num_states = sys.settings.num_states
        states0 = sys0.settings.states
        q0 = jnp.zeros(num_states)
        for k, vi in states.items():
            if k in states0.keys():
                q0 = q0.at[vi[:]].set(sys0.qs[-1, states0[k][:]])
                q0 = q0.at[-4:].set(jnp.array([1.0, 0.0, 0.0, 0.0]))

        return q0
