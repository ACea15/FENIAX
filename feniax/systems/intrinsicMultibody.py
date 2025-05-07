import feniax.systems.intrinsic_system as intrinsic_system
import feniax.intrinsic.dynamicMB as dynamicMB
import jax.numpy as jnp

from feniax.ulogger.setup import get_logger

logger = get_logger(__name__)


class DynamicIntrinsicMB(intrinsic_system.DynamicIntrinsic, cls_name="dynamic_intrinsicMB"):
    
    def set_xloading(self):

        ...

    def set_states(self):
        self.settings.build_states(self.fem.num_modes, self.fem.num_nodes)

    def set_args(self):
        label = self.settings.label.split("_")[-1]
        logger.info(f"Setting arguments for System main function with label {label}")
        solver_args = getattr(libargs, f"arg_{label}")
        self.args1 = solver_args(self.sol, self.settings, self.fem, eta_0=self.eta0)
        
    def set_eta0(self, eta0=None):
        num_modes = self.fem.num_modes
        if eta0 is None:
            self.eta0 = jnp.zeros(num_modes)
        else:
            assert len(eta0) == num_modes, "wrong length in eta0"
            self.eta0 = eta0

    def set_ic(self, q0):
        logger.info(f"Setting initial conditions for the System (qs(0))")
        if q0 is None:
            self.q0 = jnp.zeros(self.settings.num_states)
            if self.settings.init_states is not None:
                for k, v in self.settings.init_states.items():
                    if type(v[0]) == "str" and v[0].lower == "prescribed":
                        qi0 = jnp.array(v[1])
                        assert len(qi0) == len(
                            self.settings.states[k]
                        ), f"error prescribed {k} length"
                    else:
                        if callable(v[0]):
                            init_f = v[0]
                        else:
                            init_f = getattr(initcond.Container, v[0])
                        x = init_f(
                            *v[1], fem=self.fem
                        )  # 6xNn inputs to approx (e.g. a velocity field).
                        # function to calculate qs
                        init_x = initcond.mapper[self.settings.init_mapper[k]]
                        sol_lstsq = init_x(self.sol.data.modes, x)
                        qi0 = sol_lstsq[0]  # TODO: save to sol
                    self.q0 = self.q0.at[self.settings.states[k]].set(qi0)
            if "qr" in list(self.settings.states.keys()):
                if (
                    self.settings.init_states is None
                    or "qr" not in self.settings.init_states.keys()
                ):  # set rotational quaternions
                    # if they have not been set before
                    qr0 = jnp.hstack(
                        [jnp.array([1.0, 0.0, 0.0, 0.0])]
                        * (len(self.settings.states["qr"]) // 4)
                    )
                    self.q0 = self.q0.at[self.settings.states["qr"]].set(qr0)
        else:
            self.q0 = q0
            
    def set_system(self):
        
        label = f"main_{self.settings.label}"
        logger.debug(f"Setting {self.__class__.__name__} with label {label}")                      
        self.dFq = getattr(dynamicMB, label)

    def build_solution(self):
        
        logger.info(f"Building postprocessing fields (strains, velocities, positions, etc.)")        
        # X1 = postprocess.compute_velocities(
        #     self.sol.data.modes.phi1l, self.qs[:, self.settings.states["q1"]]
        # )
        # X2 = postprocess.compute_internalforces(
        #     self.sol.data.modes.phi2l, self.qs[:, self.settings.states["q2"]]
        # )
        # X3 = postprocess.compute_strains(
        #     self.sol.data.modes.psi2l, self.qs[:, self.settings.states["q2"]]
        # )
        # if self.settings.bc1.lower() == "clamped":
        #     tn = len(self.qs)
        #     ra0 = jnp.broadcast_to(self.fem.X[0], (tn, 3))
        #     Cab0 = jnp.broadcast_to(jnp.eye(3), (tn, 3, 3))
        # else:
        #     if self.settings.rb_treatment == 1:
        #         ra_n0 = self.fem.X[0]
        #         Rab_n0 = jnp.eye(3)
        #         Cab0, ra0 = postprocess.integrate_node0(
        #             X1[:, :, 0], self.settings.dt, ra_n0, Rab_n0
        #         )
        # Cab, ra = postprocess.integrate_strains_t(
        #     ra0,
        #     Cab0,
        #     X3,
        #     self.sol.data.modes.X_xdelta,
        #     self.sol.data.modes.C0ab,
        #     self.config,
        # )
        # self.sol.add_container(
        #     "DynamicSystem",
        #     label="_" + self.name,
        #     q=self.qs,
        #     X1=X1,
        #     X2=X2,
        #     X3=X3,
        #     Cab=Cab,
        #     ra=ra,
        #     t=self.settings.t,
        # )
        # if self.settings.save:
        #     self.sol.save_container("DynamicSystem", label="_" + self.name)
