from functools import partial

import feniax.intrinsic.args as libargs
import feniax.intrinsic.dq_dynamic as dq_dynamic
import feniax.intrinsic.dq_static as dq_static
import feniax.intrinsic.initcond as initcond
import feniax.intrinsic.postprocess as postprocess
import feniax.preprocessor.containers.intrinsicmodal as intrinsicmodal
import feniax.preprocessor.solution as solution
import feniax.systems.sollibs as sollibs
import feniax.intrinsic.xloads as xloads

import jax
import jax.numpy as jnp
from feniax.systems.system import System


def _staticSolve(eqsolver, dq, t_loads, q0, dq_args, sett):
    def _iter(qim1, t):
        args = dq_args + (t,)
        sol = eqsolver(dq, qim1, args, sett)
        qi = jnp.array(sol.value)  # _static_optx(dq, qim1, args, config)
        return qi, qi

    qcarry, qs = jax.lax.scan(_iter, q0, jnp.array(t_loads))
    return qs


@partial(jax.jit, static_argnames=["config", "tn"])
def recover_fields(q1, q2, tn, X, phi1l, phi2l, psi2l, X_xdelta, C0ab, config):
    ra0 = jnp.broadcast_to(X[0], (tn, 3))
    Cab0 = jnp.broadcast_to(jnp.eye(3), (tn, 3, 3))
    X1 = postprocess.compute_velocities(phi1l, q1)

    X2 = postprocess.compute_internalforces(phi2l, q2)
    X3 = postprocess.compute_strains(psi2l, q2)
    Cab, ra = postprocess.integrate_strains_t(ra0, Cab0, X3, X_xdelta, C0ab, config)

    return X1, X2, X3, ra, Cab


@partial(jax.jit, static_argnames=["config", "tn"])
def recover_staticfields(q2, tn, X, phi2l, psi2l, X_xdelta, C0ab, config):
    ra0 = jnp.broadcast_to(X[0], (tn, 3))
    Cab0 = jnp.broadcast_to(jnp.eye(3), (tn, 3, 3))
    X2 = postprocess.compute_internalforces(phi2l, q2)
    X3 = postprocess.compute_strains(psi2l, q2)
    Cab, ra = postprocess.integrate_strains_t(
        ra0,
        Cab0,
        X3,
        # fem,
        X_xdelta,
        C0ab,
        config,
    )

    return X2, X3, ra, Cab


class IntrinsicSystem(System, cls_name="intrinsic"):
    def __init__(
        self,
        name: str,
        settings: intrinsicmodal.Dsystem,
        fem: intrinsicmodal.Dfem,
        sol: solution.IntrinsicSolution,
        config,
    ):
        self.name = name
        self.settings = settings
        self.fem = fem
        self.sol = sol
        self.config = config
        # self._set_xloading()
        # self._set_generator()
        # self._set_solver()

    def set_args(self):
        label = self.settings.label.split("_")[-1]
        solver_args = getattr(libargs, f"arg_{label}")
        self.args1 = solver_args(self.sol, self.settings, self.fem, eta_0=self.eta0)
        
    def set_eta0(self, eta0=None):
        num_modes = self.fem.num_modes
        if eta0 is None:
            self.eta0 = jnp.zeros(num_modes)
        else:
            assert len(eta0) == num_modes, "wrong length in eta0"
            self.eta0 = eta0
        self.set_args()

    def set_ic(self, q0):
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

    def set_xloading(self, compute_follower=True, compute_dead=True, compute_gravity=True):

        force_follower = None
        force_dead = None
        force_gravity = None
        if self.settings.xloads.follower_forces and compute_follower:
           # self.settings.xloads.build_point_follower(
           #     self.fem.num_nodes, self.sol.data.modes.C06ab
           # )           
           force_follower = xloads.build_point_follower(self.settings.xloads.x,
                                                         self.settings.xloads.follower_points,
                                                         self.settings.xloads.follower_interpolation,
                                                         self.fem.num_nodes,
                                                         self.sol.data.modes.C06ab
                                                         )
        if self.settings.xloads.dead_forces and compute_dead:
            force_dead = xloads.build_point_dead(self.settings.xloads.x,
                                                 self.settings.xloads.dead_points,
                                                 self.settings.xloads.dead_interpolation,
                                                 self.fem.num_nodes,
                                                 )
        if self.settings.xloads.gravity_forces and compute_gravity:
            if self.fem.constrainedDoF:
                force_gravity = xloads.build_gravity(self.settings.xloads.x,
                                     self.settings.xloads.gravity,
                                     self.settings.xloads.gravity_vect,
                                     self.fem.Ma0s,
                                     self.fem.Mfe_order0s)
            else:
                force_gravity = xloads.build_gravity(self.settings.xloads.x,
                                     self.settings.xloads.gravity,
                                     self.settings.xloads.gravity_vect,
                                     self.fem.Ma,
                                     self.fem.Mfe_order)
        self.sol.add_container(
            "PointForces", label="_" + self.settings.name,
            force_follower=force_follower,
            force_dead=force_dead,
            force_gravity=force_gravity,
            x=self.settings.xloads.x
        )

    def set_states(self):
        self.settings.build_states(self.fem.num_modes, self.fem.num_nodes)

    def set_solver(self):
        self.states_puller, self.eqsolver = sollibs.factory(
            self.settings.solver_library, self.settings.solver_function
        )

    def build_connection_eta(self):

        elevator_index = self.settings.aero.elevator_index
        elevator_link = self.settings.aero.elevator_link
        aero = getattr(self.sol.data, f"modalaeroroger_{self.settings.name}")
        A0hat = aero.A0hat
        B0hat = aero.B0hat
        q = self.qs[-1]
        states = self.settings.states
        omega = self.sol.data.modes.omega
        q2 = q[states["q2"]]
        q0i = -q2[2:] / omega[2:]
        q0 = jnp.hstack([q2[:2], q0i])
        qx = q[states["qx"]]
        eta_aero = xloads.eta_steadyaero(q0, A0hat)
        eta_elevator = xloads.eta_controls(qx, B0hat, elevator_index, elevator_link)
        eta0 = eta_aero + eta_elevator
        return eta0

    def save(self):
        pass


class StaticIntrinsic(IntrinsicSystem, cls_name="static_intrinsic"):
    # def set_ic(self, q0):
    #     if q0 is None:
    #         self.q0 = jnp.zeros(self.fem.num_modes)
    #     else:
    #         self.q0 = q0

    def set_xloading(self):
        super().set_xloading()
        if self.settings.aero is not None:
            import feniax.intrinsic.aero as aero

            approx = self.settings.aero.approx.capitalize()
            aeroobj = aero.Registry.create_instance(
                f"Aero{approx}", self.settings, self.sol
            )
            aeroobj.get_matrices()
            aeroobj.save_sol()

    def set_system(self):
        label = f"dq_{self.settings.label}"
        print(f"***** Setting intrinsinc static system with label {label}")
        self.dFq = getattr(dq_static, label)

    def solve(self):
        # label = self.settings.label.split("_")[-1]
        # solver_args = getattr(libargs, f"arg_{label}")
        # args1 = solver_args(self.sol, self.settings, self.fem, eta_0=self.eta0)

        self.qs = _staticSolve(
            self.eqsolver,
            self.dFq,
            self.settings.t,
            self.q0,
            self.args1,
            self.settings.solver_settings,
        )
        self.build_solution()

    def solve_forloop(self):
        label = self.settings.label.split("_")[-1]
        solver_args = getattr(libargs, f"arg_{label}")
        qs = [self.q0]
        for i, ti in enumerate(self.settings.t):
            args1 = solver_args(self.sol, self.settings, self.fem, ti, eta_0=self.eta0)
            sol = self.eqsolver(
                self.dFq, qs[-1], args1, **self.settings.solver_settings
            )
            qi = self.states_puller(sol)
            qs.append(qi)
        self.qs = jnp.array(qs[1:])

    def build_solution(self):
        q2_index = self.settings.states["q2"]
        q2 = self.qs[:, q2_index]
        tn = len(self.qs)
        X2, X3, ra, Cab = recover_staticfields(
            q2,
            tn,
            self.fem.X,
            self.sol.data.modes.phi2l,
            self.sol.data.modes.psi2l,
            self.sol.data.modes.X_xdelta,
            self.sol.data.modes.C0ab,
            self.config,
        )

        self.sol.add_container(
            "StaticSystem",
            label="_" + self.name,
            q=self.qs,
            X2=X2,
            X3=X3,
            Cab=Cab,
            ra=ra,
        )
        if self.settings.save:
            self.sol.save_container("StaticSystem", label="_" + self.name)

    def build_solution_loop(self):
        # q1 = qs[self.settings.q1_index, :]
        # q2 = qs[self.settings.q2_index, :]
        X2 = []
        X3 = []
        Cab = []
        ra = []
        for i, ti in enumerate(self.settings.t):
            X2t = postprocess.compute_internalforces_t(
                self.sol.data.modes.phi2l, self.qs[i]
            )
            X3t = postprocess.compute_strains_t(self.sol.data.modes.psi2l, self.qs[i])
            Cabt, rat = postprocess.integrate_strains(
                self.fem.X[0], jnp.eye(3), X3t, self.sol, self.fem
            )
            X2.append(X2t)
            X3.append(X3t)
            Cab.append(Cabt)
            ra.append(rat)

        self.sol.add_container(
            "StaticSystem",
            label="_" + self.name,
            q=self.qs,
            X2=jnp.array(X2),
            X3=jnp.array(X3),
            Cab=jnp.array(Cab),
            ra=jnp.array(ra),
        )
        if self.settings.save:
            self.sol.save_container("StaticSystem", label="_" + self.name)


class DynamicIntrinsic(IntrinsicSystem, cls_name="dynamic_intrinsic"):
    
    def set_xloading(self):
        super().set_xloading()
        if self.settings.aero is not None:
            import feniax.intrinsic.aero as aero

            approx = self.settings.aero.approx.capitalize()
            aeroobj = aero.Registry.create_instance(
                f"Aero{approx}", self.settings, self.sol
            )
            aeroobj.get_matrices()
            aeroobj.save_sol()
            if self.settings.aero.gust is not None:
                import feniax.intrinsic.gust as gust

                profile = self.settings.aero.gust_profile.capitalize()
                gustobj = gust.Registry.create_instance(
                    f"Gust{approx}{profile}", self.settings, self.sol
                )
                gustobj.calculate_normals()
                gustobj.calculate_downwash()
                gustobj.set_solution(self.sol, self.settings.name)

    def set_system(self):
        label = f"dq_{self.settings.label}"
        print(f"***** Setting intrinsinc Dynamic system with label {label}")
        self.dFq = getattr(dq_dynamic, label)

    def solve(self):
        # label = self.settings.label.split("_")[-1]
        # solver_args = getattr(libargs, f"arg_{label}")
        # args1 = solver_args(self.sol, self.settings, self.fem, eta_0=self.eta0)
        sol = self.eqsolver(
            self.dFq,
            self.args1,
            self.settings.solver_settings,
            q0=self.q0,
            t0=self.settings.t0,
            t1=self.settings.t1,
            tn=self.settings.tn,
            dt=self.settings.dt,
            t=self.settings.t,
        )
        self.qs = self.states_puller(sol)
        self.build_solution()

    def build_solution_loop(self):
        # return
        # q1 = qs[self.settings.q1_index, :]
        # q2 = qs[self.settings.q2_index, :]
        X2 = []
        X3 = []
        Cab = []
        ra = []
        for i, ti in enumerate(self.settings.t):
            X2t = postprocess.compute_internalforces(
                self.sol.data.modes.phi2l, self.qs[i]
            )
            X3t = postprocess.compute_strains(self.sol.data.modes.psi2l, self.qs[i])
            Cabt, rat = postprocess.integrate_strains(
                self.fem.X[0], jnp.eye(3), X3t, self.sol, self.fem
            )
            X2.append(X2t)
            X3.append(X3t)
            Cab.append(Cabt)
            ra.append(rat)

        self.sol.add_container(
            "DynamicSystem",
            label="_" + self.name,
            q=self.qs,
            X2=jnp.array(X2),
            X3=jnp.array(X3),
            Cab=jnp.array(Cab),
            ra=jnp.array(ra),
        )
        if self.settings.save:
            self.sol.save_container("DynamicSystem", label="_" + self.name)

    def build_solution(self):
        # q1_index = self.settings.states['q1']
        # q2_index = self.settings.states['q2']
        # q1 = self.qs[:, q1_index]
        # q2 = self.qs[:, q2_index]
        # tn = len(self.qs)
        # X1, X2, X3, ra, Cab = recover_fields(q1,
        #                                      q2,
        #                                      tn,
        #                                      self.fem.X,
        #                                      self.sol.data.modes.phi1l,
        #                                      self.sol.data.modes.phi2l,
        #                                      self.sol.data.modes.psi2l,
        #                                      self.sol.data.modes.X_xdelta,
        #                                      self.sol.data.modes.C0ab,
        #                                      self.config
        #                                      )

        # q1 = qs[self.settings.q1_index, :]
        # q2 = qs[self.settings.q2_index, :]
        X1 = postprocess.compute_velocities(
            self.sol.data.modes.phi1l, self.qs[:, self.settings.states["q1"]]
        )
        X2 = postprocess.compute_internalforces(
            self.sol.data.modes.phi2l, self.qs[:, self.settings.states["q2"]]
        )
        X3 = postprocess.compute_strains(
            self.sol.data.modes.psi2l, self.qs[:, self.settings.states["q2"]]
        )
        if self.settings.bc1.lower() == "clamped":
            tn = len(self.qs)
            ra0 = jnp.broadcast_to(self.fem.X[0], (tn, 3))
            Cab0 = jnp.broadcast_to(jnp.eye(3), (tn, 3, 3))
        else:
            if self.settings.rb_treatment == 1:
                ra_n0 = self.fem.X[0]
                Rab_n0 = jnp.eye(3)
                Cab0, ra0 = postprocess.integrate_node0(
                    X1[:, :, 0], self.settings.dt, ra_n0, Rab_n0
                )
        Cab, ra = postprocess.integrate_strains_t(
            ra0,
            Cab0,
            X3,
            self.sol.data.modes.X_xdelta,
            self.sol.data.modes.C0ab,
            self.config,
        )
        self.sol.add_container(
            "DynamicSystem",
            label="_" + self.name,
            q=self.qs,
            X1=X1,
            X2=X2,
            X3=X3,
            Cab=Cab,
            ra=ra,
            t=self.settings.t,
        )
        if self.settings.save:
            self.sol.save_container("DynamicSystem", label="_" + self.name)
