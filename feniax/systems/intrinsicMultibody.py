import feniax.systems.intrinsic_system as intrinsic_system

# class DynamicIntrinsicMB(IntrinsicSystem, cls_name="dynamic_intrinsic"):
    
#     def set_xloading(self):
#         super().set_xloading()
#         if self.settings.aero is not None:
#             import feniax.intrinsic.aero as aero

#             approx = self.settings.aero.approx.capitalize()
#             aeroobj = aero.Registry.create_instance(
#                 f"Aero{approx}", self.settings, self.sol
#             )
#             aeroobj.get_matrices()
#             aeroobj.save_sol()
#             if self.settings.aero.gust is not None:
#                 import feniax.intrinsic.gust as gust

#                 profile = self.settings.aero.gust_profile.capitalize()
#                 gustobj = gust.Registry.create_instance(
#                     f"Gust{approx}{profile}", self.settings, self.sol
#                 )
#                 gustobj.calculate_normals()
#                 gustobj.calculate_downwash()
#                 gustobj.set_solution(self.sol, self.settings.name)

#     def set_system(self):
#         label = f"dq_{self.settings.label}"
#         logger.debug(f"Setting {self.__class__.__name__} with label {label}")                      
#         self.dFq = getattr(dq_dynamic, label)

#     def solve(self):

#         logger.info(f"Running System solution")
#         sol = self.eqsolver(
#             self.dFq,
#             self.args1,
#             self.settings.solver_settings,
#             q0=self.q0,
#             t0=self.settings.t0,
#             t1=self.settings.t1,
#             tn=self.settings.tn,
#             dt=self.settings.dt,
#             t=self.settings.t,
#         )
#         self.qs = self.states_puller(sol)
#         self.build_solution()

#     def build_solution_loop(self):
#         """
#         Deprecated function. Left it for info about other implementation
#         """

#         X2 = []
#         X3 = []
#         Cab = []
#         ra = []
#         for i, ti in enumerate(self.settings.t):
#             X2t = postprocess.compute_internalforces(
#                 self.sol.data.modes.phi2l, self.qs[i]
#             )
#             X3t = postprocess.compute_strains(self.sol.data.modes.psi2l, self.qs[i])
#             Cabt, rat = postprocess.integrate_strains(
#                 self.fem.X[0], jnp.eye(3), X3t, self.sol, self.fem
#             )
#             X2.append(X2t)
#             X3.append(X3t)
#             Cab.append(Cabt)
#             ra.append(rat)

#         self.sol.add_container(
#             "DynamicSystem",
#             label="_" + self.name,
#             q=self.qs,
#             X2=jnp.array(X2),
#             X3=jnp.array(X3),
#             Cab=jnp.array(Cab),
#             ra=jnp.array(ra),
#         )
#         if self.settings.save:
#             self.sol.save_container("DynamicSystem", label="_" + self.name)

#     def build_solution(self):
#         # q1_index = self.settings.states['q1']
#         # q2_index = self.settings.states['q2']
#         # q1 = self.qs[:, q1_index]
#         # q2 = self.qs[:, q2_index]
#         # tn = len(self.qs)
#         # X1, X2, X3, ra, Cab = recover_fields(q1,
#         #                                      q2,
#         #                                      tn,
#         #                                      self.fem.X,
#         #                                      self.sol.data.modes.phi1l,
#         #                                      self.sol.data.modes.phi2l,
#         #                                      self.sol.data.modes.psi2l,
#         #                                      self.sol.data.modes.X_xdelta,
#         #                                      self.sol.data.modes.C0ab,
#         #                                      self.config
#         #                                      )

#         # q1 = qs[self.settings.q1_index, :]
#         # q2 = qs[self.settings.q2_index, :]
#         logger.info(f"Building postprocessing fields (strains, velocities, positions, etc.)")        
#         X1 = postprocess.compute_velocities(
#             self.sol.data.modes.phi1l, self.qs[:, self.settings.states["q1"]]
#         )
#         X2 = postprocess.compute_internalforces(
#             self.sol.data.modes.phi2l, self.qs[:, self.settings.states["q2"]]
#         )
#         X3 = postprocess.compute_strains(
#             self.sol.data.modes.psi2l, self.qs[:, self.settings.states["q2"]]
#         )
#         if self.settings.bc1.lower() == "clamped":
#             tn = len(self.qs)
#             ra0 = jnp.broadcast_to(self.fem.X[0], (tn, 3))
#             Cab0 = jnp.broadcast_to(jnp.eye(3), (tn, 3, 3))
#         else:
#             if self.settings.rb_treatment == 1:
#                 ra_n0 = self.fem.X[0]
#                 Rab_n0 = jnp.eye(3)
#                 Cab0, ra0 = postprocess.integrate_node0(
#                     X1[:, :, 0], self.settings.dt, ra_n0, Rab_n0
#                 )
#         Cab, ra = postprocess.integrate_strains_t(
#             ra0,
#             Cab0,
#             X3,
#             self.sol.data.modes.X_xdelta,
#             self.sol.data.modes.C0ab,
#             self.config,
#         )
#         self.sol.add_container(
#             "DynamicSystem",
#             label="_" + self.name,
#             q=self.qs,
#             X1=X1,
#             X2=X2,
#             X3=X3,
#             Cab=Cab,
#             ra=ra,
#             t=self.settings.t,
#         )
#         if self.settings.save:
#             self.sol.save_container("DynamicSystem", label="_" + self.name)
