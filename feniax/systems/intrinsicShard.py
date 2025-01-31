import itertools

import feniax.intrinsic.dynamicShard as dynamicShard
import feniax.intrinsic.staticShard as staticShard
import feniax.intrinsic.xloads as xloads
from feniax.systems.intrinsic_system import IntrinsicSystem
import feniax.intrinsic.argshard as argshard
import logging
import jax
from jax.sharding import NamedSharding
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from feniax.ulogger.setup import get_logger
from functools import partial
from jax.experimental.shard_map import shard_map

logger = get_logger(__name__)

class IntrinsicShardSystem(IntrinsicSystem, cls_name="Shard_intrinsic"):

    def set_args(self):
        label = self.settings.label.split("_")[-1]
        solver_args = getattr(argshard, f"arg_{label}")
        self.args1 = solver_args(self.sol, self.settings, self.fem, eta_0=self.eta0)

    def _set_gust1(self):
        
        super().set_xloading()
        self.xpoints = xloads.shard_gust1(self.settings.shard.inputs,
                                          self.settings.aero)
        
    def _set_pointforces(self):

        num_nodes = self.fem.num_nodes
        C06ab = self.sol.data.modes.C06ab        
        if self.settings.shard.inputs.follower_points is not None:
            super().set_xloading(compute_follower=False)
            self.xpoints = xloads.shard_point_follower(self.settings.xloads.x,
                                                       self.settings.shard.inputs.follower_points,
                                                       self.settings.shard.inputs.follower_interpolation,
                                                       num_nodes,
                                                       C06ab)
        elif self.settings.shard.inputs.dead_points is not None:
            super().set_xloading(compute_dead=False)
            self.xpoints = xloads.shard_point_dead(self.settings.xloads.x,
                                                   self.settings.shard.inputs.dead_points,
                                                   self.settings.shard.inputs.dead_interpolation,
                                                   num_nodes)
        elif self.settings.shard.inputs.gravity is not None:
            super().set_xloading(compute_gravity=False)
            self.xpoints = xloads.shard_gravity(self.settings.xloads.x,
                                                self.settings.shard.inputs.gravity,
                                                self.settings.shard.inputs.gravity_vect,
                                                self.fem.Ma,
                                                self.fem.Mfe_order)

    def _set_steadyalpha(self):
        
        super().set_xloading()
        self.xpoints = xloads.shard_steadyalpha(self.settings.shard.inputs,
                                                self.settings.aero)
            
    def set_xloading(self):

        shard_type = self.settings.shard.input_type.lower()
        f_shard = getattr(self, f"_set_{shard_type}")
        f_shard()
        
    def solve(self):
        
        # Create a Sharding object to distribute a value across devices:
        mesh = Mesh(devices=mesh_utils.create_device_mesh((jax.device_count(),)),
                    axis_names=('x'))
        xshard = jax.device_put(self.xpoints, NamedSharding(mesh, P('x')))

        results = self.main(xshard,
                            q0=self.q0,
                            config=self.config,
                            args=self.args1
                            #eqsolver=self.eqsolver
                            )
        
        self.build_solution(**results)

    def build_solution(self):
        self.sol.add_container(
            "Shards",
            label="_" + self.name,
            points=self.xpoints,
            device_count=jax.device_count(),
            local_device_count=jax.local_device_count()
        )
        if self.settings.save:
            self.sol.save_container("Shards", label="_" + self.name)
        
class StaticShardIntrinsic(IntrinsicShardSystem, cls_name="staticShard_intrinsic"):
    
    def set_system(self):
        label_sys = self.settings.label
        label_shard = self.settings.shard.label
        self.label = f"main_{label_sys}_{label_shard}"
        logger.info(f"Setting {self.__class__.__name__} with label {label}")
        self.main = partial(jax.jit, static_argnames=["config"])(getattr(staticShard, self.label))

    def build_solution(self, q, X2, X3, ra, Cab, *args, **kwargs):
        
        super().build_solution()
        self.sol.add_container(
            "StaticSystem",
            label="_" + self.name,
            q=q,
            X2=X2,
            X3=X3,
            Cab=Cab,
            ra=ra,
            t=self.settings.t,
        )
        if self.settings.save:
            self.sol.save_container("StaticSystem", label="_" + self.name)


class DynamicShardIntrinsic(IntrinsicShardSystem, cls_name="dynamicShard_intrinsic"):
    
    def set_system(self):
        label_sys = self.settings.label
        label_shard = self.settings.shard.label
        label = f"main_{label_sys}_{label_shard}"
        logger.info(f"Setting {self.__class__.__name__} with label {label}")
        self.main = partial(jax.jit, static_argnames=["config"])(getattr(dynamicShard, label))

    def build_solution(self, q, X1, X2, X3, ra, Cab, *args, **kwargs):
        
        super().build_solution()
        self.sol.add_container(
            "DynamicSystem",
            label="_" + self.name,
            q=q,
            X1=X1,
            X2=X2,
            X3=X3,
            Cab=Cab,
            ra=ra,
            t=self.settings.t,
        )
        if self.settings.save:
            self.sol.save_container("DynamicSystem", label="_" + self.name)

class StaticShardMapIntrinsic(IntrinsicShardSystem, cls_name="staticShardmap_intrinsic"):

    def set_system(self):
        label_sys = self.settings.label
        label_shard = self.settings.shard.label
        label = f"main_{label_sys}_{label_shard}"
        logger.info(f"Setting {self.__class__.__name__} with label {label}")
        self.mesh = Mesh(devices=mesh_utils.create_device_mesh((jax.device_count(),)),
                         axis_names=('x'))
        f = getattr(staticShard, label)
        self.main = partial(shard_map, mesh=self.mesh, in_specs=P('x'), out_specs=P('x'))(partial(f,
                                                                                                  q0=self.q0,
                                                                                                  config=self.config,
                                                                                                  args=self.args1
                                                                                                  )
                                                                                          )
    
    def set_system(self):
        label_sys = self.settings.label
        label_shard = self.settings.shard.label
        self.label = f"main_{label_sys}_{label_shard}"
        logger.info(f"Setting {self.__class__.__name__} with label {self.label}")
        self.mesh = Mesh(devices=mesh_utils.create_device_mesh((jax.device_count(),)),
                         axis_names=('x'))
        f = getattr(staticShard, self.label)
        self.main = shard_map(f, mesh=self.mesh, in_specs=P('x'), out_specs=P('x'))

    def solve(self):
        
        xshard = jax.device_put(self.xpoints, NamedSharding(self.mesh, P('x')))

        results = self.main(xshard,
                            #q0=self.q0,
                            #config=self.config,
                            #args=self.args1
                            #eqsolver=self.eqsolver
                            )
        
        self.build_solution(**results)
        
    def build_solution(self, q, X2, X3, ra, Cab, *args, **kwargs):
        
        super().build_solution()
        self.sol.add_container(
            "StaticSystem",
            label="_" + self.name,
            q=q,
            X2=X2,
            X3=X3,
            Cab=Cab,
            ra=ra,
            t=self.settings.t,
        )
        if self.settings.save:
            self.sol.save_container("StaticSystem", label="_" + self.name)


class DynamicShardMapIntrinsic(IntrinsicShardSystem, cls_name="dynamicShardmap_intrinsic"):
    
    def set_system(self):
        label_sys = self.settings.label
        label_shard = self.settings.shard.label
        label = f"main_{label_sys}_{label_shard}"
        logger.info(f"Setting {self.__class__.__name__} with label {label}")
        self.mesh = Mesh(devices=mesh_utils.create_device_mesh((jax.device_count(),)),
                         axis_names=('x'))
        f = getattr(dynamicShard, label)
        self.main = partial(shard_map, mesh=self.mesh, in_specs=P('x'), out_specs=P('x'))(partial(f,
                                                                                                  q0=self.q0,
                                                                                                  config=self.config,
                                                                                                  args=self.args1
                                                                                                  )
                                                                                          )

    def solve(self):
        
        xshard = jax.device_put(self.xpoints, NamedSharding(self.mesh, P('x')))

        results = self.main(xshard,
                            #q0=self.q0,
                            #config=self.config,
                            #args=self.args1
                            #eqsolver=self.eqsolver
                            )
        
        self.build_solution(**results)
        
    def build_solution(self, q, X1, X2, X3, ra, Cab, *args, **kwargs):
        
        super().build_solution()
        self.sol.add_container(
            "DynamicSystem",
            label="_" + self.name,
            q=q,
            X1=X1,
            X2=X2,
            X3=X3,
            Cab=Cab,
            ra=ra,
            t=self.settings.t,
        )
        if self.settings.save:
            self.sol.save_container("DynamicSystem", label="_" + self.name)
            
            
