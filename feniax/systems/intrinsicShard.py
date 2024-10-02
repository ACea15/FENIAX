import itertools

import feniax.intrinsic.dynamicShard as dynamic_shard
import feniax.intrinsic.staticShard as static_shard
import feniax.intrinsic.xloads as xloads
from feniax.systems.intrinsic_system import IntrinsicSystem
import feniax.intrinsic.argshard as argshard

import jax
from jax.sharding import NamedSharding
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils


class IntrinsicShardSystem(IntrinsicSystem, cls_name="Shard_intrinsic"):

    def set_args(self):
        label = self.settings.label.split("_")[-1]
        solver_args = getattr(argshard, f"arg_{label}")
        self.args1 = solver_args(self.sol, self.settings, self.fem, eta_0=self.eta0)


    def set_xloading(self):

        num_nodes = self.fem.num_nodes
        C06ab = self.sol.data.modes.C06ab
        if self.settings.shard.input_type.lower() == "pointforces":
            super().set_xloading()
            if self.settings.shard.inputs.follower_points is not None:
                self.xpoints = xloads.shard_point_follower(self.settings.xloads.x,
                                                           self.settings.shard.inputs.follower_points,
                                                           self.settings.shard.inputs.follower_interpolation,
                                                           num_nodes,
                                                           C06ab)
            elif self.settings.shard.inputs.dead_points is not None:
                self.xpoints = xloads.shard_point_dead(self.settings.xloads.x,
                                                           self.settings.shard.inputs.dead_points,
                                                           self.settings.shard.inputs.dead_interpolation,
                                                       num_nodes)
            elif self.settings.shard.inputs.gravity is not None:
                self.xpoints = xloads.shard_gravity(self.settings.xloads.x,
                                                    self.settings.shard.inputs.gravity,
                                                    self.settings.shard.inputs.gravity_vect,
                                                    self.fem.Ma,
                                                    self.fem.Mfe_order)
            
        elif self.settings.shard.input_type.lower() == "gust1":
            super().set_xloading()
            self.xpoints = xloads.shard_gust(self.settings.shard.inputs)
        
        
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

class StaticShardIntrinsic(IntrinsicShardSystem, cls_name="staticShard_intrinsic"):
    
    def set_system(self):
        label_sys = self.settings.label
        label_shard = self.settings.shard.label
        self.label = f"main_{label_sys}_{label_shard}"
        print(f"***** Setting intrinsic static shard system with label {self.label}")
        self.main = getattr(static_shard, self.label)

    def build_solution(self, jac, objective, q, X2, X3, ra, Cab, *args, **kwargs):
        
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
        print(f"***** Setting intrinsic dynamic shard system with label {label}")
        self.main = getattr(dynamic_shard, label)


    def build_solution(self, q, X1, X2, X3, ra, Cab, *args, **kwargs):
        
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
        
