from feniax.systems.system import System
import feniax.systems.sollibs as sollibs
import feniax.intrinsic.dynamicADShard as dynamicADShard
import feniax.intrinsic.staticADShard as staticADShard
import feniax.intrinsic.dynamicAD as dynamic_ad
import feniax.intrinsic.postprocess as postprocess
import feniax.preprocessor.containers.intrinsicmodal as intrinsicmodal
import feniax.preprocessor.solution as solution
import feniax.intrinsic.initcond as initcond
import feniax.intrinsic.args as libargs
import feniax.intrinsic.objectives as objectives
import feniax.systems.intrinsicAD as intrinsicAD
import feniax.systems.intrinsicShard as intrinsicShard
import feniax.systems.intrinsic_system as intrinsic_system
from jax.sharding import NamedSharding
from jax.sharding import Mesh, PartitionSpec as P

from functools import partial
import jax
import jax.numpy as jnp
import logging
from feniax.ulogger.setup import get_logger

logger = get_logger(__name__)


class IntrinsicADShardSystem(intrinsic_system.IntrinsicSystem, cls_name="ADShard_intrinsic"):

    def set_xloading(self):

        intrinsicShard.IntrinsicShardSystem.set_xloading(self)

    # def solve(self, *args, **kwargs):

    #     super().solve(static_argnames=["config", "f_obj", "obj_args", "mesh"], mesh=self.mesh) # call IntrinsicADSystem.solve

    def solve(self, static_argnames=["config", "f_obj", "obj_args", "mesh"], *args, **kwargs):
        
        logger.info(f"Running System solution")
        # TODO: option to jit at the end, jit dFq, or not jit at all.
        if True: # not working with diffrax static solver
            fprime = partial(jax.jit, static_argnames=static_argnames)(self.eqsolver(self.dFq, has_aux=False))  # call to jax.grad..etc
        elif False:
            fprime = (self.eqsolver(partial(jax.jit,
                                            static_argnames=static_argnames)(self.dFq),
                                    has_aux=True))  # call to jax.grad..etc
        else:
            fprime = self.eqsolver(self.dFq, has_aux=True)
        if self.settings.ad.grad_type == "value_grad":
            ((val, fout), jac) = fprime(
                self.settings.ad.inputs,
                self.xpoints,
                mesh=self.mesh,
                q0=self.q0,
                config=self.config,
                f_obj=self.f_obj,
                obj_args=self.settings.ad.objective_args,
                *args,
                **kwargs
            )
        else:
            xshard = jax.device_put(self.xpoints, NamedSharding(self.mesh, P('x')))
            jac, fout = fprime(
                self.settings.ad.inputs,
                xshard,
                #self.xpoints,
                mesh=self.mesh,
                q0=self.q0,
                config=self.config,
                f_obj=self.f_obj,
                obj_args=self.settings.ad.objective_args,
                *args,
                **kwargs                
            )
        #import pdb; pdb.set_trace()
        #print(jac)
        #print(self.f_obj)
        self.build_solution(jac, self.f_obj, **fout)
        
class StaticADShardIntrinsic(IntrinsicADShardSystem,
                             intrinsicAD.StaticADIntrinsic,
                             intrinsicShard.IntrinsicShardSystem,
                             cls_name="staticADShard_intrinsic"):

    
    def set_system(self):
        label_sys = self.settings.label
        label_ad = self.settings.ad.label
        label_shard = self.settings.shard.label        
        label = f"main_{label_sys}_{label_ad}_{label_shard}"
        logger.debug(f"Setting {self.__class__.__name__} with label {label}")
        self.dFq = getattr(staticADShard, label)

class DynamicADShardIntrinsic(IntrinsicADShardSystem,
                              intrinsicAD.DynamicADIntrinsic,
                              intrinsicShard.IntrinsicShardSystem,
                              cls_name="dynamicADShard_intrinsic"):
    
    def set_system(self):
        label_sys = self.settings.label
        label_ad = self.settings.ad.label
        label_shard = self.settings.shard.label        
        label = f"main_{label_sys}_{label_ad}_{label_shard}"
        logger.debug(f"Setting {self.__class__.__name__} with label {label}")
        self.dFq = getattr(dynamicADShard, label)
