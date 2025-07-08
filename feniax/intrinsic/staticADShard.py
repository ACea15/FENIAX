import jax.numpy as jnp
import jax
from functools import partial
import feniax.systems.sollibs as sollibs
import feniax.intrinsic.ad_common as adcommon
import feniax.intrinsic.gust as igust
import feniax.intrinsic.couplings as couplings
import feniax.intrinsic.dq_dynamic as dq_dynamic
import feniax.systems.intrinsic_system as isys
import feniax.intrinsic.staticShard as staticShard
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P


def main_10g11_3_1(
        inputs_ad,
        inputs_shard,
        q0,
        config,
        f_obj,
        obj_args,
        mesh,
        *args,
        **kwargs,
):
    Ka = inputs_ad["Ka"]
    Ma = inputs_ad["Ma"]
    eigenvals = inputs_ad[
        "eigenvals"
    ]  # jnp.load(config.fem.folder / config.fem.eig_names[0])
    eigenvecs = inputs_ad[
        "eigenvecs"
    ]  # jnp.load(config.fem.folder / config.fem.eig_names[1])

    output_modes, input_dict = adcommon._build_intrinsic(adcommon._get_inputs, config,
                                                   Ka=Ka,
                                                   Ma=Ma,
                                                   eigenvals=eigenvals,
                                                   eigenvecs=eigenvecs)
    config.system.build_states(config.fem.num_modes, config.fem.num_nodes)
    states = config.system.states
    num_modes = config.fem.num_modes
    eta_0 = jnp.zeros(num_modes)
        
    args1 = (
             output_modes['phi2l'],
             output_modes['psi2l'],
             output_modes['X_xdelta'],
             output_modes['C0ab'],
            (
                eta_0,
                output_modes['gamma2'],
                output_modes['omega'],
                output_modes['phi1l'],
                config.system.xloads.x
            )
            )
    
    # main_shard = partial(shard_map, mesh=mesh, in_specs=P('x'),
    #                      out_specs=P('x')
    #                      )(partial(dynamicShard.main_20g21_3,
    #                                q0=q0,
    #                                config=config,
    #                                args=args1
    #                                )
    #                        )

    # @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=(P(), P('x')))
    def _fshard(inputs):
        #jax.debug.breakpoint()
        sol_dict = staticShard.main_10g11_1(inputs,
                                            q0=q0,
                                            config=config,
                                            args=args1)

        #X2filter = sol_dict['X2'][jnp.ix_(jnp.array([0]), jnp.array(obj_args.t), jnp.array(obj_args.components), jnp.array(obj_args.nodes))]
        #X2_max = jnp.max(X2filter, axis=1)
        #ra = sol_dict['ra'][:,-1, 2, 35]
        #ra = sol_dict['ra'][-1, 2, 35]
        #sol_out = sol_dict | dict(objective=ra)
        #return jax.lax.pmean(ra, axis_name="x"), sol_out #, sol_dict #jnp.mean(ra, axis=0), sol_out #jax.lax.pmean(ra, axis_name="x"), sol_out

        output = adcommon._objective_output(
            sol_dict,
            f_obj=f_obj,
            nodes=jnp.array(obj_args.nodes),
            components=jnp.array(obj_args.components),
            t=jnp.array(obj_args.t),
            axis=obj_args.axis,
        )

        return output
        
    # Number of devices (e.g., 4 if you have 4 GPUs)
    num_devices = jax.device_count()
    inputs_shape = inputs_shard.shape
    batch_per_device = inputs_shape[0] // num_devices
    
    # Reshape into [num_devices, batch_per_device]
    inputs_shard_reshaped = inputs_shard.reshape((num_devices, batch_per_device) + inputs_shape[1:])
    
    #obj, f_out = _fshard(inputs_shard)
    obj0, f_out0  = jax.pmap(_fshard, axis_name='x')(inputs_shard_reshaped)
    #TODO: generalise
    obj = jnp.mean(obj0[0])
    f_out = jax.tree.map(lambda xin: xin.reshape((inputs_shape[0],) + xin.shape[2:]), f_out0)
    #return (obj[0], 0)
    return (obj, f_out)
