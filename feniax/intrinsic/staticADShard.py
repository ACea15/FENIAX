import jax.numpy as jnp
import jax
from functools import partial
import feniax.systems.sollibs as sollibs
import feniax.intrinsic.ad_common as adcommon
import feniax.intrinsic.gust as igust
import feniax.intrinsic.couplings as couplings
import feniax.intrinsic.dq_dynamic as dq_dynamic
import feniax.systems.intrinsic_system as isys
import feniax.intrinsic.dynamicShard as dynamicShard
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P


def main_10g11_1_1(
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
    Ma = inputs_ad["Ka"]
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
    (A,
     D,
     c_ref,
     poles,
     num_poles,
     xgust
     ) = adcommon._get_aerogust(config)
        
    args1 = (output_modes['phi1l'],
             output_modes['phi2l'],
             output_modes['psi2l'],
             output_modes['X_xdelta'],
             output_modes['C0ab'],
             A,
             D,
             c_ref,
            (
                eta_0,
                output_modes['gamma1'],
                output_modes['gamma2'],
                output_modes['omega'],
                states,
                poles,        
                num_modes,
                num_poles,
                xgust,
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

    @partial(shard_map, mesh=mesh, in_specs=P('x'), out_specs=P())
    def _fshard(inputs):

        sol_dict = dynamicShard.main_20g21_3(inputs,
                                             q0=q0,
                                             config=config,
                                             args=args1)

        X2filter = sol_dict['X2'][jnp.ix_(jnp.array([0]), jnp.array(obj_args.t), jnp.array(obj_args.components), jnp.array(obj_args.nodes))]
        X2_max = jnp.max(X2filter, axis=1)
        return jax.lax.pmean(X2_max, axis_name="x")

        # output = adcommon._objective_output(
        #     **sol_dict,
        #     f_obj=f_obj,
        #     nodes=jnp.array(obj_args.nodes),
        #     components=jnp.array(obj_args.components),
        #     t=jnp.array(obj_args.t),
        #     axis=obj_args.axis,
        # )

        # return output        

    obj, f_out = _fshard(inputs_shard)
    return (obj, f_out)
