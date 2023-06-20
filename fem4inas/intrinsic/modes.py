import jax
import jax.numpy as jnp
import fem4inas.preprocessor.containers.intrinsicmodal.Dfem as Dfem

def shapes(X: jnp.ndarray, Ka: jnp.ndarray, Ma: jnp.ndarray,
           eigenval: jnp.ndarray, eigenvec: jnp.ndarray, fem: Dfem):

    component_force = {k : jnp.zeros(6) for k in fem.components}
    # Nodal force in global frame [Compute overall inbalance of the force]
    #=====================================================================
    nodal_force = jnp.dot(Ma, eigenvec) * eigenval

    for i in range(fem.num_nodes):
        component_force[fem.node_component] += nodal_force[6 * i: 6 * i + 6]
        component_force[fem.node_component].at[3: ].set(component_force[fem.node_component][3:]
                                                        + )

    for i in range(fem.num_nodes):
    
