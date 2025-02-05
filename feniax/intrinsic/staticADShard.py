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
