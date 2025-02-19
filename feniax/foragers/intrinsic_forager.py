import jax.numpy as jnp
from feniax.drivers.intrinsic_driver import IntrinsicDriver
from feniax.preprocessor.configuration import Config
import feniax.intrinsic.objectives as objectives
import feniax.preprocessor.containers.intrinsicmodal as intrinsicmodal
import feniax.preprocessor.solution as solution
from feniax.foragers.forager import Forager


class IntrinsicForager(Forager, cls_name="intrinsic_forager"):
    def __init__(self, config: Config = None, sol=None, systems: dict[str, str] = None):
        self.config = config
        self.settings = config.forager
        self.sol = sol
        self.configs = []
        self.drivers = []

    def build_configs(self):
        """ """

        self._collect()
        self._filter()
        self._build()

    def spawn(self):
        """ """

        for config_i in self.configs:
            driver_i = IntrinsicDriver(config_i)
            driver_i.pre_simulation()
            driver_i.run_cases()
            driver_i.post_simulation()
            self.drivers.append(driver_i)


class IntrinsicForager_shard2adgust(IntrinsicForager, cls_name="forager_shard2adgust"):
    def __init__(
        self,
        config: Config = None,
        sol: solution.IntrinsicSolution = None,
        systems: dict[str, str] = None,
    ):
        super().__init__(config, sol, systems)
        self.filtered_indexes = set()
        self.system1 = None
        self.field = None

    def _collect(self):
        """Collects, the run system1,
        system to be created inputs, the field to be used filter"""

        system1_name = self.settings.system_in_name
        self.system1 = self.systems[system1_name]
        self.system2_inputs = self.settings.system2_inputs
        system1_sol = getattr(self.sol.data,
                              f"DynamicSystem{system1_name}")
        self.field = getattr(system1_sol,
                             self.settings.field_name)

    def _filter(self):
        # obj_label = f"{self.settings.ad.objective_var}_{self.settings.ad.objective_fun.upper()}"
        # fobj = objectives.factory(obj_label)

        # fobj(X1=X1, X2=X2, X3=X3, ra=ra, Cab=Cab)
        nodes = self.system2_inputs["objectiveArgs"]["nodes"]
        components = self.system2_inputs["objectiveArgs"]["components"]
        for ni in nodes:
            for ci in components:
                # entries = jnp.ix_(jnp.arange(len(self.field)),
                #                   self.system2_inputs.objectiveArgs.components,
                #                   self.system2_inputs.objectiveArgs.nodes)
                field_i = self.field[:, :, ni, ci]
                index = jnp.unravel_index(jnp.argmax(field_i), field_i.shape)
                self.filtered_indexes.add(index[0])

    def _build(self):
        for i, fi in enumerate(self.filtered_indexes):
            config = self.config.clone()
            delattr(config, "shard")
            delattr(config, "forager")
            (rho_inf, u_inf, gust_length, gust_intensity) = self.system1.xpoints[fi]
            config.system.set_value("name", f"{config.system.name}_{i}")
            config.system.aero.set_value("rho_inf", rho_inf)
            config.system.aero.set_value("u_inf", u_inf)
            config.system.aero.gust.set_value("intensity", gust_intensity)
            config.system.ad = intrinsicmodal.DtoAD(**self.system2_inputs["ad"])
            config.system.aero.gust.set_value("length", gust_length)
            self.configs.append(config)
