import jax.numpy as jnp
from feniax.drivers.intrinsic_driver import IntrinsicDriver
from feniax.preprocessor.configuration import Config
import feniax.intrinsic.objectives as objectives
import feniax.preprocessor.containers.intrinsicmodal as intrinsicmodal
import feniax.preprocessor.solution as solution
from feniax.foragers.forager import Forager
from feniax.systems.intrinsic_system import IntrinsicSystem

class IntrinsicForager(Forager, cls_name="intrinsic_forager"):
    
    def __init__(self,
                 config: Config = None,
                 sol=None,
                 systems: dict[str, str] = None):
        
        self.config = config
        self.cforager = config.forager
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

class IntrinsicForager_shard2adgust(IntrinsicForager,
                                    cls_name="intrinsic_shard2adgust"):
    def __init__(
        self,
        config: Config = None,
        sol: solution.IntrinsicSolution = None,
        systems: dict[str, IntrinsicSystem] = None,
    ):
        super().__init__(config, sol, systems)
        self.filtered_indexes = set()
        self.gathersystem:IntrinsicSystem = None
        self.field = None
        
    def _collect(self):
        """Collects the system1 results to
        build the field that is to be used in filtering
        """
        
        gathersystem_name = self.cforager.gathersystem_name
        objectivefield_name = self.cforager.ad.objective_var
        self.gathersystem = self.systems[gathersystem_name]
        system1_sol = getattr(self.sol.data,
                              f"dynamicsystem_{gathersystem_name}")
        self.field = getattr(system1_sol, objectivefield_name)

    def _filter(self):
        """
        Filter the
        """
        # TODO: Generalise
        nodes = self.cforager.ad.objectiveArgs.nodes
        components = self.cforager.ad.objectiveArgs.components
        for ni in nodes:
            for ci in components:
                field_i = self.field[:, :, ni, ci]
                index = jnp.unravel_index(jnp.argmax(field_i),
                                          field_i.shape)
                self.filtered_indexes.add(index[0])

    def _build(self):
        """
        Builds configuration objects to launch new simulations
        """
        scattersystems_name = self.cforager.scattersystems_name
        for i, fi in enumerate(self.filtered_indexes):
            config = self.config.clone()
            delattr(config, "shard")
            delattr(config, "forager")
            (rho_inf, u_inf,
             gust_length,
             gust_intensity) = self.gathersystem.xpoints[fi]
            config.system.set_value("name",
            f"{config.system.name}_{scattersystems_name}{i}")
            config.system.aero.set_value("rho_inf",
                                         rho_inf)
            config.system.aero.set_value("u_inf",
                                         u_inf)
            config.system.aero.gust.set_value("intensity",
                                              gust_intensity)
            config.system.aero.gust.set_value("length",
                                              gust_length)
            config.system.ad = self.cforager.ad
            self.configs.append(config)
