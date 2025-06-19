import jax.numpy as jnp
from feniax.preprocessor.configuration import Config
import feniax.intrinsic.objectives as objectives
import feniax.preprocessor.containers.intrinsicmodal as intrinsicmodal
import feniax.preprocessor.solution as solution
from feniax.foragers.forager import Forager
from feniax.systems.intrinsic_system import IntrinsicSystem
from abc import abstractmethod
from feniax.preprocessor.inputs import Inputs


class IntrinsicForager(Forager, cls_name="intrinsic_forager"):
    
    def __init__(self,
                 config: Config = None,
                 sol=None,
                 systems: dict[str, str] = None):
        
        self.config = config
        self.sol = sol
        self.systems = systems
        self.cforager = config.forager.settings        
        self.configs = []
        self.drivers = []

    @abstractmethod
    def _collect(self):
        ...

    @abstractmethod
    def _filter(self):
        ...

    @abstractmethod
    def _build(self):
        ...
        
    def build_configs(self):
        """ """

        self._collect()
        self._filter()
        self._build()

    def spawn(self):
        """ """
        from feniax.drivers.intrinsic_driver import IntrinsicDriver
        
        for config_i in self.configs:
            driver_i = IntrinsicDriver(config_i)
            driver_i.pre_simulation()
            driver_i.run_cases()
            driver_i.post_simulation()
            self.drivers.append(driver_i)

class IntrinsicForager_shard2adgust(IntrinsicForager,
                                    cls_name="forager_shard2adgust"):
    def __init__(
        self,
        config: Config = None,
        sol: solution.IntrinsicSolution = None,
        systems: dict[str, IntrinsicSystem] = None,
    ):
        super().__init__(config, sol, systems)
        self.filtered_indexes = set() # Indexes of interest from previous parallel simulation 
        self.gathersystem: IntrinsicSystem = None # The system
        self.field = None  # The field from previous simulation that is going to be analysed (X2 for instance)
        
    def _collect(self):
        """Collects the system1 results to
        build the field that is to be used in filtering
        """
        
        gathersystem_name = self.cforager.gathersystem_name
        objectivefield_name = self.cforager.ad['objective_var']
        self.gathersystem = self.systems[gathersystem_name]
        system1_sol = getattr(self.sol.data,
                              f"dynamicsystem_{gathersystem_name}")
        self.field = getattr(system1_sol, objectivefield_name)

    def _filter(self):
        """
        Filter the simulation index t
        """
        # TODO: Generalise
        nodes = self.cforager.ad['objective_args']['nodes']
        components = self.cforager.ad['objective_args']['components']
        for ni in nodes:
            for ci in components:
                field_i = self.field[:, :, ni, ci]
                index = jnp.unravel_index(jnp.argmax(field_i),
                                          field_i.shape)
                self.filtered_indexes.add(int(index[0]))

    def _build(self):
        """
        Builds configuration objects to launch new simulations
        """

        scattersystems_name = self.cforager.scattersystems_name
        for i, fi in enumerate(self.filtered_indexes):
            (rho_inf, u_inf,
             gust_length,
             gust_intensity) = self.gathersystem.xpoints[fi]
            
            add2config = Inputs()
            add2config.system.name = f"{self.config.system.name}_{scattersystems_name}{i}"
            add2config.system.ad = intrinsicmodal.DtoAD(**self.cforager.ad,
                                                        _numtime=len(self.config.system.t),
                                                        _numnodes=self.config.system._fem.num_nodes
                                                        )
            add2config.system.aero.rho_inf = rho_inf
            add2config.system.aero.u_inf = u_inf
            add2config.system.aero.gust.intensity = gust_intensity
            add2config.system.aero.gust.length = gust_length
            config = self.config.clone(add_attr=add2config,
                                       del_attr=['forager', 'system.shard', 'system.operationalmode'])
            #delattr(config.system, "shard")
            #config.system.delete_value("shard")
            #delattr(config, "forager")
            
            self.configs.append(config)

class IntrinsicMPIForager_shard2adgust(IntrinsicForager,
                                    cls_name="intrinsicMPI_shard2adgust"):
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
        objectivefield_name = self.cforager.ad['objective_var']
        self.gathersystem = self.systems[gathersystem_name]
        #TODO: loop through sols and setup fields as iterative
        system1_sol = getattr(self.sol.data,
                              f"dynamicsystem_{gathersystem_name}")
        self.field = getattr(system1_sol, objectivefield_name)

    def _filter(self):
        """
        Filter the
        """
        # TODO: Generalise
        nodes = self.cforager.ad['objective_args']['nodes']
        components = self.cforager.ad['objective_args']['components']
        for ni in nodes:
            for ci in components:
                #TODO: loop in fields and build tuple 
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
            # TODO pick config from tuple
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
            config.system.set_value("ad", intrinsicmodal.DtoAD(_numtime=len(self.gathersystem.t),
                                                               _numnodes=self.gathersystem._fem.num_nodes,
                                                    **self.cforager.ad
                                                    ))
            self.configs.append(config)
            
