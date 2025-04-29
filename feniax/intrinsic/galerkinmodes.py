"""
Preprocessor to intrinsic modal system.

Calls computation of intrinsic modes and modal couplings
"""

from typing import Protocol

from feniax.ulogger.setup import get_logger
from feniax.preprocessor import solution, configuration
import feniax.intrinsic.modes as modes
import feniax.intrinsic.couplings as couplings

logger = get_logger(__name__)

EIG_FUNCS = dict(
    scipy=modes.compute_eigs_scipy,
    jax_custom=modes.compute_eigs,
    inputs=modes.compute_eigs_load,
    input_memory=modes.compute_eigs_pass,
)

class GalerkinInterface(Protocol):
    def compute(self) -> None:
        ...


class Galerkin:
    """Performs a Galerkin projection of the intrinsic nonlinear equations

    Computes instrinsic modes and nonlinear modal couplings

    Parameters
    ----------

    """
    
    
    def __init__(self, config: configuration.Config, sol: solution.IntrinsicSolution, label=""):
        """

        Parameters
        ----------
        config : config.Config
            Configuration object

        """

        self.config = config
        self.sol = sol
        self.label = label
        
    def compute(self):
        
        if not self.config.driver.ad_on and not self.config.driver.fast_on:
            if self.config.driver.compute_fem:
                self._compute_modalshapes()
                self._compute_modalcouplings()
                if self.config.driver.save_fem:
                    self.sol.save_container(f"Modes{self.label}")
                    self.sol.save_container(f"Couplings{self.label}")
            else:
                self._load_modalshapes()
                self._load_modalcouplings()

    def _compute_eigs(self):

        eig_type = self.config.fem.eig_type
        eig_solver = EIG_FUNCS[eig_type]
        eigenvals, eigenvecs = eig_solver(
            Ka=self.config.fem.Ka,
            Ma=self.config.fem.Ma,
            num_modes=self.config.fem.num_modes,
            path=self.config.fem.folder,
            eig_names=self.config.fem.eig_names,
            eigenvals=self.config.fem.eigenvals,
            eigenvecs=self.config.fem.eigenvecs,
        )
        logger.debug(f"Computing eigenvalue problem from {eig_type}")
        return eigenvals, eigenvecs

    def _compute_modalshapes(self):
        eigenvals, eigenvecs = self._compute_eigs()
        if self.config.fem.constrainedDoF and False:
            modal_analysis = modes.shapes(
                self.config.fem.X.T,
                self.config.fem.Ka0s,
                self.config.fem.Ma0s,
                eigenvals,
                eigenvecs,
                self.config,
            )
        else:
            modal_analysis = modes.shapes(
                self.config.fem.X.T,
                self.config.fem.Ka,
                self.config.fem.Ma,
                eigenvals,
                eigenvecs,
                self.config,
            )

        modal_analysis_scaled = modes.scale(*modal_analysis)
        self.sol.add_container(f"Modes{self.label}", *modal_analysis_scaled)

    def _compute_modalcouplings(self):
        # if self._config.numlib == "jax":

        # elif self._config.numlib == "numpy":
        #    import feniax.intrinsic.couplings_np as couplings
        alpha1, alpha2 = modes.check_alphas(
            self.sol.data.modes.phi1,
            self.sol.data.modes.psi1,
            self.sol.data.modes.phi2l,
            self.sol.data.modes.psi2l,
            self.sol.data.modes.X_xdelta,
            tolerance=self.config.jax_np.allclose,
        )
        gamma1 = couplings.f_gamma1(self.sol.data.modes.phi1, self.sol.data.modes.psi1)
        gamma2 = couplings.f_gamma2(
            self.sol.data.modes.phi1ml,
            self.sol.data.modes.phi2l,
            self.sol.data.modes.psi2l,
            self.sol.data.modes.X_xdelta,
        )

        self.sol.add_container(f"Couplings{self.label}", alpha1, alpha2, gamma1, gamma2)

    def _load_modalshapes(self):
        self.sol.load_container(f"Modes{self.label}")

    def _load_modalcouplings(self):
        self.sol.load_container(f"Couplings{self.label}")

class GalerkinMultibody:
    """Performs a Galerkin projection of the intrinsic nonlinear equations

    Computes instrinsic modes and nonlinear modal couplings for multiple bodies

    Parameters
    ----------

    """
    
    
    def __init__(self, config: configuration.Config, sol: solution.IntrinsicSolution, labels=""):
        """

        Parameters
        ----------
        config : config.Config
            Configuration object

        """

        self.config = config
        self.sol = sol
        self.labels = labels
        self.Ka = None
        self.Ma = None
        self.Ka0s = None  # FE matrices expanded with 0s from removed DoF
        self.Ma0s = None        
        self.X = None
        self.eig_type = None
        self.num_modes = None
        self.eig_names = None
        self.eigenvals = None
        self.eigenvecs = None
        self.path = None

    def set_label(self, label):
        self.label = label
        
    def set_fe(self):
        
        if self.label == "b0":
            self.eig_type = self.config.fem.eig_type
            self.Ka=self.config.fem.Ka,
            self.Ma=self.config.fem.Ma,
            self.Ka0s=self.config.fem.Ka0s,
            self.Ma0s=self.config.fem.Ma0s,            
            self.X = self.config.fem.X
            self.num_modes=self.config.fem.num_modes,
            self.path=self.config.fem.folder,
            self.eig_names=self.config.fem.eig_names,
            self.eigenvals=self.config.fem.eigenvals,
            self.eigenvecs=self.config.fem.eigenvecs,
        else:
            self.eig_type = self.config.multibody.fems[self.label].eig_type
            self.Ka=self.config.multibody.fems[self.label].Ka
            self.Ma=self.config.multibody.fems[self.label].Ma
            self.Ka0s=self.config.multibody.fems[self.label].Ka0s
            self.Ma0s=self.config.multibody.fems[self.label].Ma0s            
            self.num_modes=self.config.multibody.fems[self.label].num_modes
            self.path=self.config.multibody.fems[self.label].folder
            self.eig_names=self.config.multibody.fems[self.label].eig_names
            self.eigenvals=self.config.multibody.fems[self.label].eigenvals
            self.eigenvecs=self.config.multibody.fems[self.label].eigenvecs
        
    def compute(self):
        
        if not self.config.driver.ad_on and not self.config.driver.fast_on:
            for li in self.labels:
                self.set_label(li)
                if self.config.driver.compute_fem:
                    self.set_fe()
                    self._compute_modalshapes()
                    self._compute_modalcouplings()
                    if self.config.driver.save_fem:
                        self.sol.save_container(f"Modes{self.label}")
                        self.sol.save_container(f"Couplings{self.label}")
                else:
                    self._load_modalshapes()
                    self._load_modalcouplings()

    def _compute_eigs(self):

        eig_type = self.eig_type
        eig_solver = EIG_FUNCS[eig_type]
        eigenvals, eigenvecs = eig_solver(
            Ka=self.Ka,
            Ma=self.Ma,
            num_modes=self.num_modes,
            path=self.folder,
            eig_names=self.eig_names,
            eigenvals=self.eigenvals,
            eigenvecs=self.eigenvecs,
        )
        logger.debug(f"Computing eigenvalue problem from {eig_type}")
        return eigenvals, eigenvecs

    def _compute_modalshapes(self):
        
        eigenvals, eigenvecs = self._compute_eigs()
        if self.constrainedDoF and False:
            modal_analysis = modes.shapes(
                self.X.T,
                self.Ka0s,
                self.Ma0s,
                eigenvals,
                eigenvecs,
                self.config,
            )
        else:
            modal_analysis = modes.shapes(
                self.X.T,
                self.Ka,
                self.Ma,
                eigenvals,
                eigenvecs,
                self.config,
            )

        modal_analysis_scaled = modes.scale(*modal_analysis)
        self.sol.add_container(f"Modes{self.label}", *modal_analysis_scaled)

    def _compute_modalcouplings(self):
        # if self._config.numlib == "jax":

        # elif self._config.numlib == "numpy":
        #    import feniax.intrinsic.couplings_np as couplings
        data_modes = getattr(self.sol.data, f"modes{self.label}")
        alpha1, alpha2 = modes.check_alphas(
            data_modes.phi1,
            data_modes.psi1,
            data_modes.phi2l,
            data_modes.psi2l,
            data_modes.X_xdelta,
            tolerance=self.config.jax_np.allclose,
        )
        gamma1 = couplings.f_gamma1(data_modes.phi1, data_modes.psi1)
        gamma2 = couplings.f_gamma2(
            data_modes.phi1ml,
            data_modes.phi2l,
            data_modes.psi2l,
            data_modes.X_xdelta,
        )

        self.sol.add_container(f"Couplings{self.label}", alpha1, alpha2, gamma1, gamma2)

    def _load_modalshapes(self):
        self.sol.load_container(f"Modes{self.label}")

    def _load_modalcouplings(self):
        self.sol.load_container(f"Couplings{self.label}")


