"""
Preprocessor to intrinsic modal system.

Calls computation of intrinsic modes and modal couplings
"""

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
                    self.sol.save_container("Modes")
                    self.sol.save_container("Couplings")
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
        self.sol.add_container("Modes", *modal_analysis_scaled)

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

        self.sol.add_container("Couplings", alpha1, alpha2, gamma1, gamma2)

    def _load_modalshapes(self):
        self.sol.load_container("Modes")

    def _load_modalcouplings(self):
        self.sol.load_container("Couplings")


