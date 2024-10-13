import jax.numpy as jnp
import jax
import pyNastran.op4.op4 as OP4
import jax
import itertools
import plotly.express as px
import plotly.graph_objects as go


jax.config.update("jax_enable_x64", True)



class ComputeRoger:

    def __init__(self, sampling_aeromatrices_, redfreqs_, poles_=None, option=1):

        self.roger_matrices = None
        self.sampling_aeromatrices = sampling_aeromatrices_
        self.redfreqs = redfreqs_
        if poles_ is not None:
            self.poles = poles_

    @property
    def poles(self):

        return self.poles

    @poles.setter
    def poles(self, _poles):

        self.poles = _poles
        self._solve()

    def get_frequency_matrix(self):
        ...
    def _solve(self):
        build_gafs(self.poles, self.sampling_aeromatrices, self.redfreqs)
        build_gafs(self.poles, self.sampling_aeromatrices, self.redfreqs)

        
class EvaluateRoger:
    ...

class OptimisePoles:
    ...

class PlotGAFs:
    ...

@jax.jit
def frequency_matrix(k_array, poles):
    num_reducedfreq = len(k_array)
    num_poles = len(poles)
    k_array2 = k_array**2
    # even_ids = jnp.arange(0, num_reducedfreq)#jnp.arange(0, num_reducedfreq * 2, 2)
    # odd_ids = jnp.arange(num_reducedfreq, 2 * num_reducedfreq) #jnp.arange(1, num_reducedfreq * 2, 2)
    # k_matrix = jnp.zeros((num_reducedfreq * 2, 2 + num_poles))
    k_matrix = jnp.zeros((num_reducedfreq, 2 + num_poles), dtype=complex)
    #k_matrix = k_matrix.at[:, 0].set(1.0)
    k_matrix = k_matrix.at[:, 0].set(k_array * 1j)
    k_matrix = k_matrix.at[:, 1].set(-k_array2)
    for i, pi in enumerate(poles):
        k_matrix = k_matrix.at[:, 2 + i].set((k_array * 1j) / (k_array * 1j + pi))
        # k_matrix = k_matrix.at[even_ids, 2 + i].set(
        #  k_array2 / (k_array2 + pi ** 2))
    kmatrix = jnp.vstack([k_matrix.real, k_matrix.imag])
    return kmatrix

@jax.jit
def frequency_matrix2(k_array, poles):
    num_reducedfreq = len(k_array)
    num_poles = len(poles)
    k_array2 = k_array**2
    k_matrix = jnp.zeros((num_reducedfreq, 3 + num_poles), dtype=complex)
    k_matrix = k_matrix.at[:, 0].set(1.0)
    k_matrix = k_matrix.at[:, 1].set(k_array * 1j)
    k_matrix = k_matrix.at[:, 2].set(-k_array2)
    for i, pi in enumerate(poles):
        k_matrix = k_matrix.at[:, 3 + i].set((k_array * 1j) / (k_array * 1j + pi))
        
    kmatrix = jnp.vstack([k_matrix.real, k_matrix.imag])
    return kmatrix


def stackQk_realimag(Qk):
    
    Qk_real = Qk.real
    Qk_imag = Qk.imag
    Qk_new = jnp.vstack([Qk_real, Qk_imag])
    return Qk_new

@jax.jit
def rogerRFA(k_matrix, Qk, A0):
    
    k_matrix_inv = jnp.linalg.pinv(k_matrix)
    
    def kernel(A0ij, Qij_k):
        Ap = k_matrix_inv @ (Qij_k - A0ij)
        return Ap

    loop_cols = jax.vmap(kernel, in_axes=(0, 1), out_axes=1)
    loop_rows = jax.vmap(loop_cols, in_axes=(0, 1), out_axes=1)

    A0_reshaped = A0.reshape((1,) + A0.shape)
    roger_matrices = loop_rows(A0, Qk)
    return jnp.vstack([A0_reshaped, roger_matrices])

@jax.jit
def rogerRFA2(k_matrix, Qk, A0=None):
    
    num_freqs, Qrows, Qcols = Qk.shape
    k_matrix_inv = jnp.linalg.pinv(k_matrix) # dim: states x numfreqs

    roger_matrices = jnp.einsum('ij,jkl', k_matrix_inv, Qk)
    return roger_matrices

@jax.jit
def Q_RFAki(ki, roger_matrices, poles):
    Qk = roger_matrices[0] + roger_matrices[1] * 1j * ki - roger_matrices[2] * ki**2
    for i, pi in enumerate(poles):
        Qk += roger_matrices[i + 3] * ki * 1j / (pi + ki * 1j)

    return Qk

Q_RFA = jax.vmap(Q_RFAki, in_axes=(0, None, None))

def err_ki(ki, Qki_dlm, roger_matrices, poles, order=None):
    
    Qki_roger = Q_RFAki(ki, roger_matrices, poles)
    err = jnp.linalg.norm(Qki_dlm - Qki_roger, order) / jnp.linalg.norm(Qki_dlm, order)
    return err

err_k = jax.vmap(err_ki, in_axes=(0, 0, None, None, None))

_err_dict = dict()
def save(fun):

    _err_dict[fun.__name__.split('_')[-1]] = fun
    def wrap(*args, **kwargs):
        
        return fun(*args, **kwargs)

    return wrap

@save
def err_average(reduced_freqs, Qk_dlm, roger_matrices, poles, norm_order=None):
    
    err = err_k(reduced_freqs, Qk_dlm, roger_matrices, poles, norm_order)

    return jnp.average(err)

@save
def err_max(reduced_freqs, Qk_dlm, roger_matrices, poles, norm_order=None):
    
    err = err_k(reduced_freqs, Qk_dlm, roger_matrices, poles, norm_order)

    return jnp.max(err)


def read_sampling(op4_file: str, matrix_type="Q_HH") -> jnp.ndarray:

    op4 = OP4.OP4()
    #op4model = op4.read_op4_ascii(op4_file)
    op4model = op4.read_op4(op4_file)

    try:
        Qmatrices = jnp.array(op4model[matrix_type].data)
    except AttributeError:
        Qmatrices = jnp.array(op4model[matrix_type][1])
        
    return Qmatrices

def build_gafs(poles, sampling_aeromatrices, redfreqs):

    aeromatrices_stack = stackQk_realimag(sampling_aeromatrices[1:])
    redfreqs_matrix = frequency_matrix(redfreqs[1:], poles)
    A0 = sampling_aeromatrices[0] #jnp.vstack([sampling_aeromatrices[0], jnp.zeros_like(sampling_aeromatrices[0])])
    roger_matrices = rogerRFA(redfreqs_matrix, aeromatrices_stack, A0)
    return roger_matrices

def build_gafs2(poles, sampling_aeromatrices, redfreqs):

    aeromatrices_stack = stackQk_realimag(sampling_aeromatrices)
    redfreqs_matrix = frequency_matrix2(redfreqs, poles)
    roger_matrices2 = rogerRFA2(redfreqs_matrix, aeromatrices_stack)
    return roger_matrices2

vbuild_gafs = jax.vmap(build_gafs, in_axes=(0, None, None))
vbuild_gafs2 = jax.vmap(build_gafs2, in_axes=(0, None, None))

def err_poles(roger_matrices: jnp.ndarray,
              poles: jnp.ndarray,
              error_name: str,
              redfreqs,
              Qk_dlm,
              norm_order):
    
    err_fun = _err_dict[error_name]
    loss = err_fun(redfreqs, Qk_dlm, roger_matrices, poles, norm_order)
    return loss

verr_poles = jax.vmap(err_poles, in_axes=(0, 0, None, None, None, None))

def build_polesgrid(num_poles, poles_step, poles_range):

    poles = jnp.arange(poles_range[0],
                       poles_range[1] + poles_step,
                       poles_step)
    poles_grid = jnp.array(list(itertools.combinations(poles, num_poles)))
    return poles_grid
    
def optimise_brute(poles_grid, redfreqs, sampling_aeromatrices, error_name="average",
                   norm_order=2
                   ):

    roger_matrices = vbuild_gafs2(poles_grid, sampling_aeromatrices, redfreqs)
    verror = verr_poles(roger_matrices,
               poles_grid,
               error_name,
               redfreqs,
               sampling_aeromatrices,
               norm_order)

    min_index = jnp.argmin(verror)
    return roger_matrices[min_index], poles_grid[min_index], verror[min_index]

def plot_gafs(irow, jcolumn, Qdlm, Qroger):

    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=Qdlm[:, irow, jcolumn].real,
            y=Qdlm[:, irow, jcolumn].imag,
            mode="markers",
            # name='lines'
        ),
    )
    for i, Qi in enumerate(Qroger):
        fig.add_trace(
            go.Scatter(
                x=Qi[:, irow, jcolumn].real,
                y=Qi[:, irow, jcolumn].imag,
                mode="lines",
                name=f"Q{i}"
            ),
        )
    fig.show()


if __name__ == "__main__":

    from pathlib import Path
    home = Path.home()
    op4 = OP4.OP4()

    aero = op4.read_op4_ascii(f"{home}/pCloudDrive/tmp/Qhh50-50.op4")
    aero2 = op4.read_op4(f"{home}/pCloudDrive/tmp/Qhj0_8-50.op4")

    try:
        qhh = jnp.array(aero["Q_HH"].data)
        qhj = jnp.array(aero2["Q_HJ"].data)
    except AttributeError:
        qhh = jnp.array(aero["Q_HH"][1])
        qhj = jnp.array(aero2["Q_HJ"][1])

    num_poles = 7
    poles_range = [0.01, 1]
    poles_step = 0.1
    polesgrid = build_polesgrid(num_poles, poles_step, poles_range)
    k_array = jnp.linspace(1e-3, 1, 50)
    ks = jnp.hstack([0, k_array])
    roger_matrices, poles, verror = optimise_brute(polesgrid, ks, qhh)
    roger_matrices2, poles2, verror2 = optimise_brute(jnp.array((jnp.linspace(0.01,1,num_poles),)), ks, qhh)
    Qroger = Q_RFA(ks, roger_matrices, poles)
    Qroger2 = Q_RFA(ks, roger_matrices2, poles2)
    plot_gafs(2, 1, qhh, [Qroger, Qroger2])
