import jax.numpy as jnp
import jax
import pyNastran.op4.op4 as OP4

@jax.jit
def frequency_matrix(k_array, poles):

    num_reducedfreq = len(k_array)
    num_poles = len(poles)
    k_array2 = k_array ** 2
    even_ids = jnp.arange(0, num_reducedfreq * 2, 2)
    odd_ids = jnp.arange(1, num_reducedfreq * 2, 2)
    k_matrix = jnp.zeros((num_reducedfreq * 2, 2 + num_poles))
    k_matrix = k_matrix.at[odd_ids, 0].set(k_array)
    k_matrix = k_matrix.at[even_ids, 1].set(k_array2)
    for i, pi in enumerate(poles):
        k_matrix = k_matrix.at[even_ids, 2 + i].set(
        pi / (k_array2 + pi ** 2))
        k_matrix = k_matrix.at[odd_ids, 2 + i].set(
         -k_array / (k_array2 + pi ** 2))

    return k_matrix

def stackQk_realimag(Qk):

    num_freqs, Qrows, Qcols  = Qk.shape
    Qk_real = Qk.real
    Qk_imag = Qk.imag
    even_ids = jnp.arange(0, num_freqs * 2, 2)
    odd_ids = jnp.arange(1, num_freqs * 2, 2)
    Qk_new = jnp.zeros((num_freqs * 2, Qrows, Qcols))
    Qk_new = Qk_new.at[even_ids].set(Qk_real)
    Qk_new = Qk_new.at[odd_ids].set(Qk_imag)
    return Qk_new

@jax.jit
def rogerRFA(k_matrix, Qk):

    num_freqs, Qrows, Qcols  = Qk.shape
    k_matrix_inv = jnp.linalg.pinv(k_matrix)
    A0 = Qk[0]
    Qk1_ = Qk[2:]

    def kernel(A0ij, Qij_k):

        Ap = k_matrix_inv @ (Qij_k - A0ij)
        return Ap

    loop_cols = jax.vmap(kernel, in_axes=(0, 1), out_axes=1)
    loop_rows = jax.vmap(loop_cols, in_axes=(0, 1), out_axes=1)

    A0_reshaped = A0.reshape((1,) + A0.shape)
    roger_matrices = loop_rows(A0, Qk1_)
    return jnp.vstack([A0_reshaped, roger_matrices])

@jax.jit
def Q_RFA(ki, roger_matrices, poles):

    Qk = roger_matrices[0] + roger_matrices[1]*1j*ki - roger_matrices[2]*ki**2
    for i, pi in enumerate(poles):
        Qk +=  roger_matrices[i+3] *ki *1j /(pi + ki * 1j)

    return Qk

def err_ki(ki, Qki_dlm, roger_matrices, poles, order=None):

    Qki_roger = Q_RFA(ki, roger_matrices, poles)
    err = jnp.linalg.norm((Qki_dlm - Qki_roger), order)
    return err

err_k = jax.vmap(err_ki, in_axes=(0,0, None, None, None))


def compute_err(reduced_freqs, Qk_dlm, roger_matrices, poles, norm_order=None):

    err = err_k(reduced_freqs, Qk_dlm, roger_matrices, poles, norm_order=None)
    err_avg = err.sum()/len(err)

    return err, err_avg

op4 = OP4.OP4()

aero = op4.read_op4_ascii("/Users/ac5015/pCloud Drive/tmp/Qhh50-50.op4")
aero2 = op4.read_op4("/Users/ac5015/pCloud Drive/tmp/Qhj0_8-50.op4")

qhh = jnp.array(aero['Q_HH'].data)
qhj =  jnp.array(aero2['Q_HJ'].data)


k_array = jnp.linspace(0,1,50)
poles = jnp.array([0.1,0.2,0.5, 0.9])
qhj_new = stackQk_realimag(qhj)

k_matrix = frequency_matrix(k_array, poles)
roger_matrices = rogerRFA(k_matrix, qhj_new)

# Q_RFA(ki, roger_matrices, poles)
