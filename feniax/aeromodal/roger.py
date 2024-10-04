import jax.numpy as jnp
import jax
import pyNastran.op4.op4 as OP4
import jax

jax.config.update("jax_enable_x64", True)


@jax.jit
def frequency_matrixold(k_array, poles):
    num_reducedfreq = len(k_array)
    num_poles = len(poles)
    k_array2 = k_array**2
    even_ids = jnp.arange(0, num_reducedfreq * 2, 2)
    odd_ids = jnp.arange(1, num_reducedfreq * 2, 2)
    k_matrix = jnp.zeros((num_reducedfreq * 2, 2 + num_poles))
    k_matrix = k_matrix.at[odd_ids, 0].set(k_array)
    k_matrix = k_matrix.at[even_ids, 1].set(-k_array2)
    for i, pi in enumerate(poles):
        k_matrix = k_matrix.at[odd_ids, 2 + i].set((pi * k_array) / (k_array2 + pi**2))
        k_matrix = k_matrix.at[even_ids, 2 + i].set(k_array2 / (k_array2 + pi**2))

    return k_matrix


@jax.jit
def frequency_matrixold2(k_array, poles):
    num_reducedfreq = len(k_array)
    num_poles = len(poles)
    k_array2 = k_array**2
    # even_ids = jnp.arange(0, num_reducedfreq)#jnp.arange(0, num_reducedfreq * 2, 2)
    # odd_ids = jnp.arange(num_reducedfreq, 2 * num_reducedfreq) #jnp.arange(1, num_reducedfreq * 2, 2)
    # k_matrix = jnp.zeros((num_reducedfreq * 2, 2 + num_poles))
    k_matrix = jnp.zeros((num_reducedfreq, 3 + num_poles), dtype=complex)
    k_matrix = k_matrix.at[:, 0].set(k_array * 1j)
    k_matrix = k_matrix.at[:, 1].set(-k_array2)
    for i, pi in enumerate(poles):
        k_matrix = k_matrix.at[:, 2 + i].set((k_array * 1j) / (k_array * 1j + pi))
        # k_matrix = k_matrix.at[even_ids, 2 + i].set(
        #  k_array2 / (k_array2 + pi ** 2))
    kmatrix = jnp.vstack([k_matrix.real, k_matrix.imag])
    return kmatrix


@jax.jit
def frequency_matrix(k_array, poles):
    num_reducedfreq = len(k_array)
    num_poles = len(poles)
    k_array2 = k_array**2
    # even_ids = jnp.arange(0, num_reducedfreq)#jnp.arange(0, num_reducedfreq * 2, 2)
    # odd_ids = jnp.arange(num_reducedfreq, 2 * num_reducedfreq) #jnp.arange(1, num_reducedfreq * 2, 2)
    # k_matrix = jnp.zeros((num_reducedfreq * 2, 2 + num_poles))
    k_matrix = jnp.zeros((num_reducedfreq, 3 + num_poles), dtype=complex)
    k_matrix = k_matrix.at[:, 0].set(1.0)
    k_matrix = k_matrix.at[:, 1].set(k_array * 1j)
    k_matrix = k_matrix.at[:, 2].set(-k_array2)
    for i, pi in enumerate(poles):
        k_matrix = k_matrix.at[:, 3 + i].set((k_array * 1j) / (k_array * 1j + pi))
        # k_matrix = k_matrix.at[even_ids, 2 + i].set(
        #  k_array2 / (k_array2 + pi ** 2))
    kmatrix = jnp.vstack([k_matrix.real, k_matrix.imag])
    return kmatrix


def stackQk_realimag(Qk):
    num_freqs, Qrows, Qcols = Qk.shape
    Qk_real = Qk.real
    Qk_imag = Qk.imag
    # even_ids = jnp.arange(0, num_freqs * 2, 2)
    # odd_ids = jnp.arange(1, num_freqs * 2, 2)
    # Qk_new = jnp.zeros((num_freqs * 2, Qrows, Qcols))
    # Qk_new = Qk_new.at[even_ids].set(Qk_real)
    # Qk_new = Qk_new.at[odd_ids].set(Qk_imag)
    Qk_new = jnp.vstack([Qk_real, Qk_imag])
    return Qk_new


def stackQk_realimagold(Qk):
    num_freqs, Qrows, Qcols = Qk.shape
    Qk_real = Qk.real
    Qk_imag = Qk.imag
    even_ids = jnp.arange(0, num_freqs * 2, 2)
    odd_ids = jnp.arange(1, num_freqs * 2, 2)
    Qk_new = jnp.zeros((num_freqs * 2, Qrows, Qcols))
    Qk_new = Qk_new.at[even_ids].set(Qk_real)
    Qk_new = Qk_new.at[odd_ids].set(Qk_imag)
    return Qk_new


@jax.jit
def rogerRFAold(k_matrix, Qk):
    num_freqs, Qrows, Qcols = Qk.shape
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
def rogerRFAold2(k_matrix, Qk):
    num_freqs, Qrows, Qcols = Qk.shape
    k_matrix_inv = jnp.linalg.pinv(k_matrix)
    A0 = Qk[0]
    Qk1_ = Qk[
        jnp.hstack(
            [
                jnp.arange(1, int(num_freqs / 2)),
                jnp.arange(int(num_freqs / 2) + 1, num_freqs),
            ]
        )
    ]

    def kernel(A0ij, Qij_k):
        Ap = k_matrix_inv @ (Qij_k)
        return Ap

    loop_cols = jax.vmap(kernel, in_axes=(0, 1), out_axes=1)
    loop_rows = jax.vmap(loop_cols, in_axes=(0, 1), out_axes=1)

    A0_reshaped = A0.reshape((1,) + A0.shape)
    roger_matrices = loop_rows(A0, Qk1_)
    return jnp.vstack([A0_reshaped, roger_matrices])


@jax.jit
def rogerRFA(k_matrix, Qk):
    num_freqs, Qrows, Qcols = Qk.shape
    k_matrix_inv = jnp.linalg.pinv(k_matrix)
    A0 = Qk[0]
    Qk1_ = Qk[
        jnp.hstack(
            [
                jnp.arange(1, int(num_freqs / 2)),
                jnp.arange(int(num_freqs / 2) + 1, num_freqs),
            ]
        )
    ]

    def kernel(A0ij, Qij_k):
        Ap = k_matrix_inv @ (Qij_k)
        return Ap

    loop_cols = jax.vmap(kernel, in_axes=(0, 1), out_axes=1)
    loop_rows = jax.vmap(loop_cols, in_axes=(0, 1), out_axes=1)

    A0_reshaped = A0.reshape((1,) + A0.shape)
    roger_matrices = loop_rows(A0, Qk1_)
    return roger_matrices  # jnp.vstack([A0_reshaped, roger_matrices])


@jax.jit
def Q_RFAki(ki, roger_matrices, poles):
    Qk = roger_matrices[0] + roger_matrices[1] * 1j * ki - roger_matrices[2] * ki**2
    for i, pi in enumerate(poles):
        Qk += roger_matrices[i + 3] * ki * 1j / (pi + ki * 1j)

    return Qk


if __name__ == "__main__":
    Q_RFA = jax.vmap(Q_RFAki, in_axes=(0, None, None))

    def err_ki(ki, Qki_dlm, roger_matrices, poles, order=None):
        Qki_roger = Q_RFA(ki, roger_matrices, poles)
        err = jnp.linalg.norm((Qki_dlm - Qki_roger), order)
        return err

    err_k = jax.vmap(err_ki, in_axes=(0, 0, None, None, None))

    def compute_err(reduced_freqs, Qk_dlm, roger_matrices, poles, norm_order=None):
        err = err_k(reduced_freqs, Qk_dlm, roger_matrices, poles, norm_order=None)
        err_avg = err.sum() / len(err)

        return err, err_avg

    op4 = OP4.OP4()

    from pathlib import Path
    home = Path.home()
    aero = op4.read_op4_ascii(f"{home}/pCloudDrive/tmp/Qhh50-50.op4")
    aero2 = op4.read_op4(f"{home}/pCloudDrive/tmp/Qhj0_8-50.op4")

    try:
        qhh = jnp.array(aero["Q_HH"].data)
        qhj = jnp.array(aero2["Q_HJ"].data)
    except AttributeError:
        qhh = jnp.array(aero["Q_HH"][1])
        qhj = jnp.array(aero2["Q_HJ"][1])

    k_array = jnp.linspace(1e-3, 1, 50)  # jnp.linspace(0,1,50)
    poles = jnp.linspace(
        1e-3, 1, 40
    )  # jnp.array([0.05, 0.1, 0.15, 0.2, 0.5, 0.7, 0.9, 1.2, 1.6, 1.8, 2.5])
    qhj_new = stackQk_realimag(qhj)
    qhh_new = stackQk_realimag(qhh)

    k_matrix = frequency_matrix(k_array, poles)
    k_matrixold = frequency_matrixold(k_array, poles)
    roger_matricesQhj = rogerRFA(k_matrix, qhj_new)
    roger_matricesQhh = rogerRFA(k_matrix, qhh_new)

    ks = jnp.hstack([1e-6, k_array])
    Qk_hj = Q_RFA(ks, roger_matricesQhj, poles)
    Qk_hh = Q_RFA(ks, roger_matricesQhh, poles)

    PLOT = True
    if PLOT:
        import plotly.express as px
        import plotly.graph_objects as go

        i = 1
        j = 50
        #i = 20
        #j = 18

        fig = go.Figure()
        # fig.add_trace(go.Scatter(x=qhh[:,i,j].real, y=qhh[:,i,j].imag),
        #                     mode='makers',
        #                     # name='lines'
        #               )
        fig.add_trace(
            go.Scatter(
                x=qhh[:, i, j].real,
                y=qhh[:, i, j].imag,
                mode="markers",
                # name='lines'
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=Qk_hh[:, i, j].real,
                y=Qk_hh[:, i, j].imag,
                mode="lines",
                # name='lines'
            ),
        )
        # fig.add_trace(px.scatter(x=qhh[:,i,j].real, y=qhh[:,i,j].imag))

        # fig = px.scatter(x=qhh[:,i,j].real, y=qhh[:,i,j].imag)
        fig.show()

        # Q_RFA(ki, roger_matrices, poles)

        fig = go.Figure()
        # fig.add_trace(go.Scatter(x=qhh[:,i,j].real, y=qhh[:,i,j].imag),
        #                     mode='makers',
        #                     # name='lines'
        #               )
        fig.add_trace(
            go.Scatter(
                x=qhj[:, i, j].real,
                y=qhj[:, i, j].imag,
                mode="markers",
                # name='lines'
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=Qk_hj[:, i, j].real,
                y=Qk_hj[:, i, j].imag,
                mode="lines",
                # name='lines'
            ),
        )
        # fig.add_trace(px.scatter(x=qhh[:,i,j].real, y=qhh[:,i,j].imag))

        # fig = px.scatter(x=qhh[:,i,j].real, y=qhh[:,i,j].imag)
        fig.show()
