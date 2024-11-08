import jax.numpy as jnp
from jax import jit

from py2d.dealias import multiply_dealias_spectral_jit

@jit
def convection_conserved_dealias(Omega1_hat, U1_hat, V1_hat, Kx, Ky):
    
    # Convservative form
    # U1_hat = (1.j) * Ky * Psi1_hat
    # V1_hat = -(1.j) * Kx * Psi1_hat
    # U1 = jnp.real(jnp.fft.irfft2(U1_hat, s=[N,N]))
    # V1 = jnp.real(jnp.fft.irfft2(V1_hat, s=[N,N]))
    # Omega1 = jnp.real(jnp.fft.irfft2(Omega1_hat, s=[N,N]))

    # dealiasing
    U1Omega1_hat = multiply_dealias_spectral_jit(U1_hat, Omega1_hat)
    V1Omega1_hat = multiply_dealias_spectral_jit(V1_hat, Omega1_hat)

    conu1 = (1.j) * Kx * U1Omega1_hat
    conv1 = (1.j) * Ky * V1Omega1_hat
    convec_hat = conu1 + conv1
    
    # Non-conservative form
    Omega1x_hat = (1.j) * Kx * Omega1_hat
    Omega1y_hat = (1.j) * Ky * Omega1_hat

    # dealiasing
    U1Omega1x_hat = multiply_dealias_spectral_jit(U1_hat, Omega1x_hat)
    V1Omega1y_hat = multiply_dealias_spectral_jit(V1_hat, Omega1y_hat)

    conu1 = U1Omega1x_hat
    conv1 = V1Omega1y_hat
    convecN_hat = conu1 + conv1
    
    convec_hat = 0.5 * (convec_hat + convecN_hat)
    
    return convec_hat

@jit
def convection_conserved(Omega1_hat, U1_hat, V1_hat, Kx, Ky):
    
    # Convservative form
    # U1_hat = (1.j) * Ky * Psi1_hat
    # V1_hat = -(1.j) * Kx * Psi1_hat

    N = Omega1_hat.shape[0]

    U1 = jnp.fft.irfft2(U1_hat, s=[N,N])
    V1 = jnp.fft.irfft2(V1_hat, s=[N,N])
    Omega1 = jnp.fft.irfft2(Omega1_hat, s=[N,N])

    U1Omega1 = U1*Omega1
    V1Omega1 = V1*Omega1

    conu1 = (1.j) * Kx * jnp.fft.rfft2(U1Omega1)
    conv1 = (1.j) * Ky * jnp.fft.rfft2(V1Omega1)
    convec_hat = conu1 + conv1
    
    # Non-conservative form
    Omega1x_hat = (1.j) * Kx * Omega1_hat
    Omega1y_hat = (1.j) * Ky * Omega1_hat

    Omega1x = jnp.fft.irfft2(Omega1x_hat, s=[N,N])
    Omega1y = jnp.fft.irfft2(Omega1y_hat, s=[N,N])

    U1Omega1x = U1*Omega1x
    V1Omega1y = V1*Omega1y

    conu1 = jnp.fft.rfft2(U1Omega1x)
    conv1 = jnp.fft.rfft2(V1Omega1y)
    convecN_hat = conu1 + conv1
    
    convec_hat = 0.5 * (convec_hat + convecN_hat)
    
    return convec_hat