import jax.numpy as jnp
import numpy as np
from py2d.eddy_viscosity_models import (
    characteristic_strain_rate_smag, coefficient_dsmaglocal_PsiOmega, eddy_viscosity_smag_local, Tau_eddy_viscosity, eddy_viscosity_smag
)
from py2d.convert import Tau2PiOmega

def dsmaglocal(Psi, Omega, Kx, Ky, Ksq, Delta, localflag='from_tau', charflag='local', spectral=False):
    """
    Compute the dynamic Smagorinsky model for local eddy viscosity.

    Parameters:
    Psi (array): Stream function.
    Omega (array): Vorticity.
    Kx (array): Wavenumber in x-direction.
    Ky (array): Wavenumber in y-direction.
    Ksq (array): Squared wavenumber.
    Delta (float): Filter width.
    localflag (str): Method to compute Pi term ('from_tau' or 'from_sigma'). Default is 'from_tau'.
    charflag (str): Characteristic strain rate flag ('local' or other). Default is 'local'.
    spectral (bool): If True, inputs and outputs are in spectral space. Default is False.

    Returns:
    tuple: PiOmega_hat, eddy_viscosity, Cs
    """
    if spectral:
        Psi_hat = Psi
        Omega_hat = Omega
    else:
        Psi_hat = jnp.fft.rfft2(Psi)
        Omega_hat = jnp.fft.rfft2(Omega)

    # Compute characteristic strain rate
    characteristic_S = characteristic_strain_rate_smag(Psi_hat, Kx, Ky, Ksq)
    # Compute dynamic coefficient
    c_dynamic = coefficient_dsmaglocal_PsiOmega(Psi_hat, Omega_hat, characteristic_S, Kx, Ky, Ksq, Delta)
    Cs = jnp.sqrt(c_dynamic)

    if "local" in charflag:
        # Compute local eddy viscosity
        eddy_viscosity = eddy_viscosity_smag_local(Cs, Delta, characteristic_S)
    else:
        # Compute global eddy viscosity
        eddy_viscosity = eddy_viscosity_smag(Cs, Delta, characteristic_S)

    if localflag == 'from_sigma':
        # Calculate the PI term for local PI = ∇.(ν_e ∇ω )
        Grad_Omega_hat_dirx = Kx * np.fft.rfft2(eddy_viscosity * np.fft.irfft2(Kx * Omega_hat))
        Grad_Omega_hat_diry = Ky * np.fft.rfft2(eddy_viscosity * np.fft.irfft2(Ky * Omega_hat))
        PiOmega_hat = Grad_Omega_hat_dirx + Grad_Omega_hat_diry
    elif localflag == 'from_tau':
        # Calculate the PI term for local: ∇×∇.(-2 ν_e S_{ij} )
        Tau11, Tau12, Tau22 = Tau_eddy_viscosity(eddy_viscosity, Psi_hat, Kx, Ky)
        Tau11_hat = np.fft.fft2(Tau11)
        Tau12_hat = np.fft.fft2(Tau12)
        Tau22_hat = np.fft.fft2(Tau22)
        PiOmega_hat = Tau2PiOmega(Tau11_hat, Tau12_hat, Tau22_hat, Kx, Ky, spectral=True)

    if spectral:
        return PiOmega_hat, eddy_viscosity, Cs
    else:
        return jnp.fft.irfft2(PiOmega_hat), eddy_viscosity, Cs
