import jax.numpy as jnp
import numpy as np
import yaml

from py2d.eddy_viscosity_models import eddy_viscosity_smag, characteristic_strain_rate_smag, coefficient_dsmag_PsiOmega
from py2d.eddy_viscosity_models import eddy_viscosity_leith, characteristic_omega_leith, coefficient_dleith_PsiOmega
from py2d.eddy_viscosity_models import characteristic_omega_leith, coefficient_dleithlocal_PsiOmega, coefficient_dsmaglocal_PsiOmega
from py2d.gradient_model import PiOmegaGM2_gaussian, PiOmegaGM2_gaussian_dealias_spectral, PiOmegaGM4_gaussian, PiOmegaGM4_gaussian_dealias_spectral, PiOmegaGM6_gaussian, PiOmegaGM6_gaussian_dealias_spectral
from py2d.gradient_model import PiOmegaGM4_box, PiOmegaGM4_box_dealias_spectral, PiOmegaGM6_box, PiOmegaGM6_box_dealias_spectral
from py2d.gradient_model import PiOmega_gaussian_invert, PiOmega_gaussian_invert_dealias_spectral, PiOmega_box_invert, PiOmega_box_invert_dealias_spectral
from py2d.eddy_viscosity_models import Tau_eddy_viscosity
from py2d.convert import Tau2PiOmega, Psi2UV

from py2d.sgs_dl.eval import evaluate_model
from py2d.sgs_dl.init import initialize_model, initialize_model_norm

class SGSModel:

    def __init__(self, Kx, Ky, Ksq, Delta, method = 'NoSGS', C_MODEL=0, dealias=True, cnn_config_path=None):
        self.cnn_config_path = cnn_config_path
        self.set_method(method)
        # Constants
        self.Kx = Kx
        self.Ky = Ky
        self.Ksq = Ksq
        self.Delta = Delta
        self.C_MODEL = C_MODEL
        self.dealias = dealias
        # States
        self.Psi_hat, self.PiOmega_hat, self.eddy_viscosity, self.Cl, self.Cs = 0, 0, 0, None, None

    def set_method(self, method):
        if method == 'NoSGS':
            self.calculate = self.no_sgs_method
        #----------------------------------------------------------------------
        # Smagorinsky
        elif method == 'SMAG':
            self.calculate = self.smag_method
        #----------------------
        elif method == 'DSMAG':
            self.calculate = self.dsmag_method
        #----------------------
        elif method == 'DSMAG_tau_Local':
            print('SGS model: Dynamic Smagorinsky with local Cs(x,y), Π=∇×∇.(-2 ν_e S_{ij} )')
            self.calculate = self.dsmaglocal_method
            self.localflag='from_tau'
            self.charflag='mean'
        #----------------------
        elif method == 'DSMAG_sigma_Local':
            print('SGS model: Dynamic Smagorinsky with local Cs(x,y), Π=∇.(ν_e ∇ω )')
            self.calculate = self.dsmaglocal_method
            self.localflag='from_sigma'
            self.charflag='mean'
        #----------------------------------------------------------------------
        elif method == 'DSMAG_tau_Local_LocalS':
            print('SGS model: Dynamic Smagorinsky with local Cs(x,y), Π=∇×∇.(-2 ν_e S_{ij} ); where characteristics lenght is local')
            self.calculate = self.dsmaglocal_method
            self.localflag='from_tau'
            self.charflag='local'
        #----------------------
        elif method == 'DSMAG_sigma_Local_LocalS':
            print('SGS model: Dynamic Smagorinsky with local Cs(x,y), Π=∇.(ν_e ∇ω ); where characteristics lenght is local')
            self.calculate = self.dsmaglocal_method
            self.localflag='from_sigma'
            self.charflag='local'
        #----------------------------------------------------------------------
        # Leith
        elif method == 'LEITH':
            self.calculate = self.leith_method
        #----------------------
        elif method == 'DLEITH':
            self.calculate = self.dleith_method
        #----------------------
        elif method == 'DLEITH_tau_Local':
            self.calculate = self.dleithlocal_method
            self.localflag='from_tau'
        #----------------------
        elif method == 'DLEITH_sigma_Local':
            self.calculate = self.dleithlocal_method
            self.localflag='from_sigma'
        #----------------------------------------------------------------------
        # Gradient models
        elif method == 'PiOmegaGM2':
            self.calculate = self.PiOmegaGM2_method
        elif method == 'PiOmegaGM2_box':
            self.calculate = self.PiOmegaGM2_method
        #----------------------
        elif method == 'PiOmegaGM4':
            self.calculate = self.PiOmegaGM4_method
        elif method == 'PiOmegaGM4_box':
            self.calculate = self.PiOmegaGM4_box_method
        #----------------------
        elif method == 'PiOmegaGM6':
            self.calculate = self.PiOmegaGM6_method
        elif method == 'PiOmegaGM6_box':
            self.calculate = self.PiOmegaGM6_box_method
        #----------------------------------------------------------------------
        # Filter-inversion Closures
        elif method == 'gaussian_invert':
            self.calculate = self.gaussian_invert_method
        elif method == 'box_invert':
            self.calculate = self.box_invert_method
        #----------------------------------------------------------------------
        # NN models
        elif method == 'CNN':
            self.calculate = self.cnn_method

            assert self.cnn_config_path is not None, "CNN config path must be provided"

            with open(self.cnn_config_path) as f:
                self.cnn_config = yaml.safe_load(f)
        #----------------------
        elif method == 'GAN':
            self.calculate = self.gan_method
        #----------------------------------------------------------------------
        else:
            raise ValueError(f"Invalid method: {method}")

    def __expand_self__(self):
        Kx = self.Kx
        Ky = self.Ky
        Ksq = self.Ksq
        Delta = self.Delta
        C_MODEL = self.C_MODEL
        dealias = self.dealias
        return Kx, Ky, Ksq, Delta, C_MODEL, dealias

    def update_state(self, Psi_hat, Omega_hat, U_hat, V_hat):
        self.Psi_hat, self.Omega_hat = Psi_hat, Omega_hat
        self.U_hat, self.V_hat = U_hat, V_hat
        return None

    def no_sgs_method(self):
        PiOmega_hat = 0.0
        eddy_viscosity = 0.0
        return PiOmega_hat, eddy_viscosity


    def smag_method(self):#, Psi_hat, Cs, Delta):
        Kx, Ky, Ksq, Delta, Cs, _ = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat

        PiOmega_hat = 0.0
        characteristic_S = characteristic_strain_rate_smag(Psi_hat, Kx, Ky, Ksq)
        eddy_viscosity = eddy_viscosity_smag(Cs, Delta, characteristic_S)

        self.PiOmega_hat, self.eddy_viscosity, self.C_MODEL = PiOmega_hat, eddy_viscosity, Cs
        return PiOmega_hat, eddy_viscosity, Cs


    def leith_method(self):#, Omega_hat, Kx, Ky, Cl, Delta):
        Kx, Ky, Ksq, Delta, Cl, _ = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat

        PiOmega_hat = 0.0
        characteristic_Omega = characteristic_omega_leith(Omega_hat, Kx, Ky)
        eddy_viscosity = eddy_viscosity_leith(Cl, Delta, characteristic_Omega)

        self.PiOmega_hat, self.eddy_viscosity, self.C_MODEL = PiOmega_hat, eddy_viscosity, Cl
        return PiOmega_hat, eddy_viscosity, Cl


    def dsmag_method(self):#, Psi_hat, Omega_hat, Kx, Ky, Ksq, Delta):
        Kx, Ky, Ksq, Delta, _, _ = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat

        PiOmega_hat = 0.0
        characteristic_S = characteristic_strain_rate_smag(Psi_hat, Kx, Ky, Ksq)
        c_dynamic = coefficient_dsmag_PsiOmega(Psi_hat, Omega_hat, characteristic_S, Kx, Ky, Ksq, Delta)
        Cs = jnp.sqrt(c_dynamic)
        eddy_viscosity = eddy_viscosity_smag(Cs, Delta, characteristic_S)

        self.PiOmega_hat, self.eddy_viscosity, self.C_MODEL = PiOmega_hat, eddy_viscosity, Cs
        return PiOmega_hat, eddy_viscosity, Cs

    def dsmaglocal_method(self):#, Psi_hat, Omega_hat, Kx, Ky, Ksq, Delta):
        '''
        Smagorinsky model with local Cs
        '''
        Kx, Ky, Ksq, Delta, _, _ = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat

        PiOmega_hat = 0.0
        characteristic_S = characteristic_strain_rate_smag(Psi_hat, Kx, Ky, Ksq)
        c_dynamic = coefficient_dsmaglocal_PsiOmega(Psi_hat, Omega_hat, characteristic_S, Kx, Ky, Ksq, Delta)
        Cs = jnp.sqrt(c_dynamic)

        if "local" in self.charflag:
            eddy_viscosity = eddy_viscosity_smag_local(Cs, Delta, characteristic_S)
        else:
            '<|S|> is averaged in the domain'
            eddy_viscosity = eddy_viscosity_smag(Cs, Delta, characteristic_S)


        if self.localflag=='from_sigma':
            # Calculate the PI term for local PI = ∇.(ν_e ∇ω )
            Grad_Omega_hat_dirx = Kx*np.fft.fft2( eddy_viscosity * np.fft.ifft2(Kx*Omega_hat) )
            Grad_Omega_hat_diry = Ky*np.fft.fft2( eddy_viscosity * np.fft.ifft2(Ky*Omega_hat) )
            PiOmega_hat = Grad_Omega_hat_dirx + Grad_Omega_hat_diry

        elif self.localflag=='from_tau':
            # Calculate the PI term for local: ∇×∇.(-2 ν_e S_{ij} )
            Tau11, Tau12, Tau22 = Tau_eddy_viscosity(eddy_viscosity, Psi_hat, Kx, Ky)

            Tau11_hat = np.fft.fft2(Tau11)
            Tau12_hat = np.fft.fft2(Tau12)
            Tau22_hat = np.fft.fft2(Tau22)

            PiOmega_hat = Tau2PiOmega(Tau11_hat, Tau12_hat, Tau22_hat, Kx, Ky, spectral=True)

        #--------- DEBUG MODE ------------------------------------------------
        # import matplotlib.pyplot as plt
        # plt.rcParams['figure.dpi'] = 350
        # #''' test: difference between local  ∇.(ν_e ∇ω ) and not (ν_e ∇.(∇ω)=ν_e ∇^2 ω)
        # c_dynamic_old = coefficient_dsmag_PsiOmega(Psi_hat, Omega_hat, characteristic_S, Kx, Ky, Ksq, Delta)
        # Cs_old = np.sqrt(c_dynamic_old)
        # eddy_viscosity_old = eddy_viscosity_smag(Cs_old, Delta, characteristic_S)
        # Grad_Omega_hat_old = eddy_viscosity_old*(Ksq*Omega_hat)

        # Tau11, Tau12, Tau22 = Tau_eddy_viscosity(eddy_viscosity, Psi_hat, Kx, Ky)

        # Tau11_hat = np.fft.fft2(Tau11)
        # Tau12_hat = np.fft.fft2(Tau12)
        # Tau22_hat = np.fft.fft2(Tau22)

        # PiOmega_hat_tau = Tau2PiOmega_2DFHIT(Tau11_hat, Tau12_hat, Tau22_hat, Kx, Ky, spectral=True)


        # VMIN, VMAX = -3, 3
        # fig, axes = plt.subplots(2,3, figsize=(12,8))
        # plt.subplot(2,3,1)
        # plt.title(r'$\Pi=\nu_e \nabla.(\nabla \omega)$, (domain average)')
        # plt.pcolor(np.fft.ifft2(Grad_Omega_hat_old).real,vmin=VMIN,vmax=VMAX,cmap='bwr');plt.colorbar()
        # plt.subplot(2,3,2)
        # plt.title(r'$\Pi=\nabla.(\nu_e \nabla \omega)$, (local)')
        # plt.pcolor(np.fft.ifft2(PiOmega_hat).real,vmin=VMIN,vmax=VMAX,cmap='bwr');plt.colorbar()

        # plt.subplot(2,3,3) #  ∇×∇.(-2 ν_e S_{ij} )
        # plt.title(r'$\Pi=\nabla \times \nabla . ( -2 \nu_e \overline{S}_{ij})$, (local)')
        # plt.pcolor(np.fft.ifft2(PiOmega_hat_tau).real,vmin=VMIN,vmax=VMAX,cmap='bwr');plt.colorbar()

        # plt.subplot(2,3,4)
        # plt.title(r'$\nu_e \nabla.(\nabla \omega) - \nabla.(\nu_e \nabla \omega) $')
        # plt.pcolor(np.fft.ifft2(Grad_Omega_hat_old).real-np.fft.ifft2(PiOmega_hat).real,vmin=VMIN,vmax=VMAX,cmap='bwr');plt.colorbar()

        # plt.subplot(2,3,6)
        # plt.title(r'$C_S$')
        # plt.pcolor(Cs,vmin=VMIN,vmax=VMAX,cmap='bwr');plt.colorbar()
        # plt.subplot(2,3,5)
        # plt.title(r'$Local, \nu_e(x,y)$')
        # plt.pcolor(eddy_viscosity,cmap='gray_r');plt.colorbar()
        # # plt.subplot(2,3,6)
        # # plt.title(r'$\nu_e$')
        # # plt.pcolor(eddy_viscosity, cmap='gray_r');plt.colorbar()


        # for i, ax in enumerate(axes.flat):
        #     # Set the aspect ratio to equal
        #     ax.set_aspect('equal')

        # plt.show()
        # stop_test

        #PiOmega_hat is instead replaced
        eddy_viscosity = 0
        Cs = 0
        self.PiOmega_hat, self.eddy_viscosity, self.C_MODEL = PiOmega_hat, eddy_viscosity, Cs
        return PiOmega_hat, eddy_viscosity, Cs

    def dleith_method(self):#, Psi_hat, Omega_hat, Kx, Ky, Ksq, Delta):
        Kx, Ky, Ksq, Delta, _, _ = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat

        PiOmega_hat = 0.0
        characteristic_Omega = characteristic_omega_leith(Omega_hat, Kx, Ky)
        c_dynamic = coefficient_dleith_PsiOmega(Psi_hat, Omega_hat, characteristic_Omega, Kx, Ky, Ksq, Delta)
        Cl = c_dynamic ** (1/3)
        eddy_viscosity = eddy_viscosity_leith(Cl, Delta, characteristic_Omega)

        self.PiOmega_hat, self.eddy_viscosity, self.C_MODEL = PiOmega_hat, eddy_viscosity, Cl

        return PiOmega_hat, eddy_viscosity, Cl

    def dleithlocal_method(self):#, Psi_hat, Omega_hat, Kx, Ky, Ksq, Delta):
        Kx, Ky, Ksq, Delta, _ , _= self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat

        PiOmega_hat = 0.0
        characteristic_Omega = characteristic_omega_leith(Omega_hat, Kx, Ky)
        #
        c_dynamic = coefficient_dleithlocal_PsiOmega(Psi_hat, Omega_hat, characteristic_Omega, Kx, Ky, Ksq, Delta)
        Cl = c_dynamic ** (1/3)
        eddy_viscosity = eddy_viscosity_leith(Cl, Delta, characteristic_Omega)

        if self.localflag=='from_sigma':
            # Calculate the PI term for local PI = ∇.(ν_e ∇ω )
            Grad_Omega_hat_dirx = Kx*np.fft.fft2( eddy_viscosity * np.fft.ifft2(Kx*Omega_hat) )
            Grad_Omega_hat_diry = Ky*np.fft.fft2( eddy_viscosity * np.fft.ifft2(Ky*Omega_hat) )
            PiOmega_hat = Grad_Omega_hat_dirx + Grad_Omega_hat_diry

        elif self.localflag=='from_tau':
            # Calculate the PI term for local: ∇×∇.(-2 ν_e S_{ij} )
            Tau11, Tau12, Tau22 = Tau_eddy_viscosity(eddy_viscosity, Psi_hat, Kx, Ky)

            Tau11_hat = np.fft.fft2(Tau11)
            Tau12_hat = np.fft.fft2(Tau12)
            Tau22_hat = np.fft.fft2(Tau22)

            PiOmega_hat = Tau2PiOmega_2DFHIT(Tau11_hat, Tau12_hat, Tau22_hat, Kx, Ky, spectral=True)

        # --------- DEBUG MODE ------------------------------------------------
        # #''' test: difference between local  ∇.(ν_e ∇ω ) and not (ν_e ∇.(∇ω)=ν_e ∇^2 ω)
        # c_dynamic_old = coefficient_dleith_PsiOmega(Psi_hat, Omega_hat, characteristic_Omega, Kx, Ky, Ksq, Delta)
        # Cl_old = c_dynamic_old ** (1/3)
        # eddy_viscosity_old = eddy_viscosity_leith(Cl_old, Delta, characteristic_Omega)
        # Grad_Omega_hat_old = eddy_viscosity_old *(Ksq*Omega_hat)

        # PiOmega_hat_tau = Tau2PiOmega_2DFHIT(Tau11_hat, Tau12_hat, Tau22_hat, Kx, Ky, spectral=True)

        # import matplotlib.pyplot as plt
        # plt.rcParams['figure.dpi'] = 350
        # VMIN, VMAX = -2, 2
        # fig, axes = plt.subplots(2,3, figsize=(12,8))
        # plt.subplot(2,3,1)
        # plt.title(r'$\Pi=\nu_e \nabla.(\nabla \omega)$, Leith (domain average)')
        # plt.pcolor(np.fft.ifft2(Grad_Omega_hat_old).real,vmin=VMIN,vmax=VMAX,cmap='bwr');plt.colorbar()
        # plt.subplot(2,3,2)
        # plt.title(r'$\Pi=\nabla.(\nu_e \nabla \omega)$, Leith (local)')
        # plt.pcolor(np.fft.ifft2(PiOmega_hat).real,vmin=VMIN,vmax=VMAX,cmap='bwr');plt.colorbar()

        # plt.subplot(2,3,3) #  ∇×∇.(-2 ν_e S_{ij} )
        # plt.title(r'$\Pi=\nabla \times \nabla . ( -2 \nu_e \overline{S}_{ij})$, Leith (local)')
        # plt.pcolor(np.fft.ifft2(PiOmega_hat_tau).real,vmin=VMIN,vmax=VMAX,cmap='bwr');plt.colorbar()

        # plt.subplot(2,3,4)
        # plt.title(r'$\nu_e \nabla.(\nabla \omega) - \nabla.(\nu_e \nabla \omega) $')
        # plt.pcolor(np.fft.ifft2(Grad_Omega_hat_old).real-np.fft.ifft2(PiOmega_hat).real,vmin=VMIN,vmax=VMAX,cmap='bwr');plt.colorbar()

        # plt.subplot(2,3,6)
        # plt.title(r'$C_L$')
        # plt.pcolor(Cl,vmin=VMIN,vmax=VMAX,cmap='bwr');plt.colorbar()
        # plt.subplot(2,3,5)
        # plt.title(r'$Local, \nu_e(x,y)$')
        # plt.pcolor(eddy_viscosity,cmap='gray_r');plt.colorbar()
        # # plt.subplot(2,3,6)
        # # plt.title(r'$\nu_e$')
        # # plt.pcolor(eddy_viscosity, cmap='gray_r');plt.colorbar()


        # for i, ax in enumerate(axes.flat):
        #     # Set the aspect ratio to equal
        #     ax.set_aspect('equal')

        # plt.show()
        # stop_test

        #PiOmega_hat is instead replaced
        eddy_viscosity = 0
        Cl = 0
        self.PiOmega_hat, self.eddy_viscosity, self.C_MODEL = PiOmega_hat, eddy_viscosity, Cl

        return PiOmega_hat, eddy_viscosity, Cl

    def PiOmegaGM2_method(self):#, Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
        Kx, Ky, Ksq, Delta, _, dealias = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat
        U_hat, V_hat = self.U_hat, self.V_hat

        eddy_viscosity = 0
        if dealias:
            PiOmega_hat = PiOmegaGM2_gaussian_dealias_spectral(Omega_hat=Omega_hat, U_hat=U_hat, V_hat=V_hat, Kx=Kx, Ky=Ky, Delta=Delta)
        else:
            PiOmega = PiOmegaGM2_gaussian(Omega_hat=Omega_hat, U_hat=U_hat, V_hat=V_hat, Kx=Kx, Ky=Ky, Delta=Delta)
            PiOmega_hat = jnp.fft.rfft2(PiOmega)

        self.PiOmega_hat, self.eddy_viscosity = PiOmega_hat, eddy_viscosity

        return PiOmega_hat, eddy_viscosity

    def PiOmegaGM4_method(self):#, Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
        Kx, Ky, Ksq, Delta, _, dealias = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat
        U_hat, V_hat = self.U_hat, self.V_hat

        eddy_viscosity = 0
        if dealias:
            PiOmega_hat = PiOmegaGM4_gaussian_dealias_spectral(Omega_hat=Omega_hat, U_hat=U_hat, V_hat=V_hat, Kx=Kx, Ky=Ky, Delta=Delta)
        else:
            PiOmega = PiOmegaGM4_gaussian(Omega_hat=Omega_hat, U_hat=U_hat, V_hat=V_hat, Kx=Kx, Ky=Ky, Delta=Delta)
            PiOmega_hat = jnp.fft.rfft2(PiOmega)

        self.PiOmega_hat, self.eddy_viscosity = PiOmega_hat, eddy_viscosity

        return PiOmega_hat, eddy_viscosity

    def PiOmegaGM4_box_method(self):#, Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
        Kx, Ky, Ksq, Delta, _, dealias = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat
        U_hat, V_hat = self.U_hat, self.V_hat

        eddy_viscosity = 0
        if dealias:
            PiOmega_hat = PiOmegaGM4_box_dealias_spectral(Omega_hat=Omega_hat, U_hat=U_hat, V_hat=V_hat, Kx=Kx, Ky=Ky, Delta=Delta)
        else:
            PiOmega = PiOmegaGM4_box(Omega_hat=Omega_hat, U_hat=U_hat, V_hat=V_hat, Kx=Kx, Ky=Ky, Delta=Delta)
            PiOmega_hat = jnp.fft.rfft2(PiOmega)

        self.PiOmega_hat, self.eddy_viscosity = PiOmega_hat, eddy_viscosity
        return PiOmega_hat, eddy_viscosity

    def PiOmegaGM6_method(self):#, Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
        Kx, Ky, Ksq, Delta, _, dealias = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat
        U_hat, V_hat = self.U_hat, self.V_hat

        eddy_viscosity = 0
        if dealias:
            PiOmega_hat = PiOmegaGM6_gaussian_dealias_spectral(Omega_hat=Omega_hat, U_hat=U_hat, V_hat=V_hat, Kx=Kx, Ky=Ky, Delta=Delta)
        else:
            PiOmega = PiOmegaGM6_gaussian(Omega_hat=Omega_hat, U_hat=U_hat, V_hat=V_hat, Kx=Kx, Ky=Ky, Delta=Delta)
            PiOmega_hat = jnp.fft.rfft2(PiOmega)

        self.PiOmega_hat, self.eddy_viscosity = PiOmega_hat, eddy_viscosity
        return PiOmega_hat, eddy_viscosity

    def PiOmegaGM6_box_method(self):#, Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
        Kx, Ky, Ksq, Delta, _, dealias = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat
        U_hat, V_hat = self.U_hat, self.V_hat

        eddy_viscosity = 0
        if dealias:
            PiOmega_hat = PiOmegaGM6_box_dealias_spectral(Omega_hat=Omega_hat, U_hat=U_hat, V_hat=V_hat, Kx=Kx, Ky=Ky, Delta=Delta)
        else:
            PiOmega = PiOmegaGM6_box(Omega_hat=Omega_hat, U_hat=U_hat, V_hat=V_hat, Kx=Kx, Ky=Ky, Delta=Delta)
            PiOmega_hat = jnp.fft.rfft2(PiOmega)

        self.PiOmega_hat, self.eddy_viscosity = PiOmega_hat, eddy_viscosity
        return PiOmega_hat, eddy_viscosity

    def gaussian_invert_method(self):#, Omega_hat, U_hat, V_hat, Kx, Ky, Ksq, Delta):
        Kx, Ky, Ksq, Delta, _, dealias = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat
        U_hat, V_hat = self.U_hat, self.V_hat

        eddy_viscosity = 0
        if dealias:
            PiOmega_hat = PiOmega_gaussian_invert_dealias_spectral(Omegaf_hat=Omega_hat, Uf_hat=U_hat, Vf_hat=V_hat, Kx=Kx, Ky=Ky, Ksq=Ksq, Delta=Delta)
        else:
            PiOmega = PiOmega_gaussian_invert(Omegaf_hat=Omega_hat, Uf_hat=U_hat, Vf_hat=V_hat, Kx=Kx, Ky=Ky, Ksq=Ksq, Delta=Delta)
            PiOmega_hat = jnp.fft.rfft2(PiOmega)

        self.PiOmega_hat, self.eddy_viscosity = PiOmega_hat, eddy_viscosity
        return PiOmega_hat, eddy_viscosity

    def box_invert_method(self):#, Omega_hat, U_hat, V_hat, Kx, Ky, Delta):
        Kx, Ky, Ksq, Delta, _, dealias = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat
        U_hat, V_hat = self.U_hat, self.V_hat

        eddy_viscosity = 0
        if dealias:
            PiOmega_hat = PiOmega_box_invert_dealias_spectral(Omegaf_hat=Omega_hat, Uf_hat=U_hat, Vf_hat=V_hat, Kx=Kx, Ky=Ky, Delta=Delta)
        else:
            PiOmega = PiOmega_box_invert(Omegaf_hat=Omega_hat, Uf_hat=U_hat, Vf_hat=V_hat, Kx=Kx, Ky=Ky, Delta=Delta)
            PiOmega_hat = jnp.fft.rfft2(PiOmega)

        self.PiOmega_hat, self.eddy_viscosity = PiOmega_hat, eddy_viscosity
        return PiOmega_hat, eddy_viscosity

    def cnn_method(self):
        """Perform the CNN calculation."""

        Kx, Ky, Ksq, Delta, _, dealias = self.__expand_self__()
        Psi_hat, Omega_hat = self.Psi_hat, self.Omega_hat
        Psi = jnp.fft.irfft2(Psi_hat)
        Omega = jnp.fft.irfft2(Omega_hat)

        whichlib = self.cnn_config['library']

        # initialize cnn if not already initialized
        if not hasattr(self, 'model'):
            self.model = initialize_model(self.cnn_config['model_path'], whichlib)
            self.model_norm = initialize_model_norm(self.cnn_config['norm_path'])

        input_stepnorm = self.cnn_config['input_stepnorm']

        # pass Psi, Omega into the model
        model_output = evaluate_model(
                self.model,
                self.model_norm,
                {'psi': Psi, 'omega': Omega},
                whichlib=whichlib,
                input_stepnorm=input_stepnorm
            )

        # make output array float64
        model_output = model_output.astype(jnp.float64)

        if self.cnn_config['resid']:
            Omega_hat = self.Omega_hat
            U_hat, V_hat = self.U_hat, self.V_hat
            PiOmega_GM4 = PiOmegaGM4_gaussian(Omega_hat=Omega_hat, U_hat=U_hat, V_hat=V_hat, Kx=Kx, Ky=Ky, Delta=Delta)
            PiOmega = model_output + PiOmega_GM4
        else:
            PiOmega = model_output

        # convert output back to spectral space
        PiOmega_hat = jnp.fft.rfft2(PiOmega)
        eddy_viscosity = 0

        self.PiOmega_hat, self.eddy_viscosity = PiOmega_hat, eddy_viscosity

        return PiOmega_hat, eddy_viscosity


    def gan_method(self, data):
        # Implement GAN method
        return "Calculated with GAN method"

# Usage example:

# model = PiOmegaModel()  # Default method is NO-SGS
# model.set_method('CNN')
# result = model.calculate("Sample Data")
# print(result)  # Output: Calculated with NO-SGS method

