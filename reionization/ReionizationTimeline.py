# ==============================================================================

import sys, os
from math import pi
import numpy as np
import pylab as plt

from scipy import interpolate, integrate
from scipy.integrate import ode
from astropy.cosmology import FlatLambdaCDM, Planck15
import astropy.units as u

base_path    = os.path.abspath(os.path.dirname(__file__))
obs_dir_path = os.path.join(base_path, "../obs_constraints/")

# ==============================================================================

class ReionizationTimeline(object):
    """
    NAME
        ReionizationTimeline.py

    PURPOSE
        Solves reionization ODE:
            dQ/dt = nion_dot/n_H - Q(z)/trec

            x dt/dz

        --> dQ/dz + a(z)Q(z) = b(z)

        And some other useful reionization functions

    COMMENTS

    FUNCTIONS

    BUGS

    AUTHORS

    HISTORY
        2015-07-28  started by Charlotte Mason (UCSB)
        2019-02-19  updated for Nion inference, C Mason (CfA)
    """

    # ==============================================================================

    def __init__(self, ztab=None, C=3., fesc=0.2, log_xion=25.2,
                 cosmology='Planck15'):

        if ztab is None:
            self.ztab = np.linspace(0., 16., 100)
        else:
            self.ztab = ztab

        # Load Ldens for comparisons
        self.load_Ldens()

        # Cosmology
        if cosmology == 'Planck15':
            self.cosmo = Planck15
        else:
            self.cosmo = FlatLambdaCDM(H0=70., Om0=0.3, Ob0=0.05, Tcmb0=2.7)

        print('Using %s cosmology' % self.cosmo.name)

        self.O_b   = self.cosmo.Ob0
        self.h     = self.cosmo.h
        self.rho_c = self.cosmo.critical_density0.value  # g cm^-3
        self.H0    = self.cosmo.H0.to(u.Hz).value        # s^-1

        # Constants
        self.pc      = 3.10E18   # cm
        self.c       = 3.E10     # cm/s
        self.thomson = 6.65E-25  # cm^2 Thomson scattering optical depth

        # Primordial H & He abundance
        self.X_p  = 0.75
        self.Y_p  = 1 - self.X_p
        self.m_H  = 1.67E-24  # g
        self.m_He = 6.65E-24  # g

        # mean molecular weight
        self.mu = 1. / (self.m_H * self.Y_p / self.m_He + self.X_p)

        # mean number density of hydrogen
        self.n_H = self.X_p * self.O_b * self.rho_c / self.m_H  # cm^-3

        # Case B Hydrogen recombination coefficient at T=20,000 K (from Draine)
        self.T0  = 2.e4 # K
        self.a_B = 1.43E-13  # cm^3 s^-1

        # Clumping factor and escape fraction
        self.C    = C
        self.fesc = fesc
        self.xion = 10 ** log_xion

        # tau prefactor for quicker integrals
        self.tau_prefac = self.c * self.n_H * self.thomson

        # Don't set this until we've solved for Q
        self.Q_interp = None
        self.nion_z = None

        return

    # ==============================================================================

    def load_Ldens(self, Ldens=None, Ldens_z=None, M_lim=-12.):
        """
        Load Ldens file for comparisons
        and interpolate luminosity density with redshift
        """

        if Ldens is None:
            # Load Mason, Trenti & Treu 2015 luminosity density
            Ldens_dir = obs_dir_path +'lum_dens/'
            self.Ldens_file = Ldens_dir + 'Ldens_Mlim%.1f_dustcorr.txt' % M_lim
            self.Ldens      = np.genfromtxt(self.Ldens_file, comments='#')
            self.Ldens_bin  = self.Ldens[:, 1]
            self.z_bin      = self.Ldens[:, 0]
        else:
            self.Ldens_bin = Ldens
            self.z_bin     = Ldens_z

        self.rho_L_interp = interpolate.interp1d(self.z_bin, self.Ldens_bin)

        return

    def rho_L(self, z):
        return 10.**self.rho_L_interp(z)

    # -------------------------------------------------------------------------------

    def Hz(self, z):
        """
        Hubble parameter in s^-1
        """
        return self.H0 * np.sqrt(self.cosmo.Om0 * (1. + z) ** 3 + self.cosmo.Ode0)

    def dt_dz(self, z):
        return -1. / ((1. + z) * self.Hz(z))

    def nion_dot(self, z):
        """
        Ionizing flux in s^-1 cm^-3
        """
        return self.fesc * self.xion * self.rho_L(z) / (1E6 * self.pc) ** 3.

    def nion_mod(self, lognion):
        """
        Theta values in s-1 Mpc-3, need to be s-1 cm-3 for solving dQ/dz
        """
        nion = (10 ** lognion) / (1.E6 * self.pc) ** 3.
        return nion

    def inv_t_rec(self, z):
        """
        Recombination time in seconds as a function of redshift
        and clumping factor
        """
        return self.C_func(z) * self.a_B * (1 + self.Y_p / 4. / self.X_p) * self.n_H * (1. + z) ** 3.

    def inv_t_rec_Bouwens15(self, z):
        """
        Recombination time in seconds as a function of redshift
        and clumping factor from Bouwens+2015 
        """
        return 0.88 * ((1.+z)/7)**-3. * (self.T0/2.e4)**-0.7 * (self.C_func(z)/3)**-1.

    def C_func(self, z):
        
        if self.C == 'Shull2012':
            """
            Clumping factor from Shull 2012
            """
            return 2.9 * ((1. + z)/6.)**-1.1

        else:
            return self.C

    # ==============================================================================
    # Solve dQ/dt

    def Q_solve(self):
        """
        Integrate to find Q(z) with ztab.max() redshift of complete reionization
        z can be array

        N.B. this is IONIZED fraction
        """

        # Must do in reverse as integral explodes at low z

        # Use given Ldens to solve
        # Nion_dot and n_H are both comoving (factors (1+z)^3 cancel out)
        if self.nion_z is None:
            def dQdz(z, Q):
                dQ_dt = self.nion_dot(z) / self.n_H - Q * self.inv_t_rec(z)
                return self.dt_dz(z) * dQ_dt
        else:
            nion_interp = interpolate.interp1d(self.ztab, self.nion_z)
            def dQdz(z, Q):
                dQ_dt = nion_interp(z) / self.n_H - Q * self.inv_t_rec(z)
                return self.dt_dz(z) * dQ_dt

        Q = integrate.solve_ivp(dQdz,
                                t_span=(self.ztab.max(), self.ztab.min()), y0=[0.],
                                t_eval=np.linspace(self.ztab.max(), self.ztab.min(), 50),
                                method="RK23")

        Q_HII = Q.y[0][::-1]
        
        # Max ionized fraction = 1
        Q_HII[Q_HII > 1.] = 1.

        # Now solved, interpolate so we don't have to solve it again!
        # If bound error esually interpolating at low z, where should be 1
        self.Q_interp = interpolate.interp1d(Q.t[::-1], Q_HII, bounds_error=False, fill_value=1.)

        return

    # -------------------------------------------------------------------------------

    # Analytic integration via integrating factor (much slower)

    def a(self, z):
        return -self.dt_dz(z) * self.inv_t_rec(z)

    def b(self, z):
        """
        log_rho_L[erg/s/Hz/Mpc^3]
        log_xion[Hz/erg]
        convert to s^-1 cm^-3
        """
        return -self.dt_dz(z) * self.nion_dot(z) / self.n_H

    def integrating_factor(self, z):
        ztab = np.linspace(0., z)
        integral = np.trapz(self.a(ztab), ztab)
        return np.exp(integral)

    def IF_b_integral(self, z):
        return self.integrating_factor(z) * self.b(z)

    def Q_IF(self, z):
        """
        Integrate to find Q(z) with z0 redshift of complete reionization
        Using integrating factor. z must be float
        """

        # Boundary condition when completely neutral
        z1 = 15.5

        if z < z1:
            x = np.linspace(z, z1, 20)
            # y = np.array([self.IF_b_integral(X) for X in x])
            y = np.array([self.integrating_factor(X) for X in x]) * self.b(x)
            integral = np.trapz(y, x)
            solution = integral / self.integrating_factor(z)
        else:
            solution = 0.

        if solution > 1:
            solution = 1.

        return solution

    # ==============================================================================

    def tau(self, z, z_He=4.):
        """
        Optical depth as a function of redshift

        z_He is redshift of He reionization 
        (below this redshift 2 e- are produced for every He atom, otherwise only 1)
        e.g. Kuhlen & Faucher-Giguere 2012
        """
        # Solve Q if we haven't yet
        if self.Q_interp is None:
            self.Q_solve()

        zprime = np.linspace(0, z, 50)

        # Fraction of free electrons, assume He reionization at z=4
        eta  = np.ones_like(zprime)
        eta[zprime <= z_He] = 2.
        f_e  = (1 + eta * self.Y_p / 4. / self.X_p)

        dtau = f_e * self.tau_prefac * (1. + zprime) ** 2 / self.Hz(zprime)

        if z > 4.:
            Qz = self.Q_interp(zprime)
        else:
            Qz = 1.

        tauz = np.trapz(Qz * dtau, zprime)

        return tauz