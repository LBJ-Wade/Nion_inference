# ==============================================================================

import sys, os
from math import pi
import numpy as np
import pylab as plt

from scipy import interpolate, integrate
from scipy.integrate import ode
from astropy.cosmology import FlatLambdaCDM, Planck15
import astropy.units as u

import reionization

# ==============================================================================
base_path    = os.path.abspath(os.path.dirname(__file__))
obs_dir_path = os.path.join(base_path, "../obs_constraints/")

class NionInference(object):
    """
    NAME
        NionInference.py

    PURPOSE
        Likelihoods and priors for use in Nion inference 
        (called in jupyter notebook nonparametric_nion.ipynb)

    COMMENTS

    FUNCTIONS

    BUGS

    AUTHORS

    HISTORY
        2019-04-12  started by Charlotte Mason (CfA)
    """

    # ==============================================================================

    def __init__(self, reion_obj, ztab=None, 
                 allconstraints=False, QSO_include='both', Qz6_1=False,
                 Nz_Lya=1000):
        
        self.reion_obj = reion_obj
        self.ztab      = ztab
        self.ndim      = len(ztab)

        # -------------------------------------
        # Load observational constraints

        self.Planck18 = {'tau':np.array([0.054]), 'tau_err':np.array([0.007])}

        M18a = np.genfromtxt(obs_dir_path+'Mason+18_z=7_xHI_posteriors_pentericci14_N=67.txt', dtype=None, names=True, skip_header=1)
        self.M18a_pxHI_interp = interpolate.interp1d(M18a['xHI'], M18a['pxHI'], bounds_error=False, fill_value=0.)
    
        M19 = np.genfromtxt(obs_dir_path+'Mason+19_z=8_xHI_posteriors_KLASS.txt', dtype=None, names=True, skip_header=1)
        self.M19_pxHI_interp = interpolate.interp1d(M19['xHI'], M19['pxHI'], bounds_error=False, fill_value=0.)

        H19 = np.genfromtxt(obs_dir_path+'Hoag+19_z=7.6_xHI_posterior.txt', dtype=None, names=True, skip_header=1)
        self.H19_pxHI_interp = interpolate.interp1d(H19['xHI'], H19['pxHI'], bounds_error=False, fill_value=0.)

        D18 = np.genfromtxt(obs_dir_path+'Davies+18_xhi_pdfs.dat', dtype=None, names=['xHI', 'pxHI_71', 'pxHI_75'], skip_header=3)
        self.D18_pxHI_71_interp = interpolate.interp1d(D18['xHI'], D18['pxHI_71'], bounds_error=False, fill_value=0.)
        self.D18_pxHI_75_interp = interpolate.interp1d(D18['xHI'], D18['pxHI_75'], bounds_error=False, fill_value=0.)

        G19_71 = np.genfromtxt(obs_dir_path+'Greig+19_posteriors_ULASJ1120.txt', dtype=None, names=True)
        self.G19_pxHI_71_interp = interpolate.interp1d(G19_71['xHI'], G19_71['p_med'], bounds_error=False, fill_value=0.)
        G19_75 = np.genfromtxt(obs_dir_path+'Greig+19_posteriors_ULASJ1342.txt', dtype=None, names=True)
        self.G19_pxHI_75_interp = interpolate.interp1d(G19_75['xHI'], G19_75['p_med'], bounds_error=False, fill_value=0.)

        # -------------------------------------
        # Redshifts for Lya EW constraints
        self.m18_z = np.random.normal(6.9, 0.5, Nz_Lya)
        self.m19_z = np.random.normal(7.9, 0.6, Nz_Lya)
        self.h19_z = np.random.normal(7.6, 0.6, Nz_Lya)

        # -------------------------------------
        # Inference setup
        self.allconstraints = allconstraints
        self.QSO_include    = QSO_include
        self.Qz6_1          = Qz6_1

        return


    # ================================================
    # Posterior (for emcee only)
    def lnpost(self, theta):

        lnpr = self.lnprior(theta)
        if not np.isfinite(lnpr):
            return -np.inf
        return lnpr + self.lnlike(theta)


    # ================================================
    # Likelihoods

    def lnlike(self, theta):
        """
        log likelihood for tau and dark fraction
        """

        # Load nion(z) from sample
        self.reion_obj.nion_z = self.reion_obj.nion_mod(lognion=theta)

        # Update C and solve for Q(z)
        self.reion_obj.Q_solve()

        # Tau
        tau_mod    = self.reion_obj.tau(z=14.)
        lnlike_tau = reionization.lnp_gauss(self.Planck18['tau'], tau_mod, self.Planck18['tau_err'])

        # Dark Fraction
        xHI_mod    = 1. - self.reion_obj.Q_interp(5.6)
        lnlike_DF1 = reionization.lnp_halfnormal(x=xHI_mod, mu=0.04, sig=0.05, limtype='upper')
        xHI_mod    = 1. - self.reion_obj.Q_interp(5.9)
        lnlike_DF2 = reionization.lnp_halfnormal(x=xHI_mod, mu=0.06, sig=0.05, limtype='upper')
        xHI_mod    = 1. - self.reion_obj.Q_interp(6.1)
        lnlike_DF3 = reionization.lnp_halfnormal(x=xHI_mod, mu=0.38, sig=0.20, limtype='upper')

        lnlike_all = lnlike_tau + lnlike_DF1 + lnlike_DF2 + lnlike_DF3

        if self.allconstraints:
            lnlike_all += self.ln_pxHI_other()

        lnlike_all = lnlike_all[0].astype(np.float)
        
        return lnlike_all


    def ln_pxHI_other(self):
        """
        Computer likelihood for other xHI measurements
        
        NB scipy interp is faster than np interp 
        (only have to generate interp once)
        """
        lnp = 0.
        # -----------------------------
        # Lya measurements - compute over z range

        # All Lya EW measured relative to z=6
        if self.Qz6_1:
            Q_z6 = 1.
        else:
            Q_z6 = self.reion_obj.Q_interp(6.)

        # Mason+18a
        delta_xHI_mod = Q_z6 - self.reion_obj.Q_interp(self.m18_z)
        lnp += np.log(np.nanmedian(self.M18a_pxHI_interp(delta_xHI_mod)))

        # Mason+19
        delta_xHI_mod = Q_z6 - self.reion_obj.Q_interp(self.m19_z)
        lnp += np.log(np.nanmedian(self.M19_pxHI_interp(delta_xHI_mod)))

        # Hoag+19
        delta_xHI_mod = Q_z6 - self.reion_obj.Q_interp(self.h19_z)
        lnp += np.log(np.nanmedian(self.H19_pxHI_interp(delta_xHI_mod)))

        # Sobacchi & Mesinger 2015 LAE clustering x_HI < 0.5
        z = 6.6
        xHI_mod = 1. - self.reion_obj.Q_interp(z)
        lnp += reionization.lnp_halfnormal(x=xHI_mod, mu=0.0, sig=0.5, limtype='upper')

        # ---------------------------
        # QSOs 
        delta_xHI_mod = 1. - self.reion_obj.Q_interp([7.0851, 7.5413])   
        
        if self.QSO_include == 'Greig':
            lnp += np.log(self.G19_pxHI_71_interp(delta_xHI_mod[0]))
            lnp += np.log(self.G19_pxHI_75_interp(delta_xHI_mod[1]))
        elif self.QSO_include == 'Davies':
            lnp += np.log(self.D18_pxHI_71_interp(delta_xHI_mod[0]))
            lnp += np.log(self.D18_pxHI_75_interp(delta_xHI_mod[1]))
        elif self.QSO_include == 'both':
            lnp += np.log(self.G19_pxHI_71_interp(delta_xHI_mod[0]))
            lnp += np.log(self.G19_pxHI_75_interp(delta_xHI_mod[1]))        
            lnp += np.log(self.D18_pxHI_71_interp(delta_xHI_mod[0]))
            lnp += np.log(self.D18_pxHI_75_interp(delta_xHI_mod[1]))
        else:
            print('ERROR: incorrect QSO inclusion type')       
       
        if np.isnan(lnp):
            return -np.inf
        else:
            return lnp

    # ================================================
    # Prior

    def prior_transform(self, u, a=-1, b=1, start=49.): 
        """
        prior tranform on n_ion_dot in dynesty
        
        u ~ Unif[0., 1.) for ndim
        
        Setup so that first parameter has prior U[49,53], 
        then the others have steps ~U[a,b]
        """
        nion_z      = (start + 4*u[0])*np.ones(self.ndim)
        randn       = np.random.uniform(a, b, size=self.ndim-1)
        cumsum      = np.cumsum(randn)
        nion_z[1:] += cumsum
        return nion_z

    def prior_transform_biguniform(self, u, start=48., Urange=8.): 
        """
        prior tranform on n_ion_dot in dynesty
        
        u ~ Unif[0., 1.) for ndim
        
        Setup so that each parameter has prior U[start,start+range], 
        """
        nion_z      = (start + Urange*u)
        return nion_z

    def lnprior(self, theta, a=-1, b=1, start=49.):
        """
        Prior for emcee
        """

        if start <= theta[0] <= start+4. and all(a <= diff <= b for diff in np.diff(theta)):
            return 0.
        else:
            return -np.inf

    # ================================================
    # Bouwens 2015

    def lnlike_B15(self, theta):
        """
        log likelihood similiar to Bouwens+2015
        """
        self.reion_obj.nion_z = (10**theta[0]) * (10**(theta[1] * (self.ztab - 8.)))/(1.E6 * self.reion_obj.pc) ** 3.
        self.reion_obj.nion_z[self.ztab < 6.] = self.reion_obj.nion_z[self.ztab == 6.]
        self.reion_obj.nion_z[self.ztab > 11.] = self.reion_obj.nion_z[self.ztab == 11.]

        self.reion_obj.Q_solve()

        # prior on reionization complete z=5.9-6.5 (xHII = 1)
        xHII_reion = self.reion_obj.Q_interp(zreion_Bouwens15)
        if (xHII_reion == 1.).any():        
            # Tau
            tau_mod    = self.reion_obj.tau(z=14.)
            lnlike_tau = reionization.lnp_gauss(0.066, tau_mod, 0.013) # 2015 Planck tau
            return lnlike_tau
        else:
            return -np.inf
        
    def prior_transform_B15(self, u, a=-0.6, b=0.1): 
        """
        prior tranform on n_ion_dot
        
        u ~ Unif[0., 1.) for ndim
        
        Setup so that first parameter has prior U[50,54], 
        then the others have steps ~U[a,b]
        """
        u[0] = 1.8*u[0] + 49.7
        u[1] = 0.7*u[1] - 0.6
        return u