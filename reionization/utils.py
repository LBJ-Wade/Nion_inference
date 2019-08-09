# ==============================================================================

import sys, os
import numpy as np

# ==============================================================================

def lnp_gauss(x, mu, sig):
    return -0.5 * (np.log(2 * np.pi * sig ** 2.) + (x - mu) ** 2. / sig ** 2.)

def lnp_halfnormal(x, mu, sig, limtype='lower'):
    lnp = lnp_gauss(x, mu, sig)

    # Floats
    try:
        if limtype == 'lower':
            if x > mu:
                lnp = lnp_gauss(mu, mu, sig)
        elif limtype == 'upper':
            if x < mu:
                lnp = lnp_gauss(mu, mu, sig)
        else:
            print('ERROR: incorrect limit type')

    # Arrays
    except:
        if limtype == 'lower':
            lnp[x > mu] = lnp_gauss(mu, mu, sig)
        elif limtype == 'upper':
            lnp[x < mu] = lnp_gauss(mu, mu, sig)
        else:
            print('ERROR: incorrect limit type')
            
    return lnp