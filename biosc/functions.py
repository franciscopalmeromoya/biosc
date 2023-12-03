"""Transform units functions"""
import numpy as np
import pytensor
import pytensor.tensor as T

# Set configuration
floatX = pytensor.config.floatX

def m2flux(m_true):
    """Convert relative magnitud value to flux.

        F = F0 * 10^(-m/2.5)

    Units:
        flux: [erg/s/cm^2] 
        mag : [1]
        ZP_nu : [erg/s/cm^2] <- svo2.cab.inta-csic.es/
    """

    ZeroPoints = {
        'GAIA/GAIA3.G' : 2.5e-9,         # g
        'GAIA/GAIA3.Gbp' : 4.08e-9,      # bp
        'GAIA/GAIA3.Grp' : 1.27e-9,      # rp
        '2MASS/2MASS.J' : 3.13e-10,      # J
        '2MASS/2MASS.H' : 1.13e-10,      # H
        '2MASS/2MASS.Ks' : 4.28e-11,     # K
        'PAN-STARRS/PS1.g' : 5.05e-9,    # gmag
        'PAN-STARRS/PS1.r' : 2.47e-9,    # rmag
        'PAN-STARRS/PS1.i' : 1.36e-9,    # imag
        'PAN-STARRS/PS1.y' : 7.05e-10,	 # ymag
        'PAN-STARRS/PS1.z' : 9.01e-10    # zmag
    }

    F0 = np.array([ZeroPoints[x]  for x in ZeroPoints]).astype(floatX)

    return F0*10**(-0.4*m_true)

def flux2m(flux_obs):
    """Convert flux value to relative magnitud.

        m = -2.5 log (F/F0)
    
    Units:
        flux: [erg/s/cm^2] 
        mag : [1]
        ZP_nu : [erg/s/cm^2] <- svo2.cab.inta-csic.es/
    """

    ZeroPoints = {
        'GAIA/GAIA3.G' : 2.5e-9,         # g
        'GAIA/GAIA3.Gbp' : 4.08e-9,      # bp
        'GAIA/GAIA3.Grp' : 1.27e-9,      # rp
        '2MASS/2MASS.J' : 3.13e-10,      # J
        '2MASS/2MASS.H' : 1.13e-10,      # H
        '2MASS/2MASS.Ks' : 4.28e-11,     # K
        'PAN-STARRS/PS1.g' : 5.05e-9,    # gmag
        'PAN-STARRS/PS1.r' : 2.47e-9,    # rmag
        'PAN-STARRS/PS1.i' : 1.36e-9,    # imag
        'PAN-STARRS/PS1.y' : 7.05e-10,	 # ymag
        'PAN-STARRS/PS1.z' : 9.01e-10    # zmag
    }

    F0 = np.array([ZeroPoints[x]  for x in ZeroPoints]).astype(floatX)

    return -2.5*np.log10(flux_obs/F0)

def M2m(M, distance):
    """Convert absolute magnitude to relative magnitude.
    (M-5)+5*log(d)

    Units: 
        distance : [pc]
        M : [1]
        m : [1] 
    """
    distance_v = T.stack([distance for _ in range(11)], axis=1)
    return (M-5)+5*T.log10(distance_v)

def distance2parallax(distance):
    """Convert distance to parallax.

    Units:
        distance: kiloparsecs (kpc)
        parallax: milliarcseconds (mas)
    """
    return 1/distance

def m2M(m, parallax):
    """Convert m to M.
    m + 5*log(p) + 5

    Units:
        parallax : as
        M : [1]
        m : [1] 
    """
    if (len(m.shape) > 1):
        parallax_v = [parallax for _ in range(m.shape[1])]
        parallax = np.stack(parallax_v, axis=1)

    return m + 5*(np.log10(parallax)+1)