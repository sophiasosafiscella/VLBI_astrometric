import numpy as np
import pandas as pd
from astropy.coordinates import Angle
from numpy.linalg import inv

from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
import astropy.units as u

import sys

def eq_to_cartesian(alpha, delta):

    # Add units
    alpha = Angle(alpha)
    delta = Angle(delta)

    print(alpha)
    print(delta)
    print(alpha.hms)
    print(delta.dms)
    sys.exit()

    # Convert to radians and calculate the cartesian componentes
    x = np.cos(delta.radian) * np.cos(alpha.radian)
    y = np.cos(delta.radian) * np.sin(alpha.radian)
    z = np.sin(delta.radian)

    return np.array([x, y, z])


astrometric_data = pd.read_csv("./data/frame_tie.csv", index_col=0)
PSR_names = ["J0437-4715", "J1713+0747"]
N_pulsars: int = len(PSR_names)

D_T = np.empty((1, 3 * N_pulsars))
E_T = np.empty((1, 3 * N_pulsars))
M_T = np.empty((3, 3 * N_pulsars))

for i, PSR in enumerate(PSR_names):

#    coords = []
#    timing_coords = SkyCoord(coords, frame=FK4, unit=(u.hourangle, u.deg), obstime="J1992.21")


    D_T[0, 3*i:3*(i+1)] = np.matrix((eq_to_cartesian(astrometric_data.loc[f"{PSR}_timing", "RAJ"], astrometric_data.loc[f"{PSR}_timing", "DECJ"])
                                   - eq_to_cartesian(astrometric_data.loc[f"{PSR}_VLBI", "RAJ"], astrometric_data.loc[f"{PSR}_VLBI", "DECJ"])))

    E_T[0, 3*i:3*(i+1)] = np.matrix((eq_to_cartesian(astrometric_data.loc[f"{PSR}_timing", "RAJ_err"], astrometric_data.loc[f"{PSR}_timing", "DECJ_err"])
                                   - eq_to_cartesian(astrometric_data.loc[f"{PSR}_VLBI", "RAJ_err"], astrometric_data.loc[f"{PSR}_VLBI", "DECJ_err"])))

    x, y, z = eq_to_cartesian(astrometric_data.loc[f"{PSR}_VLBI", "RAJ"], astrometric_data.loc[f"{PSR}_VLBI", "DECJ"])

    M_T[:, 3*i:3*(i+1)] = np.transpose(np.matrix([[0, -z, y],[z, 0, -x],[-y, x, 0]]))

sys.exit()
E = np.transpose(E_T)
D = np.transpose(D_T)
M = np.transpose(M_T)

sigma = np.dot(E, E_T)
sigma = np.diag(np.diag(sigma))  # We have approximated sigma as diagonal: since VLBI and timing measurement uncertainties
                                 # are uncorrelated and measurement uncertaintites between pulsars using either VLBI or timing
                                 # are uncorrelated
sigma_inv = np.linalg.inv(sigma)
print(sigma_inv)
sys.exit()

A = np.dot(np.dot(np.dot(inv(np.dot(np.dot(M_T, sigma**(-1)), M)), M_T), sigma**(-1)), D)

Ax, Ay, Az = Angle(A[0, 0], u.radian), Angle(A[1, 0], u.radian), Angle(A[2, 0], u.radian)

print(Ax.arcsec)
print(Ay.arcsec)
print(Az.arcsec)
