import numpy as np
import pandas as pd
from astropy.coordinates import Angle
from numpy.linalg import inv, multi_dot

from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.time import Time
import astropy.units as u

import sys

def eq_to_cartesian(ra, dec):
    # Add units
    ra = Angle(ra).radian
    dec = Angle(dec).radian

    # Convert to radians and calculate the cartesian componentes
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    return np.array([x, y, z])


def error_propagation(ra, dec, ra_err, dec_err):
    # Convert to angles
    ra = Angle(ra).radian
    dec = Angle(dec).radian
    ra_err = Angle(ra_err).radian
    dec_err = Angle(dec_err).radian

    # Use error propagation
    x_err = np.sqrt(
        np.sin(dec) ** 2 * np.cos(ra) ** 2 * dec_err ** 2 + np.cos(dec) ** 2 * np.sin(ra) ** 2 * ra_err ** 2)
    y_err = np.sqrt(
        np.sin(dec) ** 2 * np.sin(ra) ** 2 * dec_err ** 2 + np.cos(dec) ** 2 * np.cos(ra) ** 2 * ra_err ** 2)
    z_err = np.cos(dec) * dec_err

    return np.array([x_err, y_err, z_err])


# Load the VLBI_data
data = pd.read_csv("./data/frame_tie.csv", index_col=0)
PSR_names = ["J0437-4715", "J1713+0747"]
N_pulsars: int = len(PSR_names)

# Define the matrix
D_T = np.empty((1, 3 * N_pulsars))
E_T = np.empty((1, 3 * N_pulsars))
M_T = np.empty((3, 3 * N_pulsars))

# Iterate over the pulsars
for i, PSR in enumerate(PSR_names):
    timing_coords = SkyCoord(ra=data.loc[f"{PSR}_timing", "RAJ"], dec=data.loc[f"{PSR}_timing", "DECJ"],
                             frame=FK4, unit=(u.hourangle, u.deg),
                             obstime=Time(val=data.loc[f"{PSR}_timing", "epoch"], format='mjd', scale='utc'))

    cartesian_timing_errs = error_propagation(ra=data.loc[f"{PSR}_timing", "RAJ"],
                                              dec=data.loc[f"{PSR}_timing", "DECJ"],
                                              ra_err=data.loc[f"{PSR}_timing", "RAJ_err"],
                                              dec_err=data.loc[f"{PSR}_timing", "DECJ_err"])

    VLBI_coords = SkyCoord(ra=data.loc[f"{PSR}_VLBI", "RAJ"], dec=data.loc[f"{PSR}_VLBI", "DECJ"],
                           frame=ICRS, unit=(u.hourangle, u.deg),
                           obstime=Time(val=data.loc[f"{PSR}_VLBI", "epoch"], format='mjd', scale='utc'))

    cartesian_VLBI_errs = error_propagation(ra=data.loc[f"{PSR}_VLBI", "RAJ"],
                                              dec=data.loc[f"{PSR}_VLBI", "DECJ"],
                                              ra_err=data.loc[f"{PSR}_VLBI", "RAJ_err"],
                                              dec_err=data.loc[f"{PSR}_VLBI", "DECJ_err"])

    D_T[0, 3 * i:3 * (i + 1)] = (timing_coords.cartesian - VLBI_coords.cartesian).get_xyz()

    E_T[0, 3 * i:3 * (i + 1)] = cartesian_timing_errs - cartesian_VLBI_errs

    x, y, z = VLBI_coords.cartesian.get_xyz()
    M_T[:, 3 * i:3 * (i + 1)] = np.transpose(np.matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]]))

# Do the calculations
E = np.transpose(E_T)
D = np.transpose(D_T)
M = np.transpose(M_T)

sigma = np.dot(E, E_T)
sigma = np.diag(np.diag(sigma))  # We have approximated sigma as diagonal
sigma_inv = np.linalg.inv(sigma)

#A = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(M_T, sigma_inv), M)), M_T), sigma_inv), D)

A = multi_dot([inv(multi_dot([M_T, sigma_inv, M])), M_T, sigma_inv, D])

Ax, Ay, Az = Angle(A[0, 0], u.radian), Angle(A[1, 0], u.radian), Angle(A[2, 0], u.radian)

print(Ax.arcsec*1000)
print(Ay.arcsec*1000)
print(Az.arcsec*1000)
