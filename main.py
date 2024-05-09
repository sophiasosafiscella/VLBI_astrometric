# --------------------------------------------
# VLBI Astrometric Parameters in pulsar timing
# --------------------------------------------

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import astropy
from pint.models import get_model_and_toas
import pint.toa as toa
from pint.models import get_model
from pint.residuals import Residuals
import pint.fitter
import pandas as pd

import sys

import noise_utils
import astropy.units as u

# Pulsars we are timing
PSR_name: str = "J1012+5307"

# Load the astrometric parameters
astrometric_values = pd.read_csv("./data/astrometric_values.csv", index_col=0)

# Load data to PINT
timfile: str = f"./data/NG_15yr_dataset/tim/{PSR_name}_PINT_20220305.nb.tim"
parfile: str = f"./data/NG_15yr_dataset/par/{PSR_name}_PINT_20220305.nb.par"
timing_model, toas = get_model_and_toas(parfile, timfile)

#toas = toa.get_TOAs(f"./NG_15yr_dataset/tim/{PSR_name}_PINT_20220305.nb.tim")
#timing_model = get_model(f"./NG_15yr_dataset/par/{PSR_name}_PINT_20220305.nb.par")

# Calculate residuals
residuals = Residuals(toas, timing_model).time_resids.to(u.us).value  # Don't forget to use actual timing residuals!
xt = toas.get_mjds()
errors = toas.get_errors().to(u.us).value

# Plot the original timing residuals
plt.figure()
plt.errorbar(xt, residuals, yerr=errors, fmt ='o')
plt.title(str(timing_model.PSR.value) + " Original Timing Residuals | $\sigma_\mathrm{TOA}$ = " + str(round(np.std(residuals), 2)))
plt.xlabel("MJD")
plt.ylabel("Residual ($\mu s$)")
plt.grid()
plt.show()

# Now we're going to convert from ecliptical to celestial equatorial coordinates
equatorial_timing_model = timing_model.as_ICRS(epoch=timing_model.POSEPOCH.value)

# We build a dictionary with a key for each parameter we want to set.
# The dictionary entries can be either
#  {'pulsar name': (parameter value, TEMPO_Fit_flag, uncertainty)} akin to a TEMPO par file form
# or
# {'pulsar name': (parameter value, )} for parameters that can't be fit
params = {
    "RAJ": (astrometric_values.loc[PSR_name, 'RAJ'], 1,  0.0 * pint.hourangle_second),
    "DECJ": (astrometric_values.loc[PSR_name, 'DECJ'], 1, 0.0 * u.arcsec),
    "PMRA": (astrometric_values.loc[PSR_name, 'PMRA'] * u.mas/u.yr, 1, 0.0 * u.mas/u.yr),
    "PMDEC": (astrometric_values.loc[PSR_name, 'PMDEC'] * u.mas/u.yr, 1, 0.0 * u.mas/u.yr),
    "PX": (astrometric_values.loc[PSR_name, 'PX'] * u.mas, 1, 0.0 * u.mas),
}

# Assign the parameters
print("Changing the astrometric parameters by the VLBI")
for name, info in params.items():
    par = getattr(equatorial_timing_model, name)  # Get parameter object from name
    par.quantity = info[0]  # set parameter value
    if len(info) > 1:
        if info[1] == 1:
            par.frozen = True  # Frozen means not fit.
        par.uncertainty = info[2]

# Set up and validate the model
equatorial_timing_model.setup()
equatorial_timing_model.validate()

# Fit the other timing parameters using the new astrometric values

# Perform initial fit
f = pint.fitter.DownhillGLSFitter(toas, equatorial_timing_model)
f.fit_toas(maxiter=5)

# Re-run noise
noise_utils.model_noise(equatorial_timing_model, toas, vary_red_noise=True, n_iter=int(1e5), using_wideband=False,
                        resume=True, run_noise_analysis=True, base_op_dir="noisemodel_linear_sd/")

newmodel = noise_utils.add_noise_to_model(equatorial_timing_model, base_dir="noisemodel_linear_sd/")
print(newmodel)
#newmodel.get_params_of_component_type("NoiseComponent")

# Final fit
new_f = pint.fitter.DownhillGLSFitter(toas, newmodel)
new_f.fit_toas(maxiter=5)


# Let's plot the residuals and compare
plt.figure()
plt.errorbar(
    xt,
    new_f.resids.time_resids.to(u.us).value,
    toas.get_errors().to(u.us).value,
    fmt='o',
)
plt.title("%s Post-Fit Timing Residuals" % equatorial_timing_model.PSR.value)
plt.xlabel("MJD")
plt.ylabel("Residual ($\mu s$)")
plt.grid()
plt.show()

sys.exit()
