import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pint.toa import get_TOAs
from pint.models import get_model
from pint.residuals import Residuals
import pint.fitter
from pint_pal import noise_utils
from pint.logging import setup as setup_log

from VLBI_utils import calculate_prior, replace_params

import astropy.units as u

import glob
import sys

# Turn off log messages. They can slow down the processing.
setup_log(level="WARNING")

plot: bool = False    # Make plots?
resume: bool = False  # Resume noise MCMC?
sns.set_theme(context="paper", style="darkgrid", font_scale=1.5)

# List of pulsars
VLBI_data = pd.read_csv("./data/astrometric_values.csv", index_col=0)
PSR_list = VLBI_data.index

# Iterate over the pulsars
for i, PSR_name in enumerate(PSR_list[0:1]):

    # Names of the .tim and .par files
    timfile: str = glob.glob(f"./data/NG_15yr_dataset/tim/{PSR_name}*tim")[0]
    parfile: str = glob.glob(f"./data/NG_15yr_dataset/par/{PSR_name}*par")[0]

    # Load the timing model and convert to equatorial coordinates
    ec_timing_model = get_model(parfile)  # Ecliptical coordiantes
    eq_timing_model = ec_timing_model.as_ICRS(epoch=ec_timing_model.POSEPOCH.value)

    # Load the TOAs
    toas = get_TOAs(timfile, planets=True)

    # Read the different combinations of the mu_alpha, mu_delta, px
    values_df = pd.read_pickle(f"./results/{PSR_name}_overlap.pkl")

    # Make an array to store the posteriors values from each timing solution
    results = np.empty((len(values_df.index), 4), dtype=float)

    # Iterate over the different combinations of mu_alpha, mu_delta, px
    for j, timing_solution in enumerate(values_df.itertuples(index=True)):

        print(f"Proecssing iteration {j} of {PSR_name}")

        # Plot the original timing residuals
        if plot:
            # Calculate residuals. Don't forget to use actual timing residuals!
            residuals = Residuals(toas, eq_timing_model).time_resids.to(u.us).value
            xt = toas.get_mjds()
            errors = toas.get_errors().to(u.us).value

            plt.figure()
            plt.errorbar(xt, residuals, yerr=errors, fmt='o')
            plt.title(str(eq_timing_model.PSR.value) + " Original Timing Residuals | $\sigma_\mathrm{TOA}$ = " + str(
                round(np.std(residuals), 2)))
            plt.xlabel("MJD")
            plt.ylabel("Residual ($\mu s$)")
            plt.tight_layout()
            plt.show()

        # Replace the timing parameter values in the model with those from the new timing solution
        eq_timing_model = replace_params(eq_timing_model, timing_solution)

        # Set up and validate the new model
        eq_timing_model.setup()
        eq_timing_model.validate()

        # Perform initial fit
        initial_fit = pint.fitter.DownhillGLSFitter(toas, eq_timing_model)
        initial_fit.fit_toas(maxiter=5)

        # Re-run noise
        noise_utils.model_noise(eq_timing_model, toas, vary_red_noise=True, n_iter=int(1e5), using_wideband=False, resume=resume, run_noise_analysis=True, base_op_dir="noisemodel_linear_sd/")
        newmodel = noise_utils.add_noise_to_model(eq_timing_model, base_dir="noisemodel_linear_sd/")

        # Final fit
        final_fit = pint.fitter.DownhillGLSFitter(toas, newmodel)
        final_fit.fit_toas(maxiter=5)
        final_fit_resids = final_fit.resids

        # Calculate the posterior for this model and TOAs
        posterior = calculate_prior(eq_timing_model, VLBI_data, PSR_name) * final_fit_resids.lnlikelihood()
        results[i, :] = [timing_solution.PMRA, timing_solution.PMDEC, timing_solution.PX, posterior[0][0]]

        # Let's plot the residuals and compare
        if plot:
            plt.figure()
            plt.errorbar(
                xt,
                final_fit_resids.time_resids.to(u.us).value,
                toas.get_errors().to(u.us).value,
                fmt='o',
            )
            plt.title("%s Post-Fit Timing Residuals" % eq_timing_model.PSR_name.value)
            plt.xlabel("MJD")
            plt.ylabel("Residual ($\mu s$)")
            plt.grid()
            plt.show()

        break

    # Output the results
    res_df = pd.DataFrame(data=results, columns=["PMRA", "PMDEC", "PX", "posterior"])
    res_df.to_pickle(f"./results/{PSR_name}_posteriors_results.pkl")
    print(res_df)
