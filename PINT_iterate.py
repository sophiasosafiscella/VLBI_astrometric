import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.linalg.misc import LinAlgError

from pint.toa import get_TOAs
from pint.models import get_model
from pint.residuals import Residuals
import pint.fitter
from pint_pal import noise_utils
from pint.logging import setup as setup_log

from VLBI_utils import calculate_prior, replace_params, add_noise_params

import astropy.units as u

import glob
import sys

# Turn off log messages. They can slow down the processing.
setup_log(level="WARNING")

plot: bool = False    # Make plots?
resume: bool = True  # Resume noise MCMC?
sns.set_theme(context="paper", style="darkgrid", font_scale=1.5)

# List of pulsars
VLBI_data = pd.read_csv("./data/astrometric_values.csv", index_col=0)
PSR_list = VLBI_data.index

# Iterate over the pulsars
for PSR_name in PSR_list[0:1]:

    # Names of the .tim and .par files
    timfile: str = glob.glob(f"./data/NG_15yr_dataset/tim/{PSR_name}*tim")[0]
    parfile: str = glob.glob(f"./data/NG_15yr_dataset/par/{PSR_name}*par")[0]

    # Read the different combinations of the mu_alpha, mu_delta, px
    values_df = pd.read_pickle(f"./results/{PSR_name}_overlap.pkl")

    # Make an array to store the posteriors values from each timing solution
    results = np.empty((len(values_df.index), 4), dtype=float)

    # Iterate over the different combinations of mu_alpha, mu_delta, px
    for timing_solution in values_df.itertuples(index=True):

        print(f"Processing iteration {timing_solution.Index} of {PSR_name}")

        # Load the timing model and convert to equatorial coordinates
        ec_timing_model = get_model(parfile)  # Ecliptical coordiantes
        eq_timing_model = ec_timing_model.as_ICRS(epoch=ec_timing_model.POSEPOCH.value)

        # Load the TOAs
        toas = get_TOAs(timfile, planets=True)

        # Plot the original timing residuals
        if plot:
            # Calculate residuals. Don't forget to use actual timing residuals!
            residuals = Residuals(toas, eq_timing_model).time_resids.to(u.us).value
            xt = toas.get_mjds()
            errors = toas.get_errors().to(u.us).value

            plt.figure()
            plt.errorbar(xt, residuals, yerr=errors, fmt='o')
            plt.title(str(PSR_name) + " Original Timing Residuals | $\sigma_\mathrm{TOA}$ = " + str(
                round(np.std(residuals), 2)))
            plt.xlabel("MJD")
            plt.ylabel("Residual ($\mu s$)")
            plt.tight_layout()
            plt.savefig("./figures/residuals/" + PSR_name + "_" + str(timing_solution.Index) + "_pre.png")
            plt.show()

        # Calculate EFAC and EQUAD
#        eq_timing_model = add_noise_params(eq_timing_model, EFAC=1.3, EQUAD=1.1)

        # Make lists of the EFAC and EQUAD parameters
        EFAC_params = []
        EQUAD_params = []
        for param in eq_timing_model.params:
            if param.startswith("EFAC"):
                EFAC_params.append(param)
            elif param.startswith("EQUAD"):
                EQUAD_params.append(param)

        # Unfree the parameters
        for name in EFAC_params + EQUAD_params:
            par = getattr(eq_timing_model, name)  # Get parameter object from
            par.frozen = False  # Frozen means not fit.

        print(eq_timing_model)
        sys.exit()


        # Replace the timing parameter values in the model with those from the new timing solution
        eq_timing_model = replace_params(eq_timing_model, timing_solution)

        # Set up and validate the new model
        eq_timing_model.setup()
        eq_timing_model.validate()

        # Perform initial fit
        initial_fit = pint.fitter.DownhillGLSFitter(toas, eq_timing_model)
        try:
           initial_fit.fit_toas(maxiter=5)
        except LinAlgError:
            print(f"LinAlgError at iteration {timing_solution.Index}")

        # Re-run noise
        noise_utils.model_noise(eq_timing_model, toas, vary_red_noise=True, n_iter=int(1e5), using_wideband=False, resume=resume, run_noise_analysis=True, base_op_dir="noisemodel_linear_sd/")
        newmodel = noise_utils.add_noise_to_model(eq_timing_model, base_dir="noisemodel_linear_sd/")

        # Final fit
        final_fit = pint.fitter.DownhillGLSFitter(toas, newmodel)
        try:
            final_fit.fit_toas(maxiter=5)
            final_fit_resids = final_fit.resids

            # Calculate the posterior for this model and TOAs
            posterior = calculate_prior(eq_timing_model, VLBI_data, PSR_name) * final_fit_resids.lnlikelihood()

        except LinAlgError:
            print(f"LinAlgError at iteration {timing_solution.Index}")
            posterior = [[0.0]]

        results[timing_solution.Index, :] = [timing_solution.PMRA, timing_solution.PMDEC, timing_solution.PX, posterior[0][0]]

        # Let's plot the residuals and compare
        if plot:
            plt.figure()
            plt.errorbar(
                xt,
                final_fit_resids.time_resids.to(u.us).value,
                toas.get_errors().to(u.us).value,
                fmt='o',
            )
            plt.title("%s Post-Fit Timing Residuals" % PSR_name)
            plt.xlabel("MJD")
            plt.ylabel("Residual ($\mu s$)")
            plt.grid()
            plt.tight_layout()
            plt.savefig("./figures/residuals/" + PSR_name + "_" + str(timing_solution.Index) + "_post.png")
            plt.show()

    # Output the results
    res_df = pd.DataFrame(data=results, columns=["PMRA", "PMDEC", "PX", "posterior"])
    res_df.to_pickle(f"./results/posteriors_2/{PSR_name}_posteriors_results.pkl")
