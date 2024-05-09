import numpy as np
import scipy
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm, skewnorm

def parSkewNormal(x, uL, uR, pX=0.5, pL=0.025, pR=0.975, wX=1, wL=1, wR=1):

    ## INPUTS
    ## x  = Measured value   : x is the 100*pX percentile
    ## uL = Left uncertainty : x - uL is the 100*pL percentile
    ## uR = Right uncertainty: x + uR is the 100*pR percentile
    ## wX, wL, wR = Weights for the errors made when attempting to
    ## reproduce x, x-uL, and x+uR as percentiles of a skew-normal
    ## distribution
    ## OUTPUT
    ## Vector with the values of xi, omega, and alpha for the best
    ## fitting skew-normal distribution

    # xi : vector of location parameters.
    # omega : vector of scale parameters; must be positive.
    # alpha : vector of slant parameter(s); +/- Inf is allowed. For psn, it must be of length 1 if engine="T.Owen". For qsn, it must be of length 1.

    if any(np.array([wX, wL, wR]) < 0):
        raise ValueError("ERROR in parSkewNormal: Weights wL, wX, and wR must all be positive")
    if not ((pL < pX) and (pX < pR)):
        raise ValueError("ERROR in parSkewNormal: Probabilities must be such that pL < pX < pR")

    def fSkewNormal(theta):
        loc, scale, a = theta
        return sum(np.array([wL, wX, wR]) * (skewnorm.ppf([pL, pX, pR], loc=loc, scale=scale, a=a) - np.array([x-uL, x, x+uR]))**2)

    try:
        if abs(pR-pL) < 0.75:
            initial_guess = [x, (uL+uR)/2, 2]  # Initial guesses of the parameters of the skew-normal distribution
        else:
            initial_guess = [x, (uL+uR)/4, 2]

        res = scipy.optimize.minimize(fSkewNormal, initial_guess, method='Nelder-Mead')
        theta = res.x  # Value of parameters of the skew-normal distribution with which it attains a minimum
        return dict(zip(['loc', 'scale', 'a'], theta))
    except:
        raise ValueError("Optimization failed")

def plot_pdf(x0, uL, uR, num: int = 1000):

    # If the error bars are equal, we have a normal distribution
    if uL == uR:
        x = np.linspace(x0 - 3.5 * uL, x0 + 3.5 * uR, num)
        y = norm.pdf(x, loc=x0, scale=uL)

    # If the error bars are not equal, we have a skew-normal distribution
    if uL != uR:
        res = parSkewNormal(x=x0, uL=uL, uR=uR)
        x = np.linspace(res['loc'] - 4 * res['scale'], res['loc'] + 4 * res['scale'], num)
        y = skewnorm.pdf(x, a=res['a'], loc=res['loc'], scale=res['scale'])

    return x, y

#    return [1,2,3], [1,2,3]


#x: float = 1.17
#uL: float = 0.05
#uR: float = 0.04

#res = parSkewNormal(x=x, uL=uL, uR=uR)

#x = np.linspace(1.0, 1.5, 1000)
#y = skewnorm.pdf(x, a=res['a'], loc=res['loc'], scale=res['scale'])

#sns.set_style("darkgrid")

#plt.plot(x, y)
#plt.title("SkewNormal PDF for $\Pi=1.17^{+0.04}_{-0.05}$")
#plt.xlabel('$\Pi$')
#plt.ylabel('Probability (unnormalized)')
#plt.show()
