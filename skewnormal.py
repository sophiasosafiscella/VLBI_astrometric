import numpy as np
import scipy
from scipy.stats import skewnorm

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

    if any(np.array([wX, wL, wR]) < 0):
        raise ValueError("ERROR in parSkewNormal: Weights wL, wX, and wR must all be positive")
    if not ((pL < pX) and (pX < pR)):
        raise ValueError("ERROR in parSkewNormal: Probabilities must be such that pL < pX < pR")

    def fSkewNormal(theta):
        xi, omega, alpha = theta
        return sum(np.array([wL, wX, wR]) * (skewnorm.ppf([pL, pX, pR], xi=xi, omega=omega, alpha=alpha) - np.array([x-uL, x, x+uR]))**2)

    try:
        if abs(pR-pL) < 0.75:
            initial_guess = [x, (uL+uR)/2, 2]
        else:
            initial_guess = [x, (uL+uR)/4, 2]
        res = scipy.optimize.minimize(fSkewNormal, initial_guess, method='Nelder-Mead')
        theta = res.x
        return dict(zip(['xi', 'omega', 'alpha'], theta))
    except:
        raise ValueError("Optimization failed")

x: float = 1.17
uL: float = 0.05
uR: float = 0.04

xi, omega, alpha = parSkewNormal(x=x, uL=uL, uR=uR)
print(xi, omega, alpha)