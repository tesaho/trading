"""
https://github.com/ynouri/pysabr/blob/master/pysabr/sabr.py

http://www.frouah.com/finance%20notes/The%20SABR%20Model.pdf
based on the paper by Fabrice Douglas Rouah -> Hagan 2002
and Matlab's version of SABR
https://www.mathworks.com/help/fininst/calibrating-the-sabr-model.html#bt9yf_t-1

K = strike
F = forward price
t or tau = ttm in fraction of days in year
alpha = sabr base vol - vol param, can be atm_vol
beta = sabr skew param = beta=0 -> skew downward sloping, beta=1 -> skew flat
rho = sabr correlation param
volvol = vol of atm volatility

parameter effects (assume beta known):
alpha: controls overall height of curve
rho: controls curve's skew
volvol: controls curve's smile

"""

import numpy as np
from scipy.optimize import minimize, newton


def calc_z(F, K, alpha, volvol, beta=1):
    """
    function for the Black vol equation
    """

    z = (volvol / alpha) * (F * K) ** (0.5 * (1 - beta)) * np.log(F / K)

    return z

def calc_chiZ(z, rho):
    """
    function for the Black vol equation
    """

    numerator = np.sqrt(1 - 2*rho*z + z**2) + z - rho

    return np.log(numerator / (1-rho))

def calc_volBlack(F, K, tau, alpha, rho, volvol, beta=1):
    """
    Black volatility
    """

    z = calc_z(F, K, alpha, volvol, beta=beta)
    chi_z = calc_chiZ(z, rho)
    num1 = (alpha**2 * (1-beta)**2) / (24 * (F * K) ** (1 - beta))
    num1 += 0.25 * rho * beta * volvol * alpha / (F * K) ** (0.5 * (1 - beta))
    num1 += volvol**2*(2 - 3*rho**2) / 24
    numerator = alpha * (1 + tau * num1)

    denom1 = ((1-beta) ** 2 * np.log(F / K) ** 2) / 24
    denom1 += ((1-beta) ** 4 * np.log(F / K) ** 4) / 1920
    denominator = (F * K) ** (0.5 * (1 - beta)) * (1 + denom1)

    return (numerator / denominator) * z / chi_z

def calc_atmVol(K, tau, alpha, rho, volvol, beta=1):

    return calc_volBlack(K, K, tau, alpha, rho, volvol, beta=beta)

def calc_atmVol2SabrAlpha(F, tau, atmvol, rho, volvol, beta=1):
    """
    https://www.mathworks.com/help/fininst/calibrating-the-sabr-model.html#bt9yf_u-1
    formula based on matlab equation
    """
    c0 = (1-beta)**tau / (24*F**(2-2*beta))
    c1 = (rho * beta * volvol * tau) / (4*F**(1-beta))
    c2 = 1 + tau * volvol**2 * (2-3*rho**2) / 24
    c3 = -F**(1-beta) * atmvol
    # create a list of polynomial coefficients
    coef = [c0, c1, c2, c3]
    # print("coef: %s" %(coef))
    # select min real roots
    try:
        alpha = newton(np.poly1d(coef), 1, maxiter=100, tol=0.0001)
    except:
        print("failed to converge")
        alpha = atmvol

    return np.min(alpha)

def calc_atmVol2SabrAlphaRhoNu(mkt_vols, atmvol, F, K, tau, beta=1.0, initial_guess=[0.5, 0.1]):
    """
    uses the atmVol to substitue for alpha to estimate for optimal rho and nu(volvol)
    optimal rho and nu then used to find alpha
    x is variable solved for in optimization (rho, volvol)
    mkt_vols: array of implied market volatilites

    :returns: alpha, rho, nu
    """
    # objective function - minimize sum squared errors
    def obj_func(x, mkt_vols, atmvol, F, K, tau, beta):
        rho, volvol = x

        # objective function to minimize
        alpha_atm = calc_atmVol2SabrAlpha(F, tau, atmvol, rho, volvol, beta=beta)
        # print("alpha_atm: %s" %(alpha_atm))
        black_vols = calc_volBlack(F, K, tau, alpha_atm, rho, volvol, beta=beta)
        return np.sum((mkt_vols - black_vols)**2)

    # minimize solver [1, 0.1] is initial guess for rho and volvol
    bounds = [(-0.9999, 0.9999), (0.0001, None)]
    x_solved = minimize(obj_func, initial_guess, args=(mkt_vols, atmvol, F, K, tau, beta),
                        method='L-BFGS-B', bounds=bounds, tol=0.00001)
    print(x_solved)
    rho, volvol = x_solved.x

    # use solved rho and volvol to calc alpha_atm
    alpha = calc_atmVol2SabrAlpha(F, tau, atmvol, rho, volvol, beta)

    return alpha, rho, volvol

def calc_sabrAlphaRhoNu(mkt_vols, F, K, tau, beta=1.0, initial_guess=[0.20, 0.5, 0.01]):
    """
    solve for alpha, rho, and nu(volvol) using mkt_vols
    x is variable solved for in optimization (alpha, rho, volvol)
    mkt_vols: array of implied market volatilites

    :returns: alpha, rho, nu
    """
    # objective function - minimize sum squared errors
    def obj_func(x, mkt_vols, F, K, tau, beta):
        alpha, rho, volvol = x
        # objective function to minimize
        black_vols = calc_volBlack(F, K, tau, alpha, rho, volvol, beta=beta)
        return np.sum((mkt_vols - black_vols)**2)

    # minimize solver [1, 0.1] is initial guess for rho and volvol
    bounds = [(0.0001, None), (-0.9999, 0.9999), (0.0001, None)]
    x_solved = minimize(obj_func, initial_guess, args=(mkt_vols, F, K, tau, beta),
                        method='L-BFGS-B', bounds=bounds, tol=0.00001)
    print(x_solved)
    alpha, rho, volvol = x_solved.x

    return alpha, rho, volvol