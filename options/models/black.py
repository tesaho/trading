"""
black's model pricing for european options
https://github.com/ynouri/pysabr/blob/master/pysabr/black.py

F = spot price
K = strike price
T = expiry in years
t = time in years
tau = T - t (in years)
r = annualized risk free rate (SHIBOR)
vol = implied volatility
q = continuous dividend yield (not used in black but for consistency for Option class)
"""
"""
black's 1976 model using forward prices
- used for european options on physical commodities, forwards, futures 

F = forward price
K = strike price
T = expiry in years
t = time in years
tau = T - t (in years)
r = annualized risk free rate (SHIBOR)
vol = implied volatility
https://github.com/vollib/py_vollib/blob/master/py_vollib/black/greeks/analytical.py
"""


import numpy as np
import py_vollib.black.implied_volatility as iv
import py_lets_be_rational as lets_be_rational
from scipy.stats import norm


class Black(object):
    def __init__(self, numDaysPerYear=365):
        self.numDaysPerYear = numDaysPerYear

    @staticmethod
    def calc_d1(F, K, r, vol, tau, q=0):

        return (np.log(F / K) + vol ** 2 * tau / 2) / (vol * tau**0.5)

    @classmethod
    def calc_d2(cls, F, K, r, vol, tau, q=0):

        d1 = cls.calc_d1(F, K, r, vol, tau)
        return d1 - vol*np.sqrt(tau)

    @staticmethod
    def calc_N(x):
        """
        N(x) = standard normal cdf
        """

        return norm.cdf(x)

    @staticmethod
    def calc_N1(x):
        """
        N'(x) = standard normal pdf
        """

        return norm.pdf(x)

    @classmethod
    def calc_call(cls, F, K, r, vol, tau, q=0):
        """
        Computes the premium for a call or put option using a lognormal vol
        """
        d1 = cls.calc_d1(F, K, r, vol, tau, q)
        d2 = cls.calc_d2(F, K, r, vol, tau, q)
        N_d1 = cls.calc_N(d1)
        N_d2 = cls.calc_N(d2)

        return np.exp(-r*tau) * (F*N_d1 - K*N_d2)


    @classmethod
    def calc_put(cls, F, K, r, vol, tau, q=0):
        """
        Computes the premium for a put option using a lognormal vol
        """
        d1 = cls.calc_d1(F, K, r, vol, tau, q)
        d2 = cls.calc_d2(F, K, r, vol, tau, q)
        N_d1 = cls.calc_N(-d1)
        N_d2 = cls.calc_N(-d2)

        return np.exp(-r*tau) * (-F*N_d1 + K*N_d2)

    @classmethod
    def calc_moneyness(cls, F, K, r, vol, tau):

        return np.log(F/K) / (vol*np.sqrt(tau))


    @classmethod
    def calc_impliedVolatility(cls, optPx, F, K, r, tau, isCall=True, q=0):

        if isCall:
            return iv.implied_volatility(optPx, F, K, r, tau, "c")

        else:
            return iv.implied_volatility(optPx, F, K, r, tau, "p")

    ## first derivs
    @classmethod
    def calc_delta(cls, F, K, r, vol, tau, isCall=True, q=0):
        """
        d_optValue / d_spot
        """

        d_1 = cls.calc_d1(F, K, r, vol, tau, q)

        if isCall:
            return np.exp(-r * tau) * cls.calc_N(d_1)
        else:
            return -np.exp(-r * tau) * cls.calc_N(-d_1)

    @classmethod
    def calc_vega(cls, F, K, r, vol, tau, isCall=True, q=0):
        """
        d_optValue / d_vol
        """

        d_1 = cls.calc_d1(F, K, r, vol, tau, q)

        # multiply by 0.01 for 1% vol move
        return (F * np.exp(-r*tau) * cls.calc_N1(d_1) * np.sqrt(tau))*0.01


    def calc_theta(self, F, K, r, vol, tau, isCall=True, q=0):
        """
        d_optValue / d_tau
        """

        d_1 = self.calc_d1(F, K, r, vol, tau, q)
        d_2 = self.calc_d2(F, K, r, vol, tau, q)
        pdf_d1 = self.calc_N1(d_1)

        first_term = F * np.exp(-r*tau) * pdf_d1 * vol / (2*np.sqrt(tau))

        if isCall:
            second_term = -r * F * np.exp(-r*tau) * self.calc_N(d_1)
            third_term = r * K * np.exp(-r*tau) * self.calc_N(d_2)
        else:
            second_term = -r * F * np.exp(-r * tau) * self.calc_N(-d_1)
            third_term = r * K * np.exp(-r*tau) * self.calc_N(-d_2)

        return (-first_term + second_term + third_term)/self.numDaysPerYear


    @classmethod
    def calc_rho(cls, F, K, r, vol, tau, isCall=True, q=0):
        """
        d_optValue / d_r
        multiply by 0.01 since d_r = 1% change in r
        """

        # multiply by 0.01 for 1% rho move
        if isCall:
            return (-tau * cls.calc_call(F, K, r, vol, tau, q)) * 0.01
        else:
            return (-tau * cls.calc_put(F, K, r, vol, tau, q)) * 0.01

    ## 2nd derivs of spot
    @classmethod
    def calc_gamma(cls, F, K, r, vol, tau, isCall=True, q=0):
        """
        d_delta / d_spot
        """

        d_1 = cls.calc_d1(F, K, r, vol, tau, q)
        return cls.calc_N1(d_1) * np.exp(-r*tau) / (F*vol*np.sqrt(tau))


    def calc_vanna2(self, F, K, r, vol, tau, isCall=True, q=0):
        """
        d_vega / d_spot
        """

        d_1 = self.calc_d1(F, K, r, vol, tau, q)
        vega = self.calc_vega(F, K, r, vol, tau, q=q)

        return (vega/F)*(1 - d_1/(vol*np.sqrt(tau)))

    @classmethod
    def calc_vanna(cls, F, K, r, vol, tau, isCall=True, q=0):
        """
        d_vega / d_spot
        """
        d_1 = cls.calc_d1(F, K, r, vol, tau, q)
        d_2 = cls.calc_d2(F, K, r, vol, tau, q)

        # multiply by 0.01 for 1% vol move
        return -np.exp(-r*tau) * cls.calc_N1(d_1)* d_2 / vol


    def calc_charm(self, F, K, r, vol, tau, isCall=True, q=0):
        """
        delta decay: d_delta / d_tau
        """

        d_1 = self.calc_d1(F, K, r, vol, tau, q)
        d_2 = self.calc_d2(F, K, r, vol, tau, q)

        # divide by numDaysPerYear to get delta decay per day
        if isCall:
            first_term = r * np.exp(-r * tau) * self.calc_N(d_1)
            second_term = np.exp(-r * tau) * self.calc_N1(d_1) * -d_2*vol*np.sqrt(tau)
            return (first_term - second_term / (2*tau*vol*np.sqrt(tau))) / self.numDaysPerYear
        else:
            first_term = -r * np.exp(-r * tau) * self.calc_N(-d_1)
            second_term = np.exp(-r*tau) * self.calc_N1(d_1) * - d_2*vol*np.sqrt(tau)
            return (first_term - second_term / (2*tau*vol*np.sqrt(tau))) / self.numDaysPerYear

    ## 2nd derivs of vol
    @classmethod
    def calc_vomma(cls, F, K, r, vol, tau, isCall=True, q=0):
        """
        vega convexity: d_vega / d_vol
        """

        d_1 = cls.calc_d1(F, K, r, vol, tau, q)
        d_2 = cls.calc_d2(F, K, r, vol, tau, q)
        vega = cls.calc_vega(F, K, r, vol, tau, q=q)

        # multiply by 0.01 for 1% vol move
        return vega*d_1*d_2 / vol


    def calc_veta(self, F, K, r, vol, tau, isCall=True, q=0):
        """
        vega decay: d_vega / d_tau
        """

        d_1 = self.calc_d1(F, K, r, vol, tau, q)
        d_2 = self.calc_d2(F, K, r, vol, tau, q)

        first_term = F * np.exp(-r*tau) * self.calc_N1(d_1) * np.sqrt(tau)
        second_term = r - (1+d_1*d_2)/(2*tau)

        # divide by 100*numDaysPerYear to get vega decay per day
        return first_term*second_term / (100*self.numDaysPerYear)

    ## 3rd derivs
    @classmethod
    def calc_speed(cls, F, K, r, vol, tau, isCall=True, q=0):
        """
        change gamma wrt change spot: d_gamma / d_spot
        """

        d_1 = cls.calc_d1(F, K, r, vol, tau, q)
        gamma = cls.calc_gamma(F, K, r, vol, tau, isCall=isCall, q=q)

        return -gamma/F * (d_1/(vol*np.sqrt(tau)) + 1)

    @classmethod
    def calc_zomma(cls, F, K, r, vol, tau, isCall=True, q=0):
        """
        change gamma wrt change vol: d_gamma / d_vol
        """

        d_1 = cls.calc_d1(F, K, r, vol, tau, q)
        d_2 = cls.calc_d2(F, K, r, vol, tau, q)
        gamma = cls.calc_gamma(F, K, r, vol, tau, isCall=isCall, q=q)

        return gamma * (d_1 * d_2 - 1) / vol


    def calc_color(self, F, K, r, vol, tau, isCall=True, q=0):
        """
        gamma decay: d_gamma / d_tau
        """

        d_1 = self.calc_d1(F, K, r, vol, tau, q)
        d_2 = self.calc_d2(F, K, r, vol, tau, q)

        first_term = -np.exp(-r*tau) * self.calc_N1(d_1)/(2*F*tau*vol*np.sqrt(tau))
        second_term = 2*r*tau + 1 + -d_1 * d_2

        return first_term * second_term / self.numDaysPerYear