"""
bsm greek calculation - extends bs with dividend yield
S = spot price
K = strike price
T = expiry in years
t = time in years
tau = T - t (in years)
r = annualized risk free rate (SHIBOR)
vol = implied volatility
q = continuous dividend yield
"""

import numpy as np
from scipy.stats import norm
from py_lets_be_rational import implied_volatility_from_a_transformed_rational_guess as iv

class BlackScholesMerton(object):
    def __init__(self, numDaysPerYear=365):
        self.numDaysPerYear = numDaysPerYear

    @staticmethod
    def calc_d1(S, K, r, vol, tau, q=0):

        return (np.log(S/K) + (r - q +0.5*vol**2)*(tau)) / (vol*np.sqrt(tau))

    @classmethod
    def calc_d2(cls, S, K, r, vol, tau, q=0):

        d1 = cls.calc_d1(S, K, r, vol, tau, q)
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
    def calc_call(cls, S, K, r, vol, tau, q=0):

        d1 = cls.calc_d1(S, K, r, vol, tau, q)
        d2 = cls.calc_d2(S, K, r, vol, tau, q)
        N_d1 = cls.calc_N(d1)
        N_d2 = cls.calc_N(d2)

        return N_d1*S*np.exp(-q*tau) - N_d2*K*np.exp(-r*tau)

    @classmethod
    def calc_put(cls, S, K, r, vol, tau, q=0):

        d1 = cls.calc_d1(S, K, r, vol, tau, q)
        d2 = cls.calc_d2(S, K, r, vol, tau, q)

        return np.exp(-r*tau)*K*cls.calc_N(-d2) - S*np.exp(-q*tau)*cls.calc_N(-d1)

    @staticmethod
    def calc_discount_factor(r, tau):

        return np.exp(-r*tau)

    @staticmethod
    def calc_forward(S, r, tau, q=0):
        """
        q = dividend rate
        """

        return S*np.exp((r-q)*tau)

    def calc_forward_discrete_dividend(S, r, D, tau):
        """
        D = cumulative discounted dividend rate
        """

        return (S-D)*np.exp(r*tau)

    @classmethod
    def calc_moneyness(cls, S, K, r, vol, tau):

        return np.log(S/K) / (vol*np.sqrt(tau))

    @classmethod
    def calc_impliedVolatility(cls, optPx, F, K, r, tau, isCall=True, q=0):
        """
        F -> forward rate: F = S*exp((r-q) * tau)
        """

        undiscounted_option_price = optPx / np.exp(-r*tau)

        if isCall:
            sigma = iv(undiscounted_option_price, F, K, tau, 1)
        else:
            sigma = iv(undiscounted_option_price, F, K, tau, -1)

        if sigma >= 100:
            print("sigma exceeds max vol")
            raise Exception
        if sigma <= -100:
            print("sigma below intrinsic value")
            raise Exception

        return sigma

    ## first derivs
    @classmethod
    def calc_delta(cls, S, K, r, vol, tau, isCall=True, q=0):
        """
        d_optValue / d_spot
        """

        d_1 = cls.calc_d1(S, K, r, vol, tau, q)

        if isCall:
            return cls.calc_N(d_1) * np.exp(-q*tau)
        else:
            return (cls.calc_N(d_1) - 1.0) * np.exp(-q*tau)

    @classmethod
    def calc_vega(cls, S, K, r, vol, tau, isCall=True, q=0):
        """
        d_optValue / d_vol
        multiply by 0.01 for 1% change in volatility
        """

        d_1 = cls.calc_d1(S, K, r, vol, tau, q)

        # multiply by 0.01 for 1% vol move
        return (S*np.exp(-q*tau)*cls.calc_N1(d_1) * np.sqrt(tau))*0.01


    def calc_theta(self, S, K, r, vol, tau, isCall=True, q=0):
        """
        d_optValue / d_tau
        """

        d_1 = self.calc_d1(S, K, r, vol, tau, q)
        d_2 = self.calc_d2(S, K, r, vol, tau, q)
        first_term = -np.exp(-q*tau)*S*self.calc_N1(d_1)*vol/(2*np.sqrt(tau))

        if isCall:
            second_term = -r*K*np.exp(-r*tau)*self.calc_N(d_2)
            third_term = q * S*np.exp(-q*tau)*self.calc_N(d_1)
        else:
            second_term = r*K*np.exp(-r*tau)*self.calc_N(-d_2)
            third_term = -q * S*np.exp(-q*tau)*self.calc_N(-d_1)

        return (first_term + second_term + third_term)/self.numDaysPerYear

    @classmethod
    def calc_rho(cls, S, K, r, vol, tau, isCall=True, q=0):
        """
        d_optValue / d_r
        multiply by 0.01 since d_r = 1% change in r
        """

        d_2 = cls.calc_d2(S, K, r, vol, tau, q)

        # multiply by 0.01 for 1% rho move
        if isCall:
            return (K*tau*np.exp(-r*tau)*cls.calc_N(d_2))*0.01
        else:
            return (-K*tau*np.exp(-r*tau)*cls.calc_N(-d_2))*0.01

    ## 2nd derivs of spot
    @classmethod
    def calc_gamma(cls, S, K, r, vol, tau, isCall=True, q=0):
        """
        d_delta / d_spot
        """

        d_1 = cls.calc_d1(S, K, r, vol, tau, q)
        return np.exp(-q*tau)*cls.calc_N1(d_1) / (S*vol*np.sqrt(tau))


    def calc_vanna2(self, S, K, r, vol, tau, isCall=True, q=0):
        """
        d_vega / d_spot
        """

        d_1 = self.calc_d1(S, K, r, vol, tau, q)
        vega = self.calc_vega(S, K, r, vol, tau, q)

        return (vega/S)*(1 - d_1/(vol*np.sqrt(tau)))

    @classmethod
    def calc_vanna(cls, S, K, r, vol, tau, isCall=True, q=0):
        """
        d_vega / d_spot
        """
        d_1 = cls.calc_d1(S, K, r, vol, tau, q)
        d_2 = cls.calc_d2(S, K, r, vol, tau, q)

        # multiply by 0.01 for 1% vol move
        return -np.exp(-q*tau)*cls.calc_N1(d_1)* d_2 / vol

    def calc_charm(self, S, K, r, vol, tau, isCall=True, q=0):
        """
        delta decay: d_delta / d_tau
        """

        d_1 = self.calc_d1(S, K, r, vol, tau, q)
        d_2 = self.calc_d2(S, K, r, vol, tau, q)

        # divide by numDaysPerYear to get delta decay per day
        if isCall:
            first_term = q * np.exp(-q*tau) * self.calc_N(d_1)
            second_term = np.exp(-q*tau) * self.calc_N1(d_1) * (2*(r-q)*tau - d_2*vol*np.sqrt(tau)) / (2 * tau * vol * np.sqrt(tau))
            return (first_term - second_term) / self.numDaysPerYear
        else:
            first_term = -q * np.exp(-q*tau) * self.calc_N(-d_1)
            second_term = np.exp(-q*tau) * self.calc_N1(d_1) * (2*(r-q)*tau - d_2*vol*np.sqrt(tau)) / (2*tau*vol*np.sqrt(tau))
            return (first_term - second_term) / self.numDaysPerYear

    ## 2nd derivs of vol
    @classmethod
    def calc_vomma(cls, S, K, r, vol, tau, isCall=True, q=0):
        """
        vega convexity: d_vega / d_vol
        """

        d_1 = cls.calc_d1(S, K, r, vol, tau, q)
        d_2 = cls.calc_d2(S, K, r, vol, tau, q)
        vega = cls.calc_vega(S, K, r, vol, tau, q)

        # multiply by 0.01 for 1% vol move
        return vega*d_1*d_2 / vol


    def calc_veta(self, S, K, r, vol, tau, isCall=True, q=0):
        """
        vega decay: d_vega / d_tau
        """

        d_1 = self.calc_d1(S, K, r, vol, tau, q)
        d_2 = self.calc_d2(S, K, r, vol, tau, q)

        first_term = S*np.exp(-q*tau)*self.calc_N1(d_1)*np.sqrt(tau)
        second_term = q + (r-q)*d_1/(vol*np.sqrt(tau)) - (1+d_1*d_2)/(2*tau)

        # divide by 100*numDaysPerYear to get vega decay per day
        return first_term*second_term / (100*self.numDaysPerYear)

    ## 3rd derivs
    @classmethod
    def calc_speed2(cls, S, K, r, vol, tau, isCall=True, q=0):
        """
        change gamma wrt change spot: d_gamma / d_spot
        """

        d_1 = cls.calc_d1(S, K, r, vol, tau, q)
        gamma = cls.calc_gamma(S, K, r, vol, tau, q)
        second_term = d_1 / (vol*np.sqrt(tau)) + 1

        return -gamma/S * second_term

    @classmethod
    def calc_speed(cls, S, K, r, vol, tau, isCall=True, q=0):

        d_1 = cls.calc_d1(S, K, r, vol, tau, q)
        first_term = -np.exp(-q*tau) * cls.calc_N1(d_1) / (S**2 * vol * np.sqrt(tau))
        second_term = d_1 / (vol*np.sqrt(tau)) + 1

        return first_term * second_term

    @classmethod
    def calc_zomma(cls, S, K, r, vol, tau, isCall=True, q=0):
        """
        change gamma wrt change vol: d_gamma / d_vol
        """

        d_1 = cls.calc_d1(S, K, r, vol, tau, q)
        d_2 = cls.calc_d2(S, K, r, vol, tau, q)
        gamma = cls.calc_gamma(S, K, r, vol, tau, q)
        # multiply by 0.01 for 1% vol move
        return gamma*(d_1*d_2 - 1) / vol


    def calc_color(self, S, K, r, vol, tau, isCall=True, q=0):
        """
        gamma decay: d_gamma / d_tau
        """

        d_1 = self.calc_d1(S, K, r, vol, tau, q)
        d_2 = self.calc_d2(S, K, r, vol, tau, q)

        first_term = -np.exp(-q*tau)*self.calc_N1(d_1)/(2*S*tau*vol*np.sqrt(tau))
        second_term = 2*q*tau + 1 + d_1*(2*(r-q)*tau - d_2*vol*np.sqrt(tau))/(vol*np.sqrt(tau))

        return first_term * second_term * (-1/self.numDaysPerYear)