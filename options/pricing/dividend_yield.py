"""
class to calculate curve dividend yield for a term
"""

import numpy as np


def calcDividendYield(S, K, r, tau, C, P):

    return -(1 / tau) * np.log( (C - P + K * np.exp(-r * tau)) / S)

def getDividendYieldCurve(calls, puts, S, r, tau):

    dy_curve = {}

    for i in range(len(calls)):
        K = calls.strike.iloc[i]
        if K in puts.strike.values:
            C = calls[calls.strike == K].mid.iloc[0]
            P = puts[puts.strike == K].mid.iloc[0]
            dy = calcDividendYield(S, K, r, tau, C, P)
            dy_curve[K] = dy
        else:
            print ("skipping strike %s" %(K))

    return dy_curve




