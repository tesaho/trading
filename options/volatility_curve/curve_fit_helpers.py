"""
helpers for fitting options
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import research.option_implied_yield as oiy
from options.pricing.option_greeks import Option

# return otm put deltas
def getPutDeltas(delta, optType):
    """
    delta: array or list of deltas
    optType: array or list of optType "C", "P"
    :return:
    """
    # otm_x = put deltas
    otm_x = []
    for i in range(len(delta)):
        if optType[i] == "C":
            otm_x.append(1-delta[i])
        else:
            otm_x.append(abs(delta[i]))

    return otm_x

def getAtmOptionImpliedDividendYield(df, spot, r, tau):

    atm_up = df[df.strike / spot > 1.0].iloc[0]
    atm_down = df[df.strike / spot < 1.0].iloc[-1]

    atm_up_strike = atm_up.strike
    dy_up = oiy.getDividendYield(df[(df.strike == atm_up_strike) & (df.optType == "C")].mid.iloc[0],
                                 df[(df.strike == atm_up_strike) & (df.optType == "P")].mid.iloc[0],
                                 atm_up_strike, spot, r, tau)

    atm_down_strike = atm_down.strike
    dy_down = oiy.getDividendYield(df[(df.strike == atm_down_strike) & (df.optType == "C")].mid.iloc[0],
                                 df[(df.strike == atm_down_strike) & (df.optType == "P")].mid.iloc[0],
                                 atm_down_strike, spot, r, tau)

    return dy_up, dy_down

def getOptionImpliedDividendYields(df, spot, r, tau):

    strikes = df.strike.unique()
    strikes.sort()
    calls = df[df.optType == "C"]
    puts = df[df.optType == "P"]

    div_yields = {}

    for strike in strikes:
        dy = oiy.getDividendYield()

    atm_up_strike = atm_up.strike
    dy_up = oiy.getDividendYield(df[(df.strike == atm_up_strike) & (df.optType == "C")].mid.iloc[0],
                                 df[(df.strike == atm_up_strike) & (df.optType == "P")].mid.iloc[0],
                                 atm_up_strike, spot, r, tau)

    atm_down_strike = atm_down.strike
    dy_down = oiy.getDividendYield(df[(df.strike == atm_down_strike) & (df.optType == "C")].mid.iloc[0],
                                 df[(df.strike == atm_down_strike) & (df.optType == "P")].mid.iloc[0],
                                 atm_down_strike, spot, r, tau)

    return dy_up, dy_down


def getAtmVol(strikes, spot, vols):
    """
    strikes:
    spot:
    vols:
    """
    atm_up_idx = np.where(strikes / spot > 1.0)[0][0]
    atm_down_idx = np.where(strikes / spot < 1.0)[0][-1]
    if vols[atm_up_idx] < vols[atm_down_idx]:
        atmVol = vols[atm_up_idx]
        atmStrike = strikes[atm_up_idx]
    else:
        atmVol = vols[atm_down_idx]
        atmStrike = strikes[atm_down_idx]

    return atmVol, atmStrike

def getVols(df, spot, r, tau, q, opt=Option("SSE50_OPT", "SSE", 365, modelName="black_scholes")):
    # calculate vols
    new_df = df.copy()

    midVols = []
    intrinsicValues = []
    moneyness = []
    deltas = []

    for i in range(len(new_df)):

        if new_df.optType.iloc[i] == "C":
            isCall = True
        else:
            isCall = False

        intrinsicValues.append(opt.getIntrinsicValue(spot, new_df.strike.iloc[i], isCall))

        ## if optBid >= intrinsicValue, calc bidVol
        if new_df.optBid.iloc[i] >= intrinsicValues[-1]:
            new_df.bidVol.iloc[i] = opt.getImpliedVol(new_df.optBid.iloc[i], spot, new_df.strike.iloc[i],
                                                      r, tau, isCall, q)
        # or optPx = IV
        else:
            new_df.bidVol.iloc[i] = opt.getImpliedVol(intrinsicValues[-1], spot, new_df.strike.iloc[i],
                                                      r, tau, isCall, q)

        # if optAsk >= IV, calc askVol
        if new_df.optAsk.iloc[i] >= intrinsicValues[-1]:
            new_df.askVol.iloc[i] = opt.getImpliedVol(new_df.optAsk.iloc[i], spot, new_df.strike.iloc[i],
                                                      r, tau, isCall, q)

        # if cross weighted avg price >= IV, recalc imp vol
        if new_df.cwap.iloc[i] >= intrinsicValues[-1]:
            new_df.impVol.iloc[i] = opt.getImpliedVol(new_df.cwap.iloc[i], spot, new_df.strike.iloc[i],
                                                      r, tau, isCall, q)
        # if mid >= IV, recalc mid vol
        if new_df.mid.iloc[i] >= intrinsicValues[-1]:
            midVols.append(opt.getImpliedVol(new_df.mid.iloc[i], spot, new_df.strike.iloc[i],
                                             r, tau, isCall, q))
        # if mid < IV, but bidVol and askVol exists then average bidVol and askVol
        elif (~np.isnan(new_df.bidVol.iloc[i]) and ~ np.isnan(new_df.askVol.iloc[i])):
            midVols.append((new_df.bidVol.iloc[i] + new_df.askVol.iloc[i]) / 2.0)
        else:
            midVols.append(100)

        # if impVol is nan (px < IV), then impVol = midVol
        if np.isnan(new_df.impVol.iloc[i]):
            new_df.impVol.iloc[i] = midVols[i]

        # calc moneyness using impVol if it exists
        if ~np.isnan(new_df.impVol.iloc[i]):
            moneyness.append(opt.getMoneyness(spot, new_df.strike.iloc[i], r, new_df.impVol.iloc[i], tau))
        # or use midVol
        else:
            moneyness.append(opt.getMoneyness(spot, new_df.strike.iloc[i], r, new_df.midVol.iloc[i], tau))

        # calc deltas using impVol
        deltas.append(opt.getDelta(spot, new_df.strike.iloc[i], r, new_df.impVol.iloc[i], tau, isCall, q))

    new_df["delta"] = deltas
    new_df["midVol"] = midVols
    new_df["intrinsic"] = intrinsicValues
    new_df["moneyness"] = moneyness
    new_df["baVolSpread"] = new_df.askVol - new_df.bidVol

    # drop any vols that can't be calculated
    new_df.dropna(subset=["bidVol", "askVol"], inplace=True)

    return new_df

def getOtmOptions(new_df, atm_strike):

    otm_calls = new_df[(new_df.optType == "C") & (new_df.strike >= atm_strike)]
    otm_puts = new_df[(new_df.optType == "P") & (new_df.strike < atm_strike)]
    otm = pd.concat([otm_puts, otm_calls])

    return otm

def plotImpliedVolatilities(calls, puts, otm):
    otm_x = getPutDeltas(otm.delta, otm.optType)

    fig = plt.figure()
    plt.plot(1 - calls.delta.values, calls.midVol.values, "bx", label='calls')
    plt.plot(1 - calls.delta.values, calls.askVol.values, "bv", label="calls_ask")
    plt.plot(1 - calls.delta.values, calls.bidVol.values, "b^", label="calls_bid")
    plt.plot(abs(puts.delta.values), puts.midVol.values, "rx", label="puts")
    plt.plot(abs(puts.delta.values), puts.askVol.values, "rv", label="puts_ask")
    plt.plot(abs(puts.delta.values), puts.bidVol.values, "r^", label="puts_bid")
    plt.plot(otm_x, otm.midVol.values, "c^", label="otmVol")
    plt.legend()
    plt.show()
