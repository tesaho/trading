"""
calculate the option dividend yield per atm strikes per day
- winsorize each strike
- average the strike yield
- boxplot each day yield range

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import helpers.step_functions as sf
from scipy.stats import mstats

# import call and put mids
def getLogFile(logFile, startTime="09:30:00.000"):

    sdf = pd.read_csv(logFile, index_col=0)
    sdf.index = pd.to_datetime(sdf.index).strftime("%H:%M:%S.%f")
    sdf = sdf[sdf.index > startTime]

    return sdf

def getDividendYield(C, P, K, S, r, tau):

    return -1/tau * np.log( (C-P + K*np.exp(-r*tau)) / S)

def getDailyStrikeDividendYield(date, logPath="~/logs/", winsorize=False):
    """
    date: 0620 (month-day)
    """

    # get strike log
    sdf = getLogFile("%s/strikes_%s_1809.csv" %(logPath, date))

    # get dividend yield log
    div_df = getLogFile("%s/div_yield_%s_1809.csv" %(logPath, date))
    rate = div_df.rate.iloc[0]
    tau = div_df.tau.iloc[0]

    # get spot and add to strike df
    spot_sf = sf.makeStepFunction(div_df.index.values, div_df.spot.values)
    # print("spot sf ", spot_sf)
    sdf.loc[:, "spot"] = spot_sf.getValue(sdf.index.values)
    strikes = sdf.strike.unique()
    strikes.sort()
    # print("strikes ", strikes)

    # calc dy per strike
    dy_dict = {}
    for K in strikes:
        df = sdf[sdf.strike == K]
        dy = getDividendYield(df.call_mid, df.put_mid, K, df.spot, rate, tau)
        dy_dict[K] = dy

    dy_df = pd.DataFrame(dy_dict, index=df.index.values)
    # print(dy_df)

    if winsorize:
        dy_df = getWinsorized(dy_df)

    # plot graph
    dy_df.plot()
    plt.title(date)

    # print stats
    print("%s stats: " %(date))
    print(dy_df.describe())

    return dy_df

# winsorize
def getWinsorized(df, upQuantile=0.05, downQuantile=0.05):

    wins_data = mstats.winsorize(df, limits=[downQuantile, upQuantile])
    return pd.DataFrame(data=wins_data, columns=df.columns, index=df.index)
