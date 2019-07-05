"""
option class to map options to groups
"""

import pandas as pd
# from helpers.instruments import Instrument
from options.pricing.option_greeks import Option
from options.option_class.option_class import option_class

class Options(option_class):
    def __init__(self, product, exchange):
        self.product = product
        self.exchange = exchange
        self.strikeList = []
        # initiate the underlying option_class
        super().__init__(product, exchange)


    def initOptionPricer(self, r, tau, numDaysPerYear, modelName="black_scholes"):
        self.r = r
        self.tau = tau
        self.modelName = modelName
        self.numDaysPerYear = numDaysPerYear
        self.model = Option(self.product, self.exchange, self.numDaysPerYear,\
                                modelName=self.modelName)

    def getOptions(self, ATM_strike, numStrikes=5, strikeInterval=0.05):
        self._getStrikeList(ATM_strike, numStrikes=numStrikes, strikeInterval=strikeInterval)
        self._getOptionShortNames()
        return self.options

    def getOtmOptionVols(self, spot, optPrices, strikeList, q=0):

        strikeImpVol = {}

        for i in range(len(strikeList)):
            K = strikeList[i]

            if K < spot:
                optName = [x for x in optPrices.index if "P%s" %K in x]
                if len(optName) > 0:
                    optPx = optPrices[optPrices.index == optName[0]].iloc[0]
                    # print(K, optName, optPx)
                    impVol = self.model.getImpliedVol(optPx, spot, K, self.r, self.tau, isCall=False, q=q)
                    strikeImpVol["%sP" %K] = impVol
            else:
                optName = [x for x in optPrices.index if "C%s" %K in x]
                if len(optName) > 0:
                    optPx = optPrices[optPrices.index == optName[0]].iloc[0]
                    # print(K, optName, optPx)
                    impVol = self.model.getImpliedVol(optPx, spot, K, self.r, self.tau, isCall=True, q=q)
                    strikeImpVol["%sC" %K] = impVol

        return strikeImpVol

    def getImpliedRate(self, spot, strike, callPx, putPx, tau):
        """
        C + K/(1+r)^t = S + P
        r = (K / (S + P - C))**1/t - 1

        """
        r = (strike / (spot + putPx - callPx))**(1/tau)-1
        return r


    def _getStrikeList(self, ATM_strike, numStrikes=5, strikeInterval=0.05):
        strikes = [round(ATM_strike, 4)]
        strikes += [round(ATM_strike + strikeInterval*i, 4) for i in range(1, numStrikes+1)]
        strikes += [round(ATM_strike - strikeInterval*i, 4) for i in range(1, numStrikes+1)]
        strikes.sort()
        self.strikeList = strikes

    def _getOptionShortNames(self):
        self.calls = ["%sC" %(x) for x in self.strikeList]
        self.puts = ["%sP" %(x) for x in self.strikeList]
        self.options = self.calls + self.puts

    def getPortOptionGreeks(self, spot, volDict, q=0):
        """
        :param volDict: {"285C":impVol}
        :return:
        """
        greekList = []

        for key in volDict.keys():
            if key[-1] == "C" or key[-1] == "P":
                K = float(key[:-1])
                optType = key[-1]
            else:
                K = float(key[1:])
                optType = key[0]

            vol = volDict[key]
            if optType == "C":
                greekList.append(self.getOptionGreeks(spot, K, vol, True, q))
            else:
                greekList.append(self.getOptionGreeks(spot, K, vol, False, q))

        greeks = pd.concat(greekList)


        return greeks

    def getPortOptionGreeksArrays(self, spot, optNames, strikes, impVols, q=0):
        """
        :param use arrays optNames, strikes, impVols
        :return:
        """
        greekList = []

        for i in range(len(optNames)):
            K = strikes[i]
            vol = impVols[i]
            if "C" in optNames[i]:
                greekList.append(self.getOptionGreeks(spot, K, vol, True, q))
            else:
                greekList.append(self.getOptionGreeks(spot, K, vol, False, q))

        greeks = pd.concat(greekList)
        greeks["optionName"] = optNames
        greeks.reset_index(drop=True, inplace=True)
        greeks.set_index("optionName", inplace=True)

        return greeks

    def getOptionGreeks(self, spot, K, vol, isCall=True, q=0):

        greeks = self.model.getOptionGreeks(spot, K, self.r, vol, self.tau, isCall, q=q)

        return greeks

    def getPortOptionTV(self, spot, volDict, q=0):

        tvList = []

        for key in volDict.keys():
            if key[-1] == "C" or key[-1] == "P":
                K = float(key[:-1])
                optType = key[-1]
            else:
                K = float(key[1:])
                optType = key[0]
            vol = volDict[key]
            if optType == "C":
                tvList.append(self.getOptionTV(spot, K, vol, isCall=True, q=q))
            else:
                tvList.append(self.getOptionTV(spot, K, vol, isCall=False, q=q))

        optTvs = pd.DataFrame({"optTv":tvList}, index=list(volDict.keys()))

        return optTvs

    def getOptionTV(self, spot, K, vol, isCall=True, q=0):

        return self.model.getTheoValue(spot, K, self.r, vol, self.tau, isCall=isCall, q=q)

    def getPortIntrinsicPremiumValues(self, spot, volDict):

        intrinsicList = []
        premiumList = []

        for key in volDict.keys():
            if key[-1] == "C" or key[-1] == "P":
                K = float(key[:-1])
                optType = key[-1]
            else:
                K = float(key[1:])
                optType = key[0]
            vol = volDict[key]
            if optType == "C":
                intrinsicList.append(self.model.getIntrinsicValue(spot, K, isCall=True))
                premiumList.append(self.model.getPremium(spot, K, self.r, vol, self.tau, isCall=True))
            else:
                intrinsicList.append(self.model.getIntrinsicValue(spot, K, isCall=False))
                premiumList.append(self.model.getPremium(spot, K, self.r, vol, self.tau, isCall=False))

        optIntPrem = pd.DataFrame({"optIntValue": intrinsicList, "optPremium":premiumList}, \
                              index=list(volDict.keys()))

        return optIntPrem


    def getPortTvGreeks(self, spot, volDict):

        optTvs = self.getPortOptionTV(spot, volDict)
        optIntPrem = self.getPortIntrinsicPremiumValues(spot, volDict)
        greeks = self.getPortOptionGreeks(spot, volDict)

        portDf = pd.concat([optTvs.T, optIntPrem.T, greeks.T]).T

        return portDf

