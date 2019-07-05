"""
class to get option greeks

S = spot price
K = strike price
T = expiry in years
t = time in years
tau = T - t (in years)
r = annualized risk free rate (SHIBOR)
vol = implied volatility
"""

import pandas as pd
from options.models.black_scholes_merton import BlackScholesMerton
from options.models.black_scholes import BlackScholes
from options.models.black import Black
from options.option_class.option_class import option_class

class Option(option_class):
    def __init__(self, product, exchange, numDaysPerYear, modelName="black_scholes"):
        self.product = product
        self.exchange = exchange
        self.modelName = modelName
        self._initializeModel(numDaysPerYear)
        # initiate the underlying option_class
        super().__init__(product, exchange)

    def _initializeModel(self, numDaysPerYear):
        if self.modelName == "black_scholes":
            self.model = BlackScholes(numDaysPerYear=numDaysPerYear)
        elif self.modelName == "black":
            self.model = Black(numDaysPerYear=numDaysPerYear)
        else:
            self.model = BlackScholesMerton(numDaysPerYear=numDaysPerYear)

    def getTheoValue(self, S, K, r, vol, tau, isCall, q):
        if isCall:
            self.tv = self.model.calc_call(S, K, r, vol, tau, q)
        else:
            self.tv = self.model.calc_put(S, K, r, vol, tau, q)
        return self.tv

    def getIntrinsicValue(self, S, K, isCall):

        self.intrinsic_value = 0.0
        if isCall:
            if S > K:
                self.intrinsic_value = S - K
        else:
            if S < K:
                self.intrinsic_value = K - S

        return self.intrinsic_value

    def getPremium(self, S, K, r, vol, tau, isCall, q):

        tv = self.getTheoValue(S, K, r, vol, tau, isCall, q)
        intrinsic_value = self.getIntrinsicValue(S, K, isCall)
        self.premium = tv - intrinsic_value

        return self.premium

    def getMoneyness(self, S, K, r, vol, tau):
        self.moneyness = self.model.calc_moneyness(S, K, r, vol, tau)
        return self.moneyness

    def getImpliedVol(self, optPx, S, K, r, tau, isCall, q):

        try:
            impVol = self.model.calc_impliedVolatility(optPx, S, K, r, tau, isCall, q)
        except:
            # print("option price below intrinsic value. using intrinsic value to calc impVol")
            # implement Newton's method
            intrinsic = self.getIntrinsicValue(S, K, isCall)
            impVol = self.model.calc_impliedVolatility(intrinsic, S, K, r, tau, isCall, q)
            # impVol = self.newtonsMethod(arbVol, optPx, S, K, r, tau, isCall)

        return impVol

    # newton's method for calculating implied volatility below intrinsic
    def newtonsMethod(self, impVolGuess, optPx, S, K, r, tau, isCall, precision=1.0e-5,
                      max_iterations=100):

        px = self.getTheoValue(S, K, r, impVolGuess, tau, isCall)

        for i in range(max_iterations):
            vega = self.model.calc_vega(S, K, r, impVolGuess, tau, isCall)
            diff = optPx - px

            if abs(diff) < precision:
                return impVolGuess
            impVolGuess += diff/vega * self.optTickIncr
            px = self.getTheoValue(S, K, r, impVolGuess, tau, isCall)
            print("new vol guess: %s optPx: %s diff: %s" %(impVolGuess, px, abs(optPx-px)))

        return impVolGuess
        
    def getDelta(self, S, K, r, vol, tau, isCall, q):
        self.delta = self.model.calc_delta(S, K, r, vol, tau, isCall, q)
        return self.delta

    def getGamma(self, S, K, r, vol, tau, isCall, q):
        self.gamma = self.model.calc_gamma(S, K, r, vol, tau, isCall, q)
        # measure in terms of tick incr
        return self.gamma

    def getTheta(self, S, K, r, vol, tau, isCall, q):
        self.theta = self.model.calc_theta(S, K, r, vol, tau, isCall, q)
        return self.theta

    def getVega(self, S, K, r, vol, tau, isCall, q):
        self.vega = self.model.calc_vega(S, K, r, vol, tau, isCall, q)
        return self.vega

    def getRho(self, S, K, r, vol, tau, isCall, q):
        self.rho = self.model.calc_rho(S, K, r, vol, tau, isCall, q)
        return self.rho

    def getVanna(self, S, K, r, vol, tau, isCall, q):
        self.vanna = self.model.calc_vanna(S, K, r, vol, tau, isCall, q)
        return self.vanna * 0.01

    def getVomma(self, S, K, r, vol, tau, isCall, q):
        self.vomma = self.model.calc_vomma(S, K, r, vol, tau, isCall, q)
        return self.vomma * 0.01 

    def getCharm(self, S, K, r, vol, tau, isCall, q):
        self.charm = self.model.calc_charm(S, K, r, vol, tau, isCall, q)
        return self.charm

    def getVeta(self, S, K, r, vol, tau, isCall, q):
        self.veta = self.model.calc_veta(S, K, r, vol, tau, isCall, q)
        return self.veta

    def getSpeed(self, S, K, r, vol, tau, isCall, q):
        self.speed = self.model.calc_speed(S, K, r, vol, tau, isCall, q)
        # measure in terms of spot tick (1 spotTickIncr for gamma, 1 spotTickIncr for speed)
        return self.speed

    def getZomma(self, S, K, r, vol, tau, isCall, q):
        self.zomma = self.model.calc_zomma(S, K, r, vol, tau, isCall, q)
        return self.zomma * 0.01

    def getColor(self, S, K, r, vol, tau, isCall, q):
        self.color = self.model.calc_color(S, K, r, vol, tau, isCall, q)
        return self.color

    def getOptionGreeks(self, S, K, r, vol, tau, isCall, q):
    
        greekDf = pd.DataFrame()
        greekDf["delta"] = [self.getDelta(S, K, r, vol, tau, isCall, q)]
        greekDf["gamma"] = [self.getGamma(S, K, r, vol, tau, isCall, q)]
        greekDf["theta"] = [self.getTheta(S, K, r, vol, tau, isCall, q)]
        greekDf["vega"] = [self.getVega(S, K, r, vol, tau, isCall, q)]
        greekDf["rho"] = [self.getRho(S, K, r, vol, tau, isCall, q)]
        greekDf["vanna"] = [self.getVanna(S, K, r, vol, tau, isCall, q)]
        greekDf["vomma"] = [self.getVomma(S, K, r, vol, tau, isCall, q)]
        greekDf["charm"] = [self.getCharm(S, K, r, vol, tau, isCall, q)]
        greekDf["veta"] = [self.getVeta(S, K, r, vol, tau, isCall, q)]
        greekDf["speed"] = [self.getSpeed(S, K, r, vol, tau, isCall, q)]
        greekDf["zomma"] = [self.getZomma(S, K, r, vol, tau, isCall, q)]
        greekDf["color"] = [self.getColor(S, K, r, vol, tau, isCall, q)]
        if isCall:
            greekDf.index = ["%sC" %(K)]
        else:
            greekDf.index = ["%sP" %(K)]

        return greekDf


