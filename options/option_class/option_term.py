"""
split a term into calls, puts
create dictionary of option
"""

import pandas as pd
from options.option_class.option_class import Option, option_class


class OptionTerm(option_class):
    def __init__(self, product, exchange):
        super().__init__(product, exchange)
        self.option = Option(product, exchange)
        self.calls = {}
        self.puts = {}

    def getStrike(self, optName):

        return optName[1:5]

    def getTerm(self, optName):

        return optName[1:5]

    def setTerm(self, optNames, exchange_symbol=None):

        self.term = self.getTerm(optNames[0])
        self.exchange_symbol = exchange_symbol
        self.strikes = []

        # add options to calls and puts dicts
        for optName in optNames:
            optType = optName[0]
            strike = self.getStrike(optName)
            self.strikes.append(strike)
            # add option instance to options dict
            if optType == "C":
                self.calls[optName] = self.option.setInstrument(optName, "C", strike, self.term,
                                                               exchange_symbol=exchange_symbol)
            else:
                self.puts[optName] = self.option.setInstrument(optName, "P", strike, self.term,
                                                               exchange_symbol=exchange_symbol)
        # get options
        self.options = {**self.calls, **self.puts}


